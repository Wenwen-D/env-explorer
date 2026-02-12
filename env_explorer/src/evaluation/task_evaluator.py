# src/evaluation/task_evaluator.py
from typing import List, Dict, Any
from collections import Counter
from src.data.data_types import TaskResult
from src.data.data_types import ExecutionResult
import os
import json
import csv
import re
from collections import defaultdict
import difflib
from thefuzz import fuzz    
import numpy as np

from src.execute_code import CodeExecutor
from src.execute_code import CodeExtractor
# from vllm.tests import metrics

def compute_ece(conf, acc, n_bins = 10, threshold = 0.9):
    if threshold:
        acc = [1 if a >= threshold else 0 for a in acc]
    conf = np.asarray(conf)
    acc = np.asarray(acc)
    N = len(conf)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(conf, bins) - 1
    ece = 0
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            bin_acc = np.mean(acc[mask])
            bin_conf = np.mean(conf[mask])
            bin_size = np.sum(mask)
            ece += (bin_size / N) * abs(bin_acc - bin_conf)
    return ece


class TaskEvaluator:
    def __init__(self):
        """Initialize the evaluator with necessary components"""
        self.code_executor = CodeExecutor()
        self.code_extractor = CodeExtractor()

    def get_id_man_cons_keys(self, instruction):
        matches = re.findall(r'`([^`]+)`', instruction)
        if '.csv' in matches[-1]:
            keys = matches[-4:-1] if len(matches) >= 4 else []
        else:
            keys = matches[-5:-2] if len(matches) >= 5 else []
        # print(f"🔑 Found keys: {keys} 🔑")
        return keys

    def check_csv_acc(self, task_id, task_base_dir, task_instruction, clean_csv=False):
        def read_and_normalize_csv(file_path, key_field="id"):
            records = {}
            # print(f"Reading and normalizing CSV file: {file_path}")
            with open(file_path, mode='r', newline='', encoding='utf-8') as infile:
                # csv.DictReader automatically uses the first row as keys
                reader = csv.DictReader(infile) # TODO: delimiter is the default ','
                # Check for empty file
                if not reader.fieldnames:
                    return {}
                # for row in reader:
                #     # To make comparison independent of column order, we can
                #     # convert each row dictionary to a sorted JSON string.
                #     # This creates a consistent, comparable representation of the row.
                #     records.append(json.dumps(row, sort_keys=True))
                for row_num, row in enumerate(reader, 1):
                    # key_value = self.clean_string(row.get(key_field, ''))
                    key_value = row.get(key_field, '')
                    if not key_value:
                        print(f"Warning: Row {row_num} has empty key field '{key_field}', skipping")
                        continue
                    # Handle duplicate keys by appending a suffix
                    original_key = key_value
                    counter = 1
                    while key_value in records:
                        key_value = f"{original_key}_{counter}"
                        counter += 1
                    records[key_value] = row
            return records

        def concat_dict_to_str(v, keys):
            id_value = v.get(keys[0], 'EMPTY') if len(keys) > 0 else 'EMPTY'
            man_value = v.get(keys[1], 'EMPTY') if len(keys) > 1 else 'EMPTY'
            cons_value = v.get(keys[2], 'EMPTY') if len(keys) > 2 else 'EMPTY'
            s = f"{id_value} {man_value} {cons_value}"
            return s.lower()
        
        if 'onefile' in task_base_dir or 'f1_' in task_base_dir.lower():
            combined_csv = f"{task_base_dir}/result_user.csv"
        else:
            combined_csv = f"{task_base_dir}/combined_user.csv"
        # print(f"Reading and normalizing CSV file: {combined_csv}")
        if not os.path.exists(combined_csv):
            return False, None
        gt_csv = f"{task_base_dir}/task_{task_id}_gt.csv"
        

        id_key, man_key, cons_key = self.get_id_man_cons_keys(task_instruction)
        pred = read_and_normalize_csv(combined_csv, id_key)
        gold = read_and_normalize_csv(gt_csv, id_key)
        pred_concat_results = {k: concat_dict_to_str(v, [id_key, man_key, cons_key]) for k, v in pred.items()}
        gold_concat_results = {k: concat_dict_to_str(v, [id_key, man_key, cons_key]) for k, v in gold.items()}

        total_count = len(gold_concat_results)
        pred_score = 0
        missing_records = 0
        
        for k, v in gold_concat_results.items():
            if k in pred_concat_results:
                score = fuzz.ratio(v, pred_concat_results[k])
                pred_score += score / 100.0
            else:
                missing_records += 1
        csv_accuracy = pred_score / total_count if total_count > 0 else 0.0
        assert csv_accuracy <= 1.0
        csv_missing_rate = missing_records / total_count if total_count > 0 else 0.0
        # pred_set = set(pred)
        # ground_truth_set = set(gold)
        # matches = len(ground_truth_set.intersection(pred_set))/ len(ground_truth_set) if ground_truth_set else 0
        # extra_in_pred = len(pred_set.difference(ground_truth_set)) / len(ground_truth_set) if ground_truth_set else 0
        # missing_from_ground_truth = len(ground_truth_set.difference(pred_set)) / len(ground_truth_set) if ground_truth_set else 0

        stats = {
                "total_records_in_ground_truth": total_count,
                "total_records_in_pred": len(pred_concat_results),
                "csv_accuracy": csv_accuracy,
                "csv_recall_rate": 1 - csv_missing_rate,
            }
        if clean_csv:
            os.remove(combined_csv)
        return True, stats
  
    def parse_error_type(self, stderr_output):
        """Extract the error type from stderr output"""
        if not stderr_output:
            return None
        
        # Common Python exceptions to look for
        error_patterns = [
            r'(\w*Error):', r'(\w*Exception):', r'(\w*Warning):'
        ]
        
        # Look for the last line that contains an error (usually the main error)
        lines = stderr_output.strip().split('\n')
        for line in reversed(lines):
            for pattern in error_patterns:
                match = re.search(pattern, line)
                if match:
                    error_name = match.group(1)
                    # Filter out generic words that might match but aren't errors
                    if error_name not in ['Error', 'Exception', 'Warning']:
                        return error_name
        
        # Special handling for syntax errors which might not follow the pattern
        if 'SyntaxError' in stderr_output or 'invalid syntax' in stderr_output:
            return 'SyntaxError'
        elif 'IndentationError' in stderr_output:
            return 'IndentationError'
        elif 'TabError' in stderr_output:
            return 'TabError'

        return 'UnknownError'

    def count_turns(self, conversations):
        conv_a = [d for d in conversations if d.role=='assistant']
        total_turns = len(conv_a)
        l_explore = []
        l_draft = []
        l_other = []
        for d in conv_a:
            if '<explore>' in d.content and '</explore>' in d.content:
                l_explore.append(d)
            elif '<draft>' in d.content and '</draft>' in d.content:
                l_draft.append(d)
            elif '```python' in d.content and '```' in d.content:
                l_other.append(d)
        # conv_explore = [d for d in conv_a if '<explore>' in d.content and '</explore>' in d.content]
        # conv_draft = [d for d in conv_a if '<draft>' in d.content and '</draft>' in d.content]

        return len(conv_a), len(l_explore), len(l_draft), len(l_other)

    def get_first_chars(self, task_id, task_base_dir):
        # file_dir = f"{task_base_dir}/task_{task_id}/"
        file_dir = task_base_dir
        file_list = ['users.csv', 'records.jsonl', 'profiles.json']
        first_chars = {}
        for file_name in file_list:
            if not os.path.exists(f'{file_dir}/{file_name}'):
                continue
            with open(f'{file_dir}/{file_name}', 'r', encoding='utf-8') as f:
                first_c = f.read(20)
                first_chars[file_name] = first_c
        return first_chars

    ##### is exploration used
    def print_fields_in_stdout(self, explore_execution_result, task_base_dir, task_instruction):

        all_std_out = ""
        for exec_result in explore_execution_result:
            if exec_result.stdout:
                all_std_out += exec_result.stdout + "\n"


        matches = re.findall(r'`([^`]+)`', task_instruction)
        if '.csv' in matches[-1]:
            file_names = matches[:-4] if len(matches) >= 4 else []
        else:
            file_names = matches[:-5] if len(matches) >= 5 else []
        
        # json file:
        json_file = f"{task_base_dir}/{file_names[-1]}"

        # print(f"📄 File names: {file_names}; 📄 json_file: {json_file}")

        # get field names of json_file
        if json_file.endswith('.json'):
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        field_names = json_data[0].keys() if json_data else []
                    elif isinstance(json_data, dict):
                        field_names = json_data.keys()
                    else:
                        field_names = []
                field_names = list(field_names)
        elif json_file.endswith('.jsonl'):
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    field_names = set()
                    for line in f:
                        json_data = json.loads(line)
                        if isinstance(json_data, dict):
                            field_names.update(json_data.keys())
                    field_names = list(field_names)
        elif json_file.endswith('.csv'):
            return None, len(all_std_out)
        else:
            raise ValueError(f"Unsupported file type: {json_file}")
        
        field_all_printed = True
        # print(f"line 185 at task_evaluator.py: Field names in {json_file}: {field_names}") # TODO:
        for field in field_names:
            if field.lower() not in all_std_out.lower():
                field_all_printed = False
                break
        return field_all_printed, len(all_std_out)

    def has_checked_before_final(self, task_result: TaskResult) -> bool: # TODO: 
        check_template = "Given the information provided, I can draft the solution."
        check_id = None
        final_id = None

        final_improve = False
        has_checked = False
        if not task_result.final_code:
            final_improve = False
        assistant_turns = [turn.content for turn in task_result.conversation if turn.role == 'assistant']
        assert len(assistant_turns) == len(task_result.execution_results), "Number of assistant turns and execution results should match"
        assistant_exec_dict = {i: {'assistant': assistant_turns[i], 'execution_result': task_result.execution_results[i]} for i in range(len(assistant_turns))}
        for i, turn in assistant_exec_dict.items():
            if check_template in turn['assistant']:
                check_id = i
                has_checked = True
                if not turn['execution_result'].success:
                    final_improve = True
                    return final_improve, has_checked
                else:
                    return False, True
        return None, False
    
    def has_draft_before_final(self, task_result: TaskResult) -> bool:
        check_template_1 = "<draft>"
        check_template_2 = "</draft>"

        final_improve = False
        has_checked = False
        if not task_result.final_code:
            final_improve = False
        assistant_turns = [turn.content for turn in task_result.conversation if turn.role == 'assistant']
        assert len(assistant_turns) == len(task_result.execution_results), "Number of assistant turns and execution results should match"
        assistant_exec_dict = {i: {'assistant': assistant_turns[i], 'execution_result': task_result.execution_results[i]} for i in range(len(assistant_turns))}
        for i, turn in assistant_exec_dict.items():
            if check_template_1 in turn['assistant'] and check_template_2 in turn['assistant']:
                check_id = i
                has_checked = True
                if not turn['execution_result'].success:
                    final_improve = True
                    return final_improve, has_checked
                else:
                    return False, True
        return None, False





    def evaluate_batch(self, results: List[TaskResult], target_base_dir: str = None, system_choice: str = 'draft') -> Dict[str, Any]:
        # id: task_result, # turn, executable, csv acc (csv matched, total, missing), has print fields, # chars printed, # has check before final, # is explore helpful
        metrics_results = {}
        results_with_metrics = []
        for result in results:
            task_id = result.task_id
            if 'task_f1' in str(task_id).lower():
                result_csv = f"{target_base_dir}/task_{task_id}/result_user.csv"
            else:
                result_csv = f"{target_base_dir}/task_{task_id}/combined_user.csv"
            num_turns, num_explore, num_draft, num_other = self.count_turns(result.conversation)
            executable = result.success
            _, csv_stats = self.check_csv_acc(task_id, f"{target_base_dir}/task_{task_id}", result.instruction)
            # stats = {
            #     "total_records_in_ground_truth": total_count,
            #     "total_records_in_pred": len(pred_concat_results),
            #     "csv_accuracy": csv_accuracy,
            #     "csv_missing_rate": csv_missing_rate,
            # }
            has_printed_all_fields, stdout_len = self.print_fields_in_stdout(result.execution_results, f"{target_base_dir}/task_{task_id}", result.instruction)
            if 'draft' in system_choice:
                final_improve, has_checked = self.has_draft_before_final(result)
            elif 'cost' in system_choice:
                final_improve, has_checked = None, None # TODO: to be implemented
            else:
                final_improve, has_checked = self.has_checked_before_final(result)
            metrics_results[task_id] = {
                'task_result': result.model_dump(),
                'num_turns': num_turns,
                'num_explore': num_explore,
                'num_draft': num_draft,
                'num_other': num_other,
                'executable': executable,
                'csv_accuracy': csv_stats['csv_accuracy'] if csv_stats else 0,
                'csv_recall_rate': csv_stats['csv_recall_rate'] if csv_stats else 0,
                'has_printed_all_fields': has_printed_all_fields,
                'stdout_len': stdout_len,
                'has_checked_before_final': has_checked, # True or False
                'final_improve_over_check': final_improve # None if no check, True if improved, False if not improved or no final
            }
            
            results_with_metrics.append(metrics_results[task_id])

        # raise ValueError("Check metrics_results")
        def get_mean_metrics(metrics_results):
            def avg_not_None(l, default=None):
                l = [x if x is not None else default for x in l]
                try:
                    return sum(x for x in l if x is not None) / len([x for x in l if x is not None]) if l else 0
                except ZeroDivisionError:
                    print("ZERODIVVV")
                # return sum(x for x in l if x is not None) / len([x for x in l if x is not None]) if l else 0
            # print(metrics_results)
            num_turns = [i['num_turns'] for i in metrics_results.values()]
            num_explore = [i['num_explore'] for i in metrics_results.values()]
            num_draft = [i['num_draft'] for i in metrics_results.values()]
            num_other = [i['num_other'] for i in metrics_results.values()]
            execution_rate = [1 if i['executable'] else 0 for i in metrics_results.values()]
            print(f"🔥🔥🔥LEN EXEC,{len(execution_rate)}, {execution_rate}")
            csv_accuracy = [i['csv_accuracy'] for i in metrics_results.values()]
            print(f"🔥🔥🔥🔥LEN {len(csv_accuracy)}")
            print(type(csv_accuracy), len(csv_accuracy), csv_accuracy)
            # print(f"csv_accuracy: {csv_accuracy}")
            # raise ValueError("Check csv_accuracy")
            csv_recall_rate = [i['csv_recall_rate'] for i in metrics_results.values()]
            has_printed_all_fields = [i['has_printed_all_fields'] for i in metrics_results.values()]
            stdout_len = [i['stdout_len'] for i in metrics_results.values()]
            has_checked_before_final = [i['has_checked_before_final'] for i in metrics_results.values()]
            final_improve_over_check = [i['final_improve_over_check'] for i in metrics_results.values()]
            return {
                "num_turns": avg_not_None(num_turns), # not None
                "num_explore": avg_not_None(num_explore), # not None
                "num_draft": avg_not_None(num_draft), # not None
                "num_other": avg_not_None(num_other), # not None
                "execution_rate": avg_not_None(execution_rate, default=0), # 0 or 1
                "csv_accuracy": avg_not_None(csv_accuracy, default=0), # 0 if None, else 0~1
                "csv_recall_rate": avg_not_None(csv_recall_rate, default=0), # 0 if None, else 0~1
                "has_printed_all_fields": avg_not_None(has_printed_all_fields, default=None), # 0 or 1 or None if source is csv file only
                "stdout_len": avg_not_None(stdout_len), # > 0
                "has_checked_before_final": avg_not_None(has_checked_before_final, default=0), # True or False
                "final_improve_over_check": avg_not_None(final_improve_over_check, default=None), # None if no check, True if improved, False if not improved or no final
            }
        # raise ValueError("Check metrics_results")
        aggregated_final_results = get_mean_metrics(metrics_results)
        # raise ValueError("Check metrics_results")
        f1_res = {k:v for k, v in metrics_results.items() if str(k).lower().startswith('f1_')}
        f2_res = {k:v for k, v in metrics_results.items() if str(k).lower().startswith('f2_')}
        f3_res = {k:v for k, v in metrics_results.items() if str(k).lower().startswith('f3_')}
        print(f"Total Tasks: {len(metrics_results)}")
        print(f"F1 Tasks: {len(f1_res)}, F2 Tasks: {len(f2_res)}, F3 Tasks: {len(f3_res)}")
        f1_final_results = get_mean_metrics(f1_res)
        f2_final_results = get_mean_metrics(f2_res)
        f3_final_results = get_mean_metrics(f3_res)
        print(f"Aggregated Final Results: {aggregated_final_results}")
        print(f"📁 F1 Final Results: {f1_final_results}")
        print(f"📁 F2 Final Results: {f2_final_results}")
        print(f"📁 F3 Final Results: {f3_final_results}")

        i0_res = {k:v for k, v in metrics_results.items() if str(k).lower().endswith('_i0')}
        i1_res = {k:v for k, v in metrics_results.items() if str(k).lower().endswith('_i1')}
        i2_res = {k:v for k, v in metrics_results.items() if str(k).lower().endswith('_i2')}
        print(f"I0 Tasks: {len(i0_res)}, I1 Tasks: {len(i1_res)}, I2 Tasks: {len(i2_res)}")
        i0_final_results = get_mean_metrics(i0_res)
        i1_final_results = get_mean_metrics(i1_res)
        i2_final_results = get_mean_metrics(i2_res)
        print(f"ℹ️ I0 Final Results: {i0_final_results}")
        print(f"ℹ️ I1 Final Results: {i1_final_results}")
        print(f"ℹ️ I2 Final Results: {i2_final_results}") 

        return aggregated_final_results, metrics_results, f1_final_results, f2_final_results, f3_final_results, i0_final_results, i1_final_results, i2_final_results, results_with_metrics

    
    def _analyze_errors(self, results: List[TaskResult]) -> Dict[str, int]:
        """Analyze error types from execution results"""
        error_types = []
        for result in results:
            for exec_result in result.execution_results:
                if not exec_result.success and exec_result.error_type:
                    error_types.append(exec_result.error_type)
        
        return dict(Counter(error_types))