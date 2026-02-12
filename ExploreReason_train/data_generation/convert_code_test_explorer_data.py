import json
import random
from tqdm import tqdm

from datasets import Dataset, DatasetDict  
from src.conversation.code_prompts import SYSTEM_PROMPT_CODE, CODE_INSTRUCTION_TEMPLATE, CODE_INSTRUCTION_TEMPLATE_WITH_LIKELIHOODS, GUIDANCE_BLOCK


def get_oracle_helper_call_trace(csv_info):
        pr_delimiter = csv_info['priors'].get('Delimiter', None)
        pr_quotechar = csv_info['priors'].get('Quotechar', None)
        pr_skiprows = csv_info['priors'].get('Skiprows', None)
        pr_delimiter_max = max(pr_delimiter.values(), default=None) if pr_delimiter else 0
        pr_quotechar_max = max(pr_quotechar.values(), default=None) if pr_quotechar else 0
        pr_skiprows_max = max(pr_skiprows.values(), default=None) if pr_skiprows else 0
        oracle_trace = []
        discount_factor = csv_info["discount_factor"]
        if pr_delimiter_max <= discount_factor:
            oracle_trace.append("delimiter")
        if pr_quotechar_max <= discount_factor:
            oracle_trace.append("quotechar")
        if pr_skiprows_max <= discount_factor:
            oracle_trace.append("skiprows")
        oracle_trace_string = "_".join(oracle_trace)
        # print("oracle_trace_string:", oracle_trace_string)
        return oracle_trace_string

def csv_info_to_input(csv_info, with_prior=True):
    task_desc = csv_info['task_instruction'][0]
    discount_factor = csv_info.get("discount_factor", 1.0)
    if with_prior:
        prior_info = csv_info['calib_pred_prior']
        instruction = CODE_INSTRUCTION_TEMPLATE_WITH_LIKELIHOODS.format(
                csv_name=csv_info["meta_info"]["path"],
                prior=prior_info,
                task_description=task_desc,
                d_unit=csv_info["d_unit"],
                d_code=csv_info["d_code"],
                rho_info=f"- Note: d_code = d_unit^{csv_info['rho']}",
                guidance_block=GUIDANCE_BLOCK,
            )
    else:
        instruction = CODE_INSTRUCTION_TEMPLATE.format(
                csv_name=csv_info["meta_info"]["path"],
                task_description=task_desc,
                d_unit=csv_info["d_unit"],
                d_code=csv_info["d_code"],
                rho_info=f"- Note: d_code = d_unit^{csv_info['rho']}",
                guidance_block=GUIDANCE_BLOCK,
            ) 
    oracle_trace = get_oracle_helper_call_trace(csv_info)
    answer = csv_info['answer']
    task_id = csv_info['task_id']
    return {
        "task_id": task_id,
        "task_id_original": csv_info['task_id_original'],
        "discount_factor": discount_factor,
        "d_unit": csv_info["d_unit"],
        "d_code": csv_info["d_code"],
        "rho": csv_info["rho"],
        "prompt": instruction,
        "oracle_trace": oracle_trace,
        "answer": str(answer),
        "sampled_format": csv_info.get("sampled_format", None),
        
    }


from collections import defaultdict

def stratified_shuffle_by_rho(examples, seed=42):
    buckets = defaultdict(list)
    for ex in examples:
        buckets[ex["rho"]].append(ex)
    rng = random.Random(seed)
    for v in buckets.values():
        rng.shuffle(v)

    out = []
    # round-robin merge
    while any(buckets.values()):
        for rho in sorted(buckets.keys()):
            if buckets[rho]:
                out.append(buckets[rho].pop())
    return out


def convert_data_split(file_path, with_prior=True):
    with open(file_path, 'r') as f:
        data = json.load(f)
    new_data = []
    for d in tqdm(data, desc=f"Converting data from: {file_path}", total=len(data)):
        new_d = csv_info_to_input(d, with_prior)
        # print(new_d)
        # print("oracle_trace:", new_d["oracle_trace"])
        # if len(new_data) > 5:
        #     raise NotImplementedError("Debugging: check converted data format.")
        new_data.append(new_d)
    print(f"Converted {len(new_data)} data points from {file_path} which contains {len(data)} tasks.")
    return new_data

if __name__ == "__main__":
    random.seed(42)
    
    test_file = '../../code_explorer/data/code_explore_data/code_explorer_balanced_data/csv_explore_tasks_test_processed_rho_0.5_1.0_2.0_4.0.json'
    train_file = '../../code_explorer/data/code_explore_data/code_explorer_balanced_data/csv_explore_tasks_train_processed_rho_0.5_1.0_2.0_4.0.json'
    val_file = '../../code_explorer/data/code_explore_data/code_explorer_balanced_data/csv_explore_tasks_val_processed_sampled_rho_0.5_1.0_2.0_4.0.json'
    
    for with_prior_setting in [True, False]:
        train_converted = convert_data_split(train_file, with_prior=with_prior_setting)
        val_converted = convert_data_split(val_file, with_prior=with_prior_setting)
        test_converted = convert_data_split(test_file, with_prior=with_prior_setting) # was True

        print("Shuffling training data stratified by rho...")
        print(len(train_converted))
        train_converted = stratified_shuffle_by_rho(train_converted, seed=42)
        print("After shuffling:")
        for _ in range(20):
            print(f"task_id: {train_converted[_]['task_id']}, task_id_original: {train_converted[_]['task_id_original']}, rho: {train_converted[_]['rho']}")
            # print([ex["rho"] for ex in train_converted[:20]])


        train_dataset = Dataset.from_list(train_converted)
        val_dataset   = Dataset.from_list(val_converted)
        test_dataset  = Dataset.from_list(test_converted)

        dataset_dict = DatasetDict({
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        })

        save_path = f'../data_generation_diverse/code_test_explorer_0.5_1_2_4_balanced_formatupdate_rl_{"with" if with_prior_setting else "no"}_prior/'
        
        dataset_dict.save_to_disk(save_path)
        print(f"Saved converted dataset to {save_path}")

