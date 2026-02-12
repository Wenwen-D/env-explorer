from pathlib import Path
from typing import List, Optional
from src.data.data_types import TaskInstance, TaskResult, ConversationTurn, ExecutionResult
from src.models.base_model import BaseModel
from src.execute_code import CodeExecutor
from src.execute_code import CodeExtractor
from src.evaluation.task_evaluator import TaskEvaluator
from src.conversation.code_prompts import HELPER_DELIMITER, HELPER_QUOTECHAR, HELPER_SKIPROWS
from src.conversation.code_prompts import SYSTEM_PROMPT_CODE, CODE_INSTRUCTION_TEMPLATE, CODE_INSTRUCTION_TEMPLATE_WITH_LIKELIHOODS, GUIDANCE_BLOCK

import re
import random
import logging
import json
import os


class ConversationManagerCode:

    CODE_PATTERN = re.compile(r"```python(.*?)```", re.DOTALL)

    def __init__(self, model: BaseModel,
                 max_turns: int = 8, verbose: bool = False, logger: Optional[logging.Logger] = None,
                 base_dir: str = "env_explorer/data/code_explore_data/code_explorer_balanced_data/",
                 enable_thinking: bool = False,
                 calibrator_model = None,
                 rho: float = -1,
                 ):
        # self.discount = discount
        self.code_executor = CodeExecutor()
        self.model = model
        self.max_turns = max_turns
        self.verbose = verbose
        self.base_dir = base_dir
        self.logger = logger or logging.getLogger(__name__)
        self.calibrator_model = calibrator_model
        self.enable_thinking = enable_thinking
        self.rho = rho

    def extract_code(self, text):
        match = self.CODE_PATTERN.search(text)
        return match.group(1).strip() if match else None
    
    def parse_turn_action(self, text):
        """
        - UNIT_TESTS calls:   UNIT_TESTS: test_xxx(...), test_yyy(...)
        - CODE blocks:    ```python ... ```
        - ANSWER turn:  ANSWER: ...
        Returns:
        ("UNIT_TESTS", [list of unit test calls])
        ("CODE", code_string)
        ("ANSWER", answer_string)
        (None, None) if nothing is detected.
        """
        text = text.strip()
        # --- 1. Detect UNIT_TESTS call ---
        m = re.search(r'UNIT_TESTS:\s*(.+)$', text, flags=re.DOTALL)
        if m:
            call_text = m.group(1).strip()
            # Extract only valid test function calls matching pattern: test_xxx(...)
            valid_calls = re.findall(r'test_\w+\([^)]*\)', call_text)
            if valid_calls:
                return "UNIT_TESTS", valid_calls
            # If no valid calls found, return empty list to trigger error handling
            return "UNIT_TESTS", []

        # --- 2. Detect CODE block ---
        m = re.search(r'```python\s*(.*?)```', text, flags=re.DOTALL)
        if m:
            code = m.group(1).strip()
            return "CODE", code

        # --- 3. Detect ANSWER ---
        m = re.search(r'ANSWER:\s*(.+)$', text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            answer = m.group(1).strip()
            return "ANSWER", answer

        return None, None
    
    def unit_test_run(self, unit_test_calls: List[str], csv_info):
        '''
            Execute a helper function call and print the result.
            helper_call is a string like:
            "test_skiprows(file)"
            "test_quotechar(file)"
            "test_delimiter(file)"
        '''
        x = "UNIT_TESTS Results:\n"
        records = []
        for helper_call in unit_test_calls:
            if "test_skiprows" in helper_call:
                x += f'- {helper_call} => skiprows is {csv_info["meta_info"]["skiprows"]}\n'
                records.append('skiprows')
            elif "test_quotechar" in helper_call:
                # to_exec = HELPER_QUOTECHAR
                x += f'- {helper_call} => quotechar is {csv_info["meta_info"]["quotechar"]}\n'
                records.append('quotechar')
            elif "test_delimiter" in helper_call:
                # to_exec = HELPER_DELIMITER
                x += f'- {helper_call} => delimiter is {csv_info["meta_info"]["delimiter"]}\n'
                records.append('delimiter')
            else:
                # Unknown helper call
                x += f'- {helper_call} => ERROR: unknown helper function\nFormat: UNIT_TESTS: test_delimiter(file) OR test_quotechar(file) OR test_skiprows(file)\n'
                # return 'print("ERROR: unknown helper function\nFormat: HELPER: detect_quotechar(file) OR detect_delimiter(file) OR detect_skiprows(file)")', "helper_unknown"

        return x, records


    def is_answer(self, text):
        return "ANSWER:" in text or text.strip().lower().startswith("answer:")
        # return text.strip().lower().startswith("answer:")
    
    def calculate_correctness(self, pred, csv_info):
        answer = csv_info['answer']
        if csv_info['task_instruction'][1] == 'max':
            # string match
            return 1.0 if str(pred).strip() == str(answer).strip() else 0.0
        elif csv_info['task_instruction'][1] in ['mean', 'min']:
            try:
                pred_value = float(pred)
                answer_value = float(answer)
                return 1.0 if abs(pred_value - answer_value) < 1e-2 else 0.0
            except:
                return 0.0
        else:
            return 0.0

    def compute_discounted_reward(self, correctness, num_unit_tests, num_code_attempts, csv_info):
        d_u = csv_info.get("discount_factor", 0.9)
        d_c = csv_info["discount_factor_code"]
        reward = correctness * (d_u ** num_unit_tests) * (d_c ** num_code_attempts)
        return reward

    def oracle_helper_call_trace(self, csv_info):
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
        return oracle_trace

    def oracle_trace_match_rate(self, pred_trace, oracle_trace):
        score = 0
        pred_trace = [s.split('(')[0].removeprefix('test_') for s in pred_trace]
        for choice in ['skiprows', 'quotechar', 'delimiter']:
            if choice in oracle_trace:
                if choice in pred_trace:
                    score += 1
            else:
                if choice not in pred_trace:
                    score += 1

        return score / 3.0
    
    def run_task(self, csv_info, max_turns = 8):
        if self.rho == -1:
            # use the rho specified in each task
            assert "d_code" in csv_info and "rho" in csv_info, "Task must specify 'd_code' and 'rho' if rho=-1."
            csv_info["discount_factor_code"] = csv_info['d_code']
            rho = csv_info['rho']
        else:
            csv_info["discount_factor_code"] = csv_info['discount_factor'] ** self.rho
            rho = self.rho
        # if "rho" not in csv_info:
        #     csv_info["discount_factor_code"] = csv_info['discount_factor'] ** self.rho
        #     rho = self.rho
        # else:
        #     csv_info["discount_factor_code"] = csv_info['d_code']
        #     rho = csv_info['rho']
        res = self.run_task_default(csv_info, max_turns, calibrator_model=self.calibrator_model, rho=rho)
        return res

    def run_task_default(self, csv_info, max_turns, calibrator_model, rho):
        messages = []
        timestep = 0
        discount_factor = csv_info.get("discount_factor", 0.9)
        if discount_factor == 0.9 and self.verbose:
            print("Using default discount factor 0.9 for CSV task.")

        system_prompt = SYSTEM_PROMPT_CODE

        task_desc = csv_info['task_instruction'][0]
        pred_priors = {}
        if calibrator_model is not None:
            if type(calibrator_model) is str:
                assert calibrator_model in ["uniform", "oracle"], "Unknown calibrator model type."
                if calibrator_model == "uniform": # uniform prior
                    prior_info = r"""delimiter: {',': 0.33, ';': 0.33, '\t': 0.33}, quotechar: {'"': 0.5, "'": 0.5}, skiprows: {0: 0.50, 1: 0.50}"""
                elif calibrator_model == "oracle": # oracle prior
                    p_delim_long = csv_info['priors']['Delimiter']
                    p_quotechar_long = csv_info['priors']['Quotechar']
                    p_skiprows_long = csv_info['priors']['Skiprows']
                    # round to 4 decimal places
                    p_delim_short = {k: round(v, 4) for k, v in p_delim_long.items()}
                    p_quotechar_short = {k: round(v, 4) for k, v in p_quotechar_long.items()}
                    p_skiprows_short = {k: round(v, 4) for k, v in p_skiprows_long.items()}
                    prior_info = (
                        f"delimiter: {p_delim_short}, "
                        f"quotechar: {p_quotechar_short}, "
                        f"skiprows: {p_skiprows_short}"
                    )
            else: # predicted by calibrator
                pred_priors = calibrator_model.predict_one(csv_info['filename'])
                prior_info = (
                    f"delimiter: {pred_priors['sep']}, "
                    f"quotechar: {pred_priors['quote']}, "
                    f"skiprows: {pred_priors['skiprows']}"
                )
            
            instruction = CODE_INSTRUCTION_TEMPLATE_WITH_LIKELIHOODS.format(
                csv_name=csv_info["meta_info"]["path"],
                prior=prior_info,
                task_description=task_desc,
                d_unit=discount_factor,
                d_code=csv_info["discount_factor_code"],
                rho_info=f"- Note: d_code = d_unit^{rho}",
                guidance_block=GUIDANCE_BLOCK,
            )

        else: # no prior provided
            instruction = CODE_INSTRUCTION_TEMPLATE.format(
                csv_name=csv_info["meta_info"]["path"],
                task_description=task_desc,
                d_unit=discount_factor,
                d_code=csv_info["discount_factor_code"],
                rho_info=f"- Note: d_code = d_unit^{rho}",
                guidance_block=GUIDANCE_BLOCK,
            ) 
        # print("instruction", instruction)
        messages.append(ConversationTurn(role="system", content=system_prompt))
        messages.append(ConversationTurn(role="user", content=instruction))
        if self.verbose:
            print(f"Running CSV task with instruction: {instruction}")

        oracle_trace = self.oracle_helper_call_trace(csv_info)
        if self.verbose:
            print(f"✅ Oracle helper call trace: {oracle_trace}")
        helper_call_trace = []
        code_attempt_trace = []
        for _ in range(max_turns): # TODO: check max_turns
            timestep += 1
            # model_input = messages[-2:]
            response = self.model.generate_response(messages, enable_thinking=self.enable_thinking)
            messages.append(ConversationTurn(role="assistant", content=response))

            # Check for truncated thinking (response cut off during thinking phase)
            if "<think>" in response and "</think>" not in response:
                truncation_feedback = """Truncated response detected (`<think>` without `</think>`). The action could not be parsed.\nPlease re-send a shorter message that ends with exactly one valid action."""
                messages.append(ConversationTurn(role="user", content=truncation_feedback))
                continue

            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            # mode, code_call = self.extract_helper_code(response)
            mode, content = self.parse_turn_action(response)
            if mode == "UNIT_TESTS":
                # code, helper_call_record = self.helper_code(content)
                helper_call_trace += content
                unit_test_feedback, _ = self.unit_test_run(content, csv_info)
                if self.verbose:
                    print(f"ℹ️ Calling additional UNIT_TESTS:\n{content}")
                    if 'task_id_original' in csv_info:
                        print("path", Path(self.base_dir) / f"task_csv_{csv_info['task_id_original']}")
                    else:
                        print("path", Path(self.base_dir) / f"task_csv_{csv_info['task_id']}")
                if self.verbose:
                    print(f"📺 unit test feedback:\n{unit_test_feedback}")
                messages.append(ConversationTurn(role="user", content=unit_test_feedback))
            elif mode == "CODE":
                if self.verbose:
                    print(f"📝 Code attempt:\n{content}")
                if 'task_id_original' in csv_info:
                    result = self.code_executor.execute_code(content, Path(self.base_dir) / f"task_csv_{csv_info['task_id_original']}")
                else:
                    result = self.code_executor.execute_code(content, Path(self.base_dir) / f"task_csv_{csv_info['task_id']}")
                if self.verbose:
                    print(f"📺 Execution result:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
                feedback = f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}" # TODO: refine feedback
                messages.append(ConversationTurn(role="user", content=feedback))
                code_attempt_trace.append(content)
            elif mode == "ANSWER":
                correctness = self.calculate_correctness(content.strip(), csv_info)
                if self.verbose:
                    print(f"🏁 Final ANSWER provided: {content}, gold answer: {csv_info['answer']} => Correctness: {correctness}")
                # no recognizable output
                if correctness > 0:
                    messages.append(ConversationTurn(role="user", content=f"Your answer ({content}) is correct."))
                    reward = self.compute_discounted_reward(correctness, len(helper_call_trace), len(code_attempt_trace), csv_info)
                else:
                    reward = 0
                    messages.append(ConversationTurn(role="user", content=f"Your answer ({content[:100]}) is incorrect."))
                tt = dict(
                    task_id=csv_info['task_id'],
                    conversation=[m.model_dump() for m in messages],
                    num_turns=len([m for m in messages if m.role == "assistant"]),
                    task=csv_info['task_instruction'],
                    csv_info=csv_info,
                    reward=reward,
                    success=correctness,
                    d_u=discount_factor,
                    d_c=csv_info["discount_factor_code"],
                    rho=rho,
                    pred_answer=content.strip(),
                    pred_priors=pred_priors,
                    helper_call_trace=helper_call_trace,
                    code_attempt_trace=code_attempt_trace,
                    oracle_trace=oracle_trace,
                    oracle_trace_match_rate=self.oracle_trace_match_rate(helper_call_trace, oracle_trace),
                )
                return tt
            else:
                # no recognizable output
                messages.append(ConversationTurn(role="user", content="Your response was not recognized as UNIT_TESTS, CODE (```python...```), or ANSWER. Please follow the specified formats."))

        tt = dict(
                task_id=csv_info['task_id'],
                conversation=[m.model_dump() for m in messages],
                num_turns=len([m for m in messages if m.role == "assistant"]),
                task=csv_info['task_instruction'],
                csv_info=csv_info,
                reward=0,
                success=0,
                d_u=discount_factor,
                d_c=csv_info["discount_factor_code"],
                rho=rho,
                pred_answer="",
                pred_priors=pred_priors,
                helper_call_trace=helper_call_trace,
                code_attempt_trace=code_attempt_trace,
                oracle_trace=oracle_trace,
                oracle_trace_match_rate=self.oracle_trace_match_rate(helper_call_trace, oracle_trace),

            )
        return tt
