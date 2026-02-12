# multi_turn_rollout_code_test_explorer.py
# Copyright 2025
# Multi-turn rollout for code test explorer: call unit tests, execute code, and provide answers

import os
import copy
import json
import re
import torch
import torch.distributed
import numpy as np
from contextlib import contextmanager
from typing import Optional, Any, List, Dict, Union
from transformers import PreTrainedTokenizer
from vllm import LLM, SamplingParams
from tensordict import TensorDict
from pathlib import Path
from datetime import datetime

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig
from src.execute_code import CodeExecutor

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

def parse_turn_action(text: str):
    """
    Parse model response to extract action type and content.
    - UNIT_TESTS calls:   UNIT_TESTS: test_xxx(...), test_yyy(...)
    - CODE blocks:    ```python ... ```
    - ANSWER turn:  ANSWER: ...
    Returns:
    ("UNIT_TESTS", [list of unit test calls])
    ("CODE", code_string)
    ("ANSWER", answer_string)
    ("TRUNCATED", None) if truncated thinking detected
    (None, None) if nothing is detected.

    For RL training, truncated thinking (has <think> but not </think>) is treated as invalid format.
    In conversation_manager_code.py, this allows regeneration, but in RL we penalize it.
    """
    # Check for truncated thinking (strict penalty for RL training)
    if "<think>" in text and "</think>" not in text:
        return "TRUNCATED", None

    # Remove thinking tags if present
    if '</think>' in text:
        text = text.split('</think>')[-1].strip()
    if '<|im_end|>' in text:
        text = text.replace('<|im_end|>', '').strip()

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

def unit_test_run(unit_test_calls: List[str], csv_info: Dict[str, Any]):
    """
    Execute unit test calls and return formatted feedback.
    unit_test_calls: List of strings like ["test_skiprows(file)", "test_delimiter(file)"]
    csv_info: Dictionary containing meta_info with delimiter, quotechar, skiprows
    Returns: (feedback_string, list_of_records)
    """
    # DEBUG: Print csv_info structure
    print(f"🔍 DEBUG unit_test_run: csv_info keys = {csv_info.keys()}")
    print(f"🔍 DEBUG unit_test_run: csv_info = {csv_info}")
    if "meta_info" in csv_info:
        print(f"🔍 DEBUG unit_test_run: meta_info = {csv_info['meta_info']}")
        print(f"🔍 DEBUG unit_test_run: delimiter = {csv_info['meta_info'].get('delimiter')}")
        print(f"🔍 DEBUG unit_test_run: quotechar = {csv_info['meta_info'].get('quotechar')}")
        print(f"🔍 DEBUG unit_test_run: skiprows = {csv_info['meta_info'].get('skiprows')}")
    else:
        raise ValueError(f"❌ 'meta_info' key missing from csv_info! csv_info keys: {csv_info.keys()}")

    feedback = "UNIT_TESTS Results:\n"
    records = []
    for helper_call in unit_test_calls:
        if "test_skiprows" in helper_call:
            skiprows_value = csv_info["meta_info"]["skiprows"]
            print(f"🔍 DEBUG unit_test_run: Returning skiprows = {skiprows_value}")
            feedback += f'- {helper_call} => skiprows is {skiprows_value}\n'
            records.append('skiprows')
        elif "test_quotechar" in helper_call:
            quotechar_value = csv_info["meta_info"]["quotechar"]
            print(f"🔍 DEBUG unit_test_run: Returning quotechar = {quotechar_value}")
            feedback += f'- {helper_call} => quotechar is {quotechar_value}\n'
            records.append('quotechar')
        elif "test_delimiter" in helper_call:
            delimiter_value = csv_info["meta_info"]["delimiter"]
            print(f"🔍 DEBUG unit_test_run: Returning delimiter = {delimiter_value}")
            feedback += f'- {helper_call} => delimiter is {delimiter_value}\n'
            records.append('delimiter')
        else:
            feedback += f'- {helper_call} => ERROR: unknown unit test function\nFormat: UNIT_TESTS: test_delimiter(file) OR test_quotechar(file) OR test_skiprows(file)\n'

    print(f"🔍 DEBUG unit_test_run: Final feedback:\n{feedback}")
    return feedback, records

def calculate_correctness(pred: str, csv_info: Dict[str, Any]) -> float:
    """Calculate if the prediction matches the gold answer."""
    answer = csv_info['answer']
    task_type = csv_info['task_instruction'][1] if len(csv_info['task_instruction']) > 1 else None

    if task_type == 'max':
        # string match
        return 1.0 if str(pred).strip() == str(answer).strip() else 0.0
    elif task_type in ['mean', 'min']:
        try:
            pred_value = float(pred)
            answer_value = float(answer)
            return 1.0 if abs(pred_value - answer_value) < 1e-2 else 0.0
        except:
            return 0.0
    else:
        # fallback to string match
        return 1.0 if str(pred).strip() == str(answer).strip() else 0.0


def log_rollout_conversations(samples_info: List[Dict[str, Any]], log_dir: str = None, num_samples: int = 5) -> None:
    """
    Log multi-turn rollout conversations to a text file for debugging.

    Args:
        samples_info: List of sample info dictionaries containing sequences and actions
        log_dir: Directory to store log files
        num_samples: Number of samples to log (default: 5)
    """
    if log_dir is None:
        return

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Use a fixed filename for all steps
    log_filename = "code_test_explorer_rollout_conversations.txt"
    log_path = os.path.join(log_dir, log_filename)

    # Determine which samples to log (evenly distributed)
    total_samples = len(samples_info)
    if total_samples <= num_samples:
        samples_to_log = list(range(total_samples))
    else:
        step = total_samples // num_samples
        samples_to_log = [i * step for i in range(num_samples)]
        samples_to_log = [min(i, total_samples - 1) for i in samples_to_log]

    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(log_path)

    with open(log_path, 'a', encoding='utf-8') as f:
        # Write header only if file is new
        if not file_exists:
            f.write(f"Code Test Explorer Rollout Conversations Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        # Write step header
        f.write(f"Rollout Step - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {total_samples}, Logged samples: {len(samples_to_log)}\n")
        f.write("-" * 80 + "\n\n")

        for idx in samples_to_log:
            sample = samples_info[idx]

            f.write(f"Sample {idx}:\n")
            f.write("-" * 60 + "\n")

            # Log task info
            task_id = sample.get("task_id", "N/A")
            f.write(f"Task ID: {task_id}\n")

            # Log answer info
            gold_answer = sample.get("answer", "")
            pred_answer = sample.get("pred_answer", "")
            f.write(f"Gold Answer: {gold_answer}\n")
            f.write(f"Pred Answer: {pred_answer}\n")

            # Log discount factors and turn count
            d_unit = sample.get("d_unit", 1.0) # TODO: IS 1.0 here
            d_code = sample.get("d_code", 1.0)
            turn_count = sample.get("turn_count", 0)
            f.write(f"Discount Factor (d_unit): {d_unit}\n")
            f.write(f"Discount Factor (d_code): {d_code}\n")
            f.write(f"Number of Turns: {turn_count}\n")

            # Log action sequences
            action_seq = sample.get("action_seq", [])
            if action_seq:
                f.write(f"Action Sequences:\n")
                for turn_idx, action in enumerate(action_seq, 1):
                    f.write(f"  Turn {turn_idx}: {action}\n")

            # Log unit test traces
            unit_test_trace = sample.get("unit_test_trace", [])
            oracle_trace = sample.get("oracle_trace", [])
            f.write(f"\nUnit Test Traces: {unit_test_trace}\n")
            f.write(f"Oracle Traces: {oracle_trace}\n")

            # Log full conversation sequence
            sequence = sample.get("sequence", "")
            if sequence:
                f.write(f"\nFull Conversation Sequence:\n")
                f.write(f"{sequence}\n")

            # Log individual turn responses
            response_list = sample.get("response_list", [])
            if response_list:
                f.write(f"\nIndividual Turn Responses:\n")
                for turn_idx, response in enumerate(response_list, 1):
                    f.write(f"  Turn {turn_idx}:\n")
                    # Truncate long responses for readability
                    response_preview = response[:500] + ('...' if len(response) > 500 else '')
                    f.write(f"    {response_preview}\n")

            # Log finish reason and stop status
            finish_reason = sample.get("finish_reason", "unknown")
            stop = sample.get("stop", False)
            correctness = sample.get("correctness", 0.0)
            f.write(f"\nFinish Reason: {finish_reason}\n")
            f.write(f"Stop: {stop}\n")
            f.write(f"Correctness: {correctness}\n")

            f.write("\n" + "=" * 80 + "\n\n")

        # Add separator between steps
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF ROLLOUT STEP\n")
        f.write("=" * 80 + "\n\n\n")

    print(f"📝 Code Test Explorer rollout conversations logged to: {log_path} (logged {len(samples_to_log)} samples out of {total_samples})")


class MultiTurnRolloutCodeTestExplorer(BaseRollout):
    """
    Multi-turn rollout for code test explorer: call unit tests, execute code, and provide answers.
    At each turn, the LLM can:
    1. Call UNIT_TESTS to query CSV format properties
    2. Write CODE to attempt solving the task
    3. Provide final ANSWER based on code execution results
    """

    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        all_task_base_path: Optional[str] = None,
        enable_thinking: bool = False,
        log_dir: Optional[str] = None,
    ):
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer

        self.max_turns = getattr(config, "max_turns", 6)
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        self.pad_token_id = tokenizer.pad_token_id

        self.single_turn_response_length = getattr(config, 'single_turn_response_length', 1024)

        self.base_dir = all_task_base_path if all_task_base_path is not None else "./"
        self.code_executor = CodeExecutor()
        self.enable_thinking = enable_thinking
        self.log_dir = log_dir

        # vLLM inference engine
        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or (config.prompt_length + config.response_length) * 3,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=20000,
            max_num_seqs=10,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
        )
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": self.single_turn_response_length,
            "stop_token_ids": [self.tokenizer.eos_token_id],
            "include_stop_str_in_output": True,
            "skip_special_tokens": False,
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Code Test Explorer sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        print("Debugging: sampling params:", self.sampling_params.n, self.sampling_params)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _is_finished(self, response: str) -> bool:
        """
        Determine if the conversation should finish based on model response.
        If the model chooses ANSWER, we finish. Otherwise, continue.
        """
        action, _ = parse_turn_action(response)
        return action == "ANSWER"

    def _get_multi_turn_mask(self, response_tokens):
        """
        Generate multi-turn conversation attention mask, masking all special tokens and prompt parts.
        Only keeps assistant response content.

        Args:
            response_tokens: Token sequence containing multi-turn conversation

        Returns:
            attention_mask: Mask of same size as response_tokens, only keeping assistant response content
        """

        # Get special token IDs
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        user_id = self.tokenizer.convert_tokens_to_ids("user")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        pad_id = self.tokenizer.pad_token_id
        newline_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]  # 198

        attention_mask = torch.zeros_like(response_tokens)  # Initialize all to 0
        current_pos = 0
        in_assistant_response = True  # Initial state is True, starting from assistant response
        while current_pos < len(response_tokens):
            if response_tokens[current_pos] == im_end_id:
                # Only count <|im_end|> if we are currently inside an assistant response
                if in_assistant_response:
                    attention_mask[current_pos] = 1
                    in_assistant_response = False
                current_pos += 1
                continue

            if (current_pos + 2 < len(response_tokens) and
                response_tokens[current_pos] == im_start_id and
                response_tokens[current_pos + 1] == assistant_id and
                response_tokens[current_pos + 2] == newline_id):
                # Find new assistant response start (including newline)
                in_assistant_response = True
                current_pos += 3  # Skip im_start, assistant and newline
                continue

            if in_assistant_response and response_tokens[current_pos] != pad_id:
                # In assistant response content and not padding
                attention_mask[current_pos] = 1

            current_pos += 1

        return attention_mask

    def _get_prompts_and_indices(self, samples_info):
        """Get prompts and indices for samples that haven't stopped."""
        prompts, indices = [], []
        for _, info in enumerate(samples_info):
            if not info['stop']:
                prompts.append(info['sequence'])
                indices.append(info['index'])
        return prompts, indices

    def _multi_turn_generate(self, vllm_inputs: List[Dict[str, Any]], sampling_params=None, use_tqdm=False):
        """
        Main Code Test Explorer multi-turn rollout logic using batch processing.
        """
        sampling_params = copy.deepcopy(self.sampling_params)
        original_n = sampling_params.n
        print("Debugging: original sampling n:", original_n)

        # prepare initial samples
        print("🌐 Debug len(vllm_inputs):", len(vllm_inputs))
        new_vllm_inputs = []
        for single_vllm_input in vllm_inputs:
            print("🌐 Debug single_vllm_input keys:", single_vllm_input.keys())
            prompt = self.tokenizer.decode(single_vllm_input['prompt_token_ids'], skip_special_tokens=False)

            # Get task_id for file system lookup (without rho suffix)
            # Prefer task_id_original if available, otherwise strip rho from task_id
            task_id = single_vllm_input.get('task_id', '0')
            task_id_original = single_vllm_input.get('task_id_original', None)
            if task_id_original is None:
                # Strip rho suffix from task_id (e.g., "task_csv_b73c1f_3.0" -> "task_csv_b73c1f")
                # Check if the last part after underscore is a number (the rho value)
                parts = task_id.rsplit('_', 1)
                if len(parts) == 2:
                    try:
                        float(parts[1])  # Check if it's a number
                        task_id_original = parts[0]
                    except ValueError:
                        task_id_original = task_id
                else:
                    task_id_original = task_id

            new_vllm_inputs.extend([{
                "prompt": prompt,
                "answer": single_vllm_input.get('answer', None),
                "d_unit": single_vllm_input.get('d_unit', 1.0),
                "d_code": single_vllm_input.get('d_code', 1.0),
                "oracle_trace": single_vllm_input.get('oracle_trace', []),
                "task_id": task_id,
                "task_id_original": task_id_original,
                "csv_info": single_vllm_input.get('csv_info', {}),
            } for _ in range(sampling_params.n)])

        sampling_params.n = 1
        sampling_params.detokenize = True

        samples_info = []
        for index, item in enumerate(new_vllm_inputs):
            base_seed = getattr(self.config, 'seed', 0) or 0
            replica_offset = (index % original_n) if (original_n and original_n > 0) else 0
            per_sample_seed = int(base_seed) + int(replica_offset)

            sample_info = {
                "prompt": item["prompt"],
                "sequence": item["prompt"],
                "response": "",
                "response_list": [],  # Track individual turn responses
                "stop": False,
                "finish_reason": None,
                "index": index,
                "turn_count": 0,
                "seed": per_sample_seed,
                "answer": item.get("answer", None),
                "task_id": item.get("task_id", '0'),
                "task_id_original": item.get("task_id_original", '0'),  # For file system lookup
                "d_unit": item['d_unit'],
                "d_code": item['d_code'],
                "action_seq": [],
                "unit_test_trace": [],
                "code_attempt_trace": [],
                "oracle_trace": item.get("oracle_trace", []),
                "pred_answer": "",
                "correctness": 0.0,
                "csv_info": item.get("csv_info", {}),
            }
            samples_info.append(sample_info)

        max_turns_allowed = copy.deepcopy(self.max_turns)
        for turn_number in range(max_turns_allowed):
            input_prompts, input_indices = self._get_prompts_and_indices(samples_info)

            if not input_prompts:
                break  # All conversations have stopped

            vllm_inputs = [{
                'prompt_token_ids': self.tokenizer.encode(prompt, add_special_tokens=False)[-(self.config.prompt_length + self.config.response_length):],
            } for prompt in input_prompts]

            # Debug: Log vLLM inputs for first sample at each turn
            if turn_number < 3 and len(vllm_inputs) > 0 and self.log_dir is not None:
                try:
                    debug_log_path = os.path.join(self.log_dir, "vllm_inputs_debug.txt")
                    decoded_input = self.tokenizer.decode(vllm_inputs[0]['prompt_token_ids'], skip_special_tokens=False)
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"🔍 Turn {turn_number+1} - vLLM Input Debug (Sample 0)\n")
                        f.write(f"{'='*80}\n")
                        f.write(f"Original sequence length: {len(input_prompts[0])} chars\n")
                        f.write(f"Tokenized length: {len(vllm_inputs[0]['prompt_token_ids'])} tokens\n")
                        f.write(f"Truncation limit: {self.config.prompt_length + self.config.response_length} tokens\n")
                        f.write(f"\nDecoded vLLM input (last 1500 chars):\n")
                        f.write(f"...{decoded_input[-1500:]}\n")
                        f.write(f"{'='*80}\n\n")
                except Exception as e:
                    print(f"⚠️ Failed to write vLLM input debug log: {e}")

            # Generate responses with per-input sampling params (unique seeds per rollout)
            per_input_sampling_params = []
            for _, index in enumerate(input_indices):
                sp = copy.deepcopy(sampling_params)
                sp.seed = samples_info[index].get('seed', None)
                per_input_sampling_params.append(sp)

            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=per_input_sampling_params,
                use_tqdm=use_tqdm
            )

            # decode some outputs for debugging
            for i in range(min(2, len(outputs))):
                decoded_output = outputs[i].outputs[0].text
                print(f"  🌐 vLLM Output {i} decoded response: {decoded_output[:100]} ...")

            # Safely sort by numeric request_id if all are digits; otherwise keep original order
            try:
                req_ids = [str(getattr(o, 'request_id', '')) for o in outputs]
                if all(s.isdigit() for s in req_ids):
                    sorted_outputs = sorted(outputs, key=lambda o: int(str(getattr(o, 'request_id', '0'))))
                else:
                    sorted_outputs = outputs
            except Exception:
                sorted_outputs = outputs

            responses = [x.outputs[0].text for x in sorted_outputs]
            action_sequences = [parse_turn_action(responses[i]) for i in range(len(responses))]

            for i, index in enumerate(input_indices):
                samples_info[index]['response'] += responses[i]
                samples_info[index]['sequence'] += responses[i]
                samples_info[index]['response_list'].append(responses[i])
                samples_info[index]['finish_reason'] = sorted_outputs[i].outputs[0].finish_reason
                samples_info[index]['action_seq'].append(action_sequences[i])

                action_type, content = action_sequences[i]

                if action_type == 'UNIT_TESTS':
                    samples_info[index]['stop'] = False
                    samples_info[index]['unit_test_trace'].extend(content)

                    # DEBUG: Print what we're passing to unit_test_run
                    print(f"🔍 DEBUG: Calling unit_test_run for sample {index}")
                    print(f"🔍 DEBUG: unit_test_calls = {content}")
                    print(f"🔍 DEBUG: csv_info keys = {samples_info[index]['csv_info'].keys()}")
                    print(f"🔍 DEBUG: csv_info = {samples_info[index]['csv_info']}")

                    # Execute unit tests
                    unit_test_feedback, _ = unit_test_run(content, samples_info[index]['csv_info'])
                    print(f"🔍 DEBUG: unit_test_feedback returned:\n{unit_test_feedback}")

                    samples_info[index]['sequence'] = samples_info[index]['sequence'].strip() + '\n'
                    if self.enable_thinking:
                        samples_info[index]['sequence'] += (
                            f"<|im_start|>user\n{unit_test_feedback.strip()}\n<|im_end|>\n"
                            f"<|im_start|>assistant\n"
                        )
                    else:
                        samples_info[index]['sequence'] += (
                            f"<|im_start|>user\n{unit_test_feedback.strip()}\n<|im_end|>\n"
                            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
                        )

                elif action_type == 'CODE':
                    samples_info[index]['stop'] = False
                    samples_info[index]['code_attempt_trace'].append(content)

                    # Execute the code using task_id_original (without rho suffix) for file system lookup
                    result = self.code_executor.execute_code(content, Path(self.base_dir) / f"task_csv_{samples_info[index]['task_id_original']}")
                    code_feedback = f"[stdout]\n{result.stdout}\n[stderr]\n{result.stderr}"

                    samples_info[index]['sequence'] = samples_info[index]['sequence'].strip() + '\n'
                    if self.enable_thinking:
                        samples_info[index]['sequence'] += (
                            f"<|im_start|>user\n{code_feedback.strip()}\n<|im_end|>\n"
                            f"<|im_start|>assistant\n"
                        )
                    else:
                        samples_info[index]['sequence'] += (
                            f"<|im_start|>user\n{code_feedback.strip()}\n<|im_end|>\n"
                            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
                        )

                elif action_type == 'ANSWER':
                    samples_info[index]['stop'] = True
                    samples_info[index]['pred_answer'] = content
                    # Calculate correctness
                    correctness = calculate_correctness(content, samples_info[index]['csv_info'])
                    samples_info[index]['correctness'] = correctness

                elif action_type == 'TRUNCATED':
                    # Truncated thinking: <think> without </think>
                    # In RL training, we treat this as invalid format and stop
                    # The format_reward will be 0.0, giving no learning signal
                    samples_info[index]['stop'] = True
                    samples_info[index]['finish_reason'] = 'truncated_thinking'
                    # Leave pred_answer empty, correctness will be 0.0
                    print(f"⚠️ Sample {index}: Truncated thinking detected (<think> without </think>). Stopping conversation.")

                else:
                    # Unrecognized response (None, None from parse_turn_action)
                    # In RL training, this will also result in format_reward = 0.0
                    samples_info[index]['stop'] = True
                    samples_info[index]['finish_reason'] = 'invalid_format'
                    print(f"⚠️ Sample {index}: Unrecognized response format. Stopping conversation.")

            if turn_number == max_turns_allowed - 1:
                print("⚠️ Reached maximum allowed turns for Code Test Explorer rollout.")
                for i, index in enumerate(input_indices):
                    samples_info[index]['turn_count'] = turn_number + 1
                    samples_info[index]['stop'] = True
                    samples_info[index]['finish_reason'] = sorted_outputs[i].outputs[0].finish_reason
                break

            is_finished = [
                self._is_finished(responses[i])
                for i, index in enumerate(input_indices)
            ]
            for i, index in enumerate(input_indices):
                samples_info[index]['turn_count'] = turn_number + 1

            if all(is_finished):  # All samples finished
                for i, index in enumerate(input_indices):
                    samples_info[index]['turn_count'] = turn_number + 1
                    samples_info[index]['stop'] = True
                    samples_info[index]['finish_reason'] = sorted_outputs[i].outputs[0].finish_reason
                break

            for i, index in enumerate(input_indices):
                if is_finished[i]:
                    samples_info[index]['stop'] = True
                    samples_info[index]['finish_reason'] = sorted_outputs[i].outputs[0].finish_reason

        responses = [sample_info['response'] for sample_info in samples_info]
        sequences = [sample_info['sequence'] for sample_info in samples_info]
        action_seqs = [sample_info['action_seq'] for sample_info in samples_info]
        unit_test_traces = [sample_info['unit_test_trace'] for sample_info in samples_info]
        code_attempt_traces = [sample_info['code_attempt_trace'] for sample_info in samples_info]
        oracle_traces = [sample_info['oracle_trace'] for sample_info in samples_info]
        d_unit_list = [sample_info['d_unit'] for sample_info in samples_info]
        d_code_list = [sample_info['d_code'] for sample_info in samples_info]
        pred_answers = [sample_info['pred_answer'] for sample_info in samples_info]
        gold_answers = [sample_info['answer'] for sample_info in samples_info]
        correctness_list = [sample_info['correctness'] for sample_info in samples_info]
        response_lists = [sample_info['response_list'] for sample_info in samples_info]

        print(f"🌐 Completed multi-turn Code Test Explorer rollout for batch. (_multi_turn_generate), len(responses)={len(responses)}, len(sequences)={len(sequences)}, len(action_seqs)={len(action_seqs)}")
        for jj in range(min(2, len(samples_info))):
            print(f"  🌐 Final Sample {jj} pred_answer: {pred_answers[jj]}")
            print(f"  🌐 Final Sample {jj} gold_answer: {gold_answers[jj]}")
            print(f"  🌐 Final Sample {jj} correctness: {correctness_list[jj]}")

        # Log rollout conversations for debugging
        if self.log_dir is not None:
            try:
                log_rollout_conversations(samples_info, log_dir=self.log_dir, num_samples=5)
            except Exception as e:
                print(f"⚠️ Failed to log rollout conversations: {e}")

        return responses, sequences, action_seqs, unit_test_traces, code_attempt_traces, oracle_traces, d_unit_list, d_code_list, pred_answers, gold_answers, correctness_list, response_lists

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Entry point — generate Code Test Explorer rollouts from a batch of prompts.
        """
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch

        # DEBUG: Check what keys are actually in non_tensor_batch
        print(f"🔍 DEBUG generate_sequences: non_tensor_batch keys = {non_tensor_batch.keys()}")
        print(f"🔍 DEBUG generate_sequences: batch_size = {batch_size}")

        # Check for critical keys
        print(f"🔍 DEBUG generate_sequences: 'd_unit' in non_tensor_batch? {('d_unit' in non_tensor_batch)}")
        print(f"🔍 DEBUG generate_sequences: 'd_code' in non_tensor_batch? {('d_code' in non_tensor_batch)}")
        print(f"🔍 DEBUG generate_sequences: 'sampled_format' in non_tensor_batch? {('sampled_format' in non_tensor_batch)}")

        answer_list = non_tensor_batch.get("answer", None)
        d_unit_list = non_tensor_batch.get("d_unit", None)
        d_code_list = non_tensor_batch.get("d_code", None)
        oracle_trace_list = non_tensor_batch.get("oracle_trace", None)
        task_id_list = non_tensor_batch.get("task_id", None)

        # DEBUG: Print what we got
        print(f"🔍 DEBUG generate_sequences: d_unit_list = {d_unit_list}")
        print(f"🔍 DEBUG generate_sequences: d_code_list = {d_code_list}")

        # Assert if critical keys are missing
        if d_unit_list is None:
            print(f"⚠️ WARNING: d_unit_list is None! Using default value 1.0")
        if d_code_list is None:
            print(f"⚠️ WARNING: d_code_list is None! Using default value 1.0")
        if "sampled_format" not in non_tensor_batch:
            print(f"❌ ERROR: 'sampled_format' key is missing from non_tensor_batch!")
            print(f"❌ Available keys: {list(non_tensor_batch.keys())}")
            # Don't raise yet, let's see if it's actually in the data

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not working properly.")

        # Build csv_info dict for each sample (needed for unit_test_run and calculate_correctness)
        # This requires reading from non_tensor_batch
        csv_info_list = []

        # DEBUG: Check what keys are in non_tensor_batch
        print(f"🔍 DEBUG: non_tensor_batch keys: {non_tensor_batch.keys()}")
        if "sampled_format" in non_tensor_batch:
            print(f"🔍 DEBUG: sampled_format exists, length: {len(non_tensor_batch['sampled_format'])}")
            print(f"🔍 DEBUG: First sampled_format sample: {non_tensor_batch['sampled_format'][0]}")
        else:
            print(f"⚠️ WARNING: 'sampled_format' NOT in non_tensor_batch!")
            print(f"⚠️ Available keys: {list(non_tensor_batch.keys())}")

        for idx in range(batch_size):
            # Extract CSV format info from sampled_format if available
            sampled_format = None
            if "sampled_format" in non_tensor_batch:
                sampled_format = non_tensor_batch["sampled_format"][idx]
                print(f"🔍 DEBUG sample {idx}: sampled_format = {sampled_format}")
                print(f"🔍 DEBUG sample {idx}: type(sampled_format) = {type(sampled_format)}")

                # Assert sampled_format is not None
                if sampled_format is None:
                    raise ValueError(f"❌ sampled_format is None for sample {idx}!")

                # Check if it's a dict and has expected keys
                if isinstance(sampled_format, dict):
                    print(f"🔍 DEBUG sample {idx}: sampled_format keys = {sampled_format.keys()}")
                    print(f"🔍 DEBUG sample {idx}: Delimiter = {sampled_format.get('Delimiter')}")
                    print(f"🔍 DEBUG sample {idx}: Quotechar = {sampled_format.get('Quotechar')}")
                    print(f"🔍 DEBUG sample {idx}: Skiprows = {sampled_format.get('Skiprows')}")
            else:
                raise ValueError(f"❌ 'sampled_format' key missing from non_tensor_batch!")

            csv_info = {
                "answer": answer_list[idx] if answer_list is not None else None,
                "task_id": task_id_list[idx] if task_id_list is not None else '0',
                "task_instruction": non_tensor_batch.get("task_instruction", [None] * batch_size)[idx] if "task_instruction" in non_tensor_batch else ["", ""],
                "meta_info": {
                    "delimiter": sampled_format.get("Delimiter") if sampled_format else None,
                    "quotechar": sampled_format.get("Quotechar") if sampled_format else None,
                    "skiprows": sampled_format.get("Skiprows") if sampled_format else None,
                }
            }

            # Assert that meta_info values are not None
            if csv_info["meta_info"]["delimiter"] is None:
                raise ValueError(f"❌ delimiter is None for sample {idx}! sampled_format was: {sampled_format}")
            if csv_info["meta_info"]["quotechar"] is None:
                raise ValueError(f"❌ quotechar is None for sample {idx}! sampled_format was: {sampled_format}")
            if csv_info["meta_info"]["skiprows"] is None:
                raise ValueError(f"❌ skiprows is None for sample {idx}! sampled_format was: {sampled_format}")

            print(f"✅ DEBUG sample {idx}: csv_info meta_info = {csv_info['meta_info']}")
            csv_info_list.append(csv_info)

        vllm_inputs = []
        popped_raw = non_tensor_batch.pop("raw_prompt_ids")
        for idx, raw_prompt_ids in enumerate(popped_raw):
            vllm_inputs.append({
                "prompt_token_ids": list(raw_prompt_ids),
                "answer": answer_list[idx] if answer_list is not None else None,
                "d_unit": d_unit_list[idx] if d_unit_list is not None else 1.0,
                "d_code": d_code_list[idx] if d_code_list is not None else 1.0,
                "oracle_trace": oracle_trace_list[idx] if oracle_trace_list is not None else [],
                "task_id": task_id_list[idx] if task_id_list is not None else '0',
                "csv_info": csv_info_list[idx],
            })

        # DEBUG: Verify csv_info was properly added to vllm_inputs
        print(f"🔍 DEBUG: Created {len(vllm_inputs)} vllm_inputs")
        if len(vllm_inputs) > 0:
            print(f"🔍 DEBUG: First vllm_input csv_info = {vllm_inputs[0]['csv_info']}")

        print(f"🌐 Starting multi-turn Code Test Explorer rollout for batch size = {batch_size}.")
        for i in range(min(2, len(vllm_inputs))):
            decoded_prompt = self.tokenizer.decode(vllm_inputs[i]['prompt_token_ids'], skip_special_tokens=False)
            print(f"  🌐 vLLM Input {i} decoded prompt: {decoded_prompt[:200]} ...")

        with self.update_sampling_params(**prompts.meta_info):
            responses, sequences, action_seqs, unit_test_traces, code_attempt_traces, oracle_traces, d_unit_list, d_code_list, pred_answers, gold_answers, correctness_list, response_lists = self._multi_turn_generate(
                vllm_inputs=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )

            # Handle sampling parameter n > 1
            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)

                # Repeat non-tensor fields to match expanded batch size
                # Need to repeat all existing non_tensor_batch fields except those that will be overwritten
                fields_to_skip = {"action_seqs", "unit_test_traces", "code_attempt_traces", "oracle_traces",
                                  "d_unit_list", "d_code_list", "pred_answers", "gold_answers",
                                  "correctness_list", "response_list", "full_sequences", "raw_prompt_ids"}
                for key in non_tensor_batch.keys():
                    if key not in fields_to_skip:
                        non_tensor_batch[key] = _repeat_interleave(non_tensor_batch[key], self.sampling_params.n)

        non_tensor_batch["action_seqs"] = action_seqs
        non_tensor_batch["unit_test_traces"] = unit_test_traces
        non_tensor_batch["code_attempt_traces"] = code_attempt_traces
        non_tensor_batch["oracle_traces"] = oracle_traces
        non_tensor_batch["d_unit_list"] = d_unit_list
        non_tensor_batch["d_code_list"] = d_code_list
        non_tensor_batch["pred_answers"] = pred_answers
        non_tensor_batch["gold_answers"] = gold_answers
        non_tensor_batch["correctness_list"] = correctness_list
        non_tensor_batch["response_list"] = response_lists
        non_tensor_batch["full_sequences"] = sequences

        non_tensor_batch["raw_prompt_ids"] = [
            self.tokenizer.encode(sequence, add_special_tokens=False)[:self.config.prompt_length + self.config.response_length]
            for sequence in sequences
        ]

        valid_prompt_len = torch.sum(attention_mask, dim=-1)
        response_ids = []
        response_mask = []
        response_position_ids = []
        model_inputs = []
        multi_turn_mask = []
        for idx, prompt_len in enumerate(valid_prompt_len):
            # Process sequence with tokenizer
            inputs = self.tokenizer(
                text=sequences[idx],
                add_special_tokens=False,
                return_tensors="pt"
            )

            resp_end = prompt_len + self.config.response_length
            response_ids.append(inputs['input_ids'][0][prompt_len:resp_end])
            response_mask.append(inputs['attention_mask'][0][prompt_len:resp_end])

            seq_len = inputs['input_ids'][0].size(0)
            new_position_ids = torch.arange(seq_len, device=inputs['input_ids'][0].device).unsqueeze(0)

            pad_position_ids = VF.pad_sequence_to_length(
                new_position_ids[:, prompt_len:resp_end],
                max_seq_len=self.config.response_length,
                pad_token_id=0,
                left_pad=False
            ).to(input_ids.device)
            response_position_ids.append(pad_position_ids)

            tmp_multi_turn_mask = self._get_multi_turn_mask(inputs['input_ids'][0][prompt_len:resp_end])
            multi_turn_mask.append(tmp_multi_turn_mask)

            # Prepare model inputs
            inputs.pop('input_ids')
            inputs.pop('attention_mask')
            model_inputs.append(dict(inputs))

        # Pad response IDs
        response_ids = VF.pad_2d_list_to_length(
            response_ids, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)

        response_length = response_ids.size(1)
        batch_size = position_ids.size(0)

        # Create a simple offset sequence [1..resp_len] and add to last prompt pos
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).view(1, -1)
        response_position_ids = position_ids[:, -1:].clone() + delta_position_id

        assert response_position_ids.ndim == 2, f"Expected 2D response_position_ids, got {response_position_ids.shape}"

        # Concatenate prompt + response positions
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)

        response_mask = VF.pad_2d_list_to_length(
            response_mask, 0, max_length=self.config.response_length
        ).to(input_ids.device)

        multi_turn_mask = VF.pad_2d_list_to_length(
            multi_turn_mask, 0, max_length=self.config.response_length
        ).to(input_ids.device)

        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # Print statistics
        valid_lengths = torch.sum(attention_mask, dim=1)
        max_valid_length = torch.max(valid_lengths).cpu()
        min_valid_length = torch.min(valid_lengths).cpu()
        avg_valid_length = torch.mean(valid_lengths.float()).cpu()

        # Create final batch
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
                "multi_turn_mask": multi_turn_mask,
            },
            batch_size=batch_size,
        )

        for key, value in non_tensor_batch.items():
            if not isinstance(value, np.ndarray):
                non_tensor_batch[key] = np.array(value, dtype=object)

        if "multi_modal_data" in non_tensor_batch:
            non_tensor_batch.pop("multi_modal_data", None)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)