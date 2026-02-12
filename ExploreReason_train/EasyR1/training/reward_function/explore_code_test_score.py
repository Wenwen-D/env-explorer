import re
import math
import os
import json
from datetime import datetime
import numpy as np
from typing import Any, Dict, List, Tuple, Optional


def log_code_test_explorer_rewards(reward_inputs: List[Dict[str, Any]], results: List[Dict[str, float]],
                              log_dir: str = "PATH_TO/reward_logs",
                              num_samples: int = 5,
                              full_response: bool = False,
                              preserve_special_chars: bool = False) -> None:
    """
    Log model responses and computed rewards to a text file for debugging.

    Args:
        reward_inputs: List of reward input dictionaries containing model responses
        results: List of computed reward results
        log_dir: Directory to store log files
        num_samples: Number of samples to log (default: 5)
        full_response: If True, log full responses without truncation (default: False)
        preserve_special_chars: If True, use repr() to preserve special characters like \n, \t (default: False)
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Use a fixed filename for all steps
    log_filename = "code_test_explorer_rewards.txt"
    log_path = os.path.join(log_dir, log_filename)

    # Determine which samples to log (evenly distributed)
    total_samples = len(reward_inputs)
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
            f.write(f"Code Test Explorer Rewards Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

        # Write step header
        f.write(f"Step Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {total_samples}, Logged samples: {len(samples_to_log)}\n")
        f.write("-" * 80 + "\n\n")

        for idx in samples_to_log:
            reward_input = reward_inputs[idx]
            result = results[idx]

            f.write(f"Sample {idx}:\n")
            f.write("-" * 60 + "\n")

            # Log task info
            task_id = reward_input.get("task_id", "N/A")
            f.write(f"Task ID: {task_id}\n")

            # Log ground truth and prediction
            gold_answer = reward_input.get("gold_answers", "")
            pred_answer = reward_input.get("pred_answers", "")
            f.write(f"Gold Answer: {gold_answer}\n")
            f.write(f"Pred Answer: {pred_answer}\n")

            # Log discount factors and action sequences
            d_unit = reward_input.get("d_unit_list", 1.0)
            d_code = reward_input.get("d_code_list", 1.0)
            action_seqs = reward_input.get("action_seqs", [])
            f.write(f"Discount Factor (d_unit): {d_unit}\n")
            f.write(f"Discount Factor (d_code): {d_code}\n")
            f.write(f"Number of Turns: {len(action_seqs)}\n")

            # Log action sequences
            if action_seqs is not None and len(action_seqs) > 0:
                f.write(f"Action Sequences:\n")
                for turn_idx, action in enumerate(action_seqs, 1):
                    f.write(f"  Turn {turn_idx}: {action}\n")

            # Log full conversation sequence
            sequence = reward_input.get("sequence", "")
            if sequence:
                f.write(f"\nFull Conversation Sequence:\n")
                # Format the entire sequence based on options
                formatted_sequence = sequence
                if preserve_special_chars:
                    formatted_sequence = repr(sequence)
                if full_response:
                    f.write(f"{formatted_sequence}\n")
                else:
                    f.write(f"{formatted_sequence[:1000]}{'...' if len(formatted_sequence) > 1000 else ''}\n")

            # Log individual model responses for format checking
            response_list = reward_input.get("response_list", [])
            if response_list is not None and len(response_list) > 0:
                f.write(f"\nIndividual Model Responses (for format checking):\n")
                for turn_idx, response in enumerate(response_list, 1):
                    f.write(f"  Response {turn_idx}:\n")
                    # Format response based on options
                    formatted_response = response
                    if preserve_special_chars:
                        formatted_response = repr(response)
                    if full_response:
                        f.write(f"    {formatted_response}\n")
                    else:
                        f.write(f"    {formatted_response[:300]}{'...' if len(formatted_response) > 300 else ''}\n")
                    # Compute format reward for this response
                    format_score = compute_format_reward(response)
                    f.write(f"    Format Score: {format_score}\n")
                    # Check for truncated thinking specifically
                    if "<think>" in response and "</think>" not in response:
                        f.write(f"    ⚠️ TRUNCATED THINKING DETECTED: Response has <think> without </think>\n")

            # Log unit test traces
            unit_test_traces = reward_input.get("unit_test_traces", [])
            code_attempt_traces = reward_input.get("code_attempt_traces", [])
            oracle_traces = reward_input.get("oracle_traces", "")
            f.write(f"\nUnit Test Traces: {unit_test_traces}\n")
            f.write(f"Code Attempt Traces (count): {len(code_attempt_traces)}\n")
            f.write(f"Oracle Traces: {oracle_traces}\n")

            # Log computed rewards
            f.write(f"\nComputed Rewards:\n")
            f.write(f"  Correctness: {result.get('correctness', 0.0):.4f}\n")
            f.write(f"  Discounted Reward: {result.get('discounted_reward', 0.0):.4f}\n")
            f.write(f"  Format Reward: {result.get('format_reward', 1.0):.4f}\n")
            f.write(f"  Match Rate: {result.get('match_rate', 0.0):.4f}\n")
            f.write(f"  Num Unit Tests: {result.get('num_unit_tests', 0)}\n")
            f.write(f"  Num Code Attempts: {result.get('num_code_attempts', 0)}\n")
            f.write(f"  Num Unit Tests Before First Code: {result.get('num_unit_tests_before_first_code', 0)}\n")
            f.write(f"  Overall Reward: {result.get('overall', 0.0):.4f}\n")

            f.write("\n" + "=" * 80 + "\n\n")

        # Add separator between steps
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF STEP\n")
        f.write("=" * 80 + "\n\n\n")

    print(f"📊 Code Test Explorer rewards logged to: {log_path} (logged {len(samples_to_log)} samples out of {total_samples})")


to_jsonable = lambda o: (
    o.tolist() if isinstance(o, np.ndarray)
    else float(o) if isinstance(o, (np.floating,))
    else int(o) if isinstance(o, (np.integer,))
    else {k: to_jsonable(v) for k, v in o.items()} if isinstance(o, dict)
    else [to_jsonable(v) for v in o] if isinstance(o, (list, tuple))
    else o
)


def check_code_test_correctness(pred, answer, tol=1e-3):
    """Check correctness for code test explorer tasks."""
    try:
        answer_value = float(answer)
        pred_value = float(pred)
        return 1.0 if abs(pred_value - answer_value) < tol else 0.0
    except (TypeError, ValueError):
        # Fallback to string match
        if pred is None or answer is None:
            return 0.0
        return 1.0 if str(pred).strip() == str(answer).strip() else 0.0


def compute_discounted_reward(d_unit, d_code, num_unit_tests, num_code_attempts, correctness):
    """
    Compute discounted reward for code test explorer.
    Formula: correctness * (d_unit ** num_unit_tests) * (d_code ** num_code_attempts)
    """
    try:
        discounted_reward = correctness * (d_unit ** num_unit_tests) * (d_code ** num_code_attempts)
        return round(discounted_reward, 6)
    except Exception as e:
        print(f"⚠️ Failed to compute discounted reward: {e}")
        return 0


def _compute_match_rate(pred_trace, oracle_trace):
    """
    Compute match rate between predicted unit test trace and oracle trace.
    For code test explorer: skiprows, quotechar, delimiter
    """
    score = 0
    for choice in ['skiprows', 'quotechar', 'delimiter']:
        if choice in oracle_trace:
            if choice in pred_trace:
                score += 1
        else:
            if choice not in pred_trace:
                score += 1

    return score / 3.0

def _compute_individual_unit_usage(pred_trace):
    unit_usage = {'skiprows': 0, 'quotechar': 0, 'delimiter': 0}
    for choice in ['skiprows', 'quotechar', 'delimiter']:
        if choice in pred_trace:
            unit_usage[choice] = 1
    return unit_usage

def compute_format_reward(response: str) -> float:
    """
    Compute format reward based on whether the model follows the expected format.
    Expected formats for code test explorer:
    1. UNIT_TESTS call: "UNIT_TESTS: test_xxx(...), test_yyy(...)"
    2. CODE block: "```python ... ```"
    3. ANSWER: "ANSWER: <text>"

    Returns:
    - 1.0: Valid format (UNIT_TESTS OR CODE OR ANSWER)
    - 0.0: Invalid format (no recognizable action or multiple conflicting actions)

    Strict RL Training Rules:
    - If response contains <think> but not </think>, return 0.0 (truncated thinking, model thinking too long)

    This logic matches the parse_turn_action function in:
    - multi_turn_rollout_code_test_explorer.py
    - conversation_manager_code.py
    """
    # Check for truncated thinking (strict penalty for RL training)
    # In conversation_manager_code.py, this allows regeneration, but in RL we penalize it
    if "<think>" in response and "</think>" not in response:
        return 0.0

    # Remove thinking tags if present (matching parse_turn_action)
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    if '<|im_end|>' in response:
        response = response.replace('<|im_end|>', '').strip()

    text = response.strip()

    # Check for UNIT_TESTS calls (matching parse_turn_action pattern)
    unit_test_match = re.search(r'UNIT_TESTS:\s*(.+)$', text, flags=re.DOTALL)
    has_unit_tests = False
    if unit_test_match:
        call_text = unit_test_match.group(1).strip()
        # Extract only valid test function calls matching pattern: test_xxx(...)
        valid_calls = re.findall(r'test_\w+\([^)]*\)', call_text)
        if valid_calls:
            has_unit_tests = True

    # Check for CODE blocks (matching parse_turn_action pattern)
    code_match = re.search(r'```python\s*(.*?)```', text, flags=re.DOTALL)
    has_code = code_match is not None

    # Check for ANSWER (matching parse_turn_action pattern)
    answer_match = re.search(r'ANSWER:\s*(.+)$', text, flags=re.DOTALL | re.IGNORECASE)
    has_answer = answer_match is not None

    # Count valid actions
    total_actions = int(has_unit_tests) + int(has_code) + int(has_answer)

    if total_actions == 0:
        # No recognizable action
        return 0.0
    elif total_actions == 1:
        # Exactly one action type - valid
        return 1.0
    else:
        # Multiple conflicting actions - invalid
        return 0.0


def count_unit_tests_and_code_attempts(action_seqs: List[Tuple[str, Any]]) -> Tuple[int, int, int]:
    """
    Count the number of unit tests, code attempts, and unit tests before the first code attempt.

    Args:
        action_seqs: List of (action_type, content) tuples

    Returns:
        (num_unit_tests, num_code_attempts, num_unit_tests_before_first_code)
    """
    num_unit_tests = 0
    num_code_attempts = 0
    num_unit_tests_before_first_code = 0
    first_code_seen = False

    for action_type, content in action_seqs:
        if action_type == "UNIT_TESTS":
            if isinstance(content, list):
                num_unit_tests += len(content)
            else:
                num_unit_tests += 1

            if not first_code_seen:
                if isinstance(content, list):
                    num_unit_tests_before_first_code += len(content)
                else:
                    num_unit_tests_before_first_code += 1

        elif action_type == "CODE":
            num_code_attempts += 1
            first_code_seen = True

    return num_unit_tests, num_code_attempts, num_unit_tests_before_first_code


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1,
                  log_dir: str = "PATH_TO/reward_logs",
                  full_response: bool = False, preserve_special_chars: bool = False,
                  max_turns: int = 10, **reward_kwargs) -> List[Dict[str, float]]:
    """
    Batch reward function entrypoint for code test explorer format.
    Each input dict provides: response, response_length, gold_answers, pred_answers, and optional metadata.
    Returns per-sample dict with keys: overall, discounted_reward, correctness, num_unit_tests,
    num_code_attempts, num_unit_tests_before_first_code, match_rate, format_reward.

    Overall reward logic:
    - If format_reward is 0 OR trace is too long, overall_reward = 0 (strict penalty)
    - Otherwise, overall_reward = 0.1 * format_reward + 0.9 * discounted_reward

    Args:
        reward_inputs: List of reward input dictionaries
        format_weight: Weight for format reward (default: 0.1)
        log_dir: Directory for logging rewards (configurable via training script)
        full_response: If True, log full responses without truncation for debugging (default: False)
        preserve_special_chars: If True, use repr() to preserve special characters in logs (default: False)
        max_turns: Maximum allowed turns before considering trace too long (default: 10)
        **reward_kwargs: Additional reward function arguments
    """
    results: List[Dict[str, float]] = []

    for reward_input in reward_inputs:
        print("reward_input:", reward_input)
        print("🌳 🌳 reward_input keys:", reward_input.keys(), "🌳 🌳")

        gold_answer = reward_input.get("gold_answers", "")
        pred_answer = reward_input.get("pred_answers", "")
        action_seqs = reward_input.get("action_seqs", [])
        d_unit = reward_input.get("d_unit_list", 1.0)
        d_code = reward_input.get("d_code_list", 1.0)
        response_list = reward_input.get("response_list", [])  # List of responses for each turn

        unit_test_traces = reward_input.get("unit_test_traces", [])
        code_attempt_traces = reward_input.get("code_attempt_traces", [])
        oracle_traces_str = reward_input.get("oracle_traces", [])
        print("oracle_traces_str:", oracle_traces_str)

        def convert_trace_str_to_list(trace_str):
            if trace_str == "":
                return []
            temp = trace_str.split("_")
            temp = [x for x in temp if x != '']
            return temp

        oracle_traces = convert_trace_str_to_list(oracle_traces_str)
        print("oracle_traces:", oracle_traces)

        correctness = check_code_test_correctness(pred_answer, gold_answer)
        print(f"pred_answer: {pred_answer}, gold_answer: {gold_answer}, correctness: {correctness}")

        # Count unit tests and code attempts from action sequences
        num_unit_tests, num_code_attempts, num_unit_tests_before_first_code = count_unit_tests_and_code_attempts(action_seqs)

        # Compute discounted reward
        discount_reward = compute_discounted_reward(d_unit, d_code, num_unit_tests, num_code_attempts, correctness)
        print(f"d_unit: {d_unit}, d_code: {d_code}, num_unit_tests: {num_unit_tests}, num_code_attempts: {num_code_attempts}, discounted_reward: {discount_reward}")

        # Compute format reward: 0 if ANY turn does not satisfy criteria
        if response_list is not None and len(response_list) > 0:
            format_rewards = [compute_format_reward(resp) for resp in response_list]
            # Format reward is 0 if any turn has format_reward < 1.0
            format_reward = 1.0 if all(fr == 1.0 for fr in format_rewards) else 0.0
        else:
            # Fallback: if no response_list, assume perfect format
            format_reward = 1.0

        print(f"format_reward: {format_reward}")

        # Check if trace is too long
        trace_too_long = len(action_seqs) > max_turns
        if trace_too_long:
            print(f"⚠️ Trace too long: {len(action_seqs)} > {max_turns}")

        # Compute match rate with oracle trace
        match_rate = _compute_match_rate(unit_test_traces, oracle_traces)
        print(f"unit_test_traces: {unit_test_traces}, oracle_traces: {oracle_traces}, match_rate: {match_rate}")
        unit_usage = _compute_individual_unit_usage(unit_test_traces)
        print(f"unit_usage: {unit_usage}")

        # Combine rewards with strict format penalty and trace length penalty:
        # If format_reward is 0 OR trace is too long, overall is 0; otherwise weighted combination
        if format_reward == 0.0 or trace_too_long:
            overall_reward = 0.0
        else:
            overall_reward = 0.1 * format_reward + 0.9 * discount_reward

        results.append({
            "overall": overall_reward,
            "discounted_reward": discount_reward,
            "correctness": correctness,
            "num_unit_tests": num_unit_tests,
            "num_code_attempts": num_code_attempts,
            "num_unit_tests_before_first_code": num_unit_tests_before_first_code,
            "match_rate": match_rate,
            "format_reward": format_reward,
            "skiprows_used": unit_usage['skiprows'],
            "quotechar_used": unit_usage['quotechar'],
            "delimiter_used": unit_usage['delimiter'],

        })

    # Log rewards for debugging
    try:
        log_code_test_explorer_rewards(reward_inputs, results, log_dir=log_dir, num_samples=5,
                                  full_response=True, preserve_special_chars=preserve_special_chars)
    except Exception as e:
        print(f"⚠️ Failed to log rewards: {e}")

    return results
