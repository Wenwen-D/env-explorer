# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        multi_turn_mask = data.batch["multi_turn_mask"]

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["answer"][i],
            }
            
            # Add additional training data fields if available
            if "table_id" in data.non_tensor_batch:
                reward_input["table_id"] = data.non_tensor_batch["table_id"][i]
            
            if "question_id" in data.non_tensor_batch:
                reward_input["question_id"] = data.non_tensor_batch["question_id"][i]
            
            if "related_data" in data.non_tensor_batch:
                reward_input["related_data"] = data.non_tensor_batch["related_data"][i]
            
            if "meta_data_path" in data.non_tensor_batch:
                reward_input["meta_data_path"] = data.non_tensor_batch["meta_data_path"][i]
            
            # if "crop_paths_data" in data.non_tensor_batch:
            #     reward_input["crop_paths_data"] = data.non_tensor_batch["crop_paths_data"][i]
            
            score = self.reward_fn(reward_input)

            # Find the last assistant token position for reward assignment
            assistant_token_positions = torch.where(multi_turn_mask[i] == 1)[0]
            if len(assistant_token_positions) > 0:
                # Assign reward to the last assistant token
                last_assistant_pos = assistant_token_positions[-1].item()
                reward_tensor[i, last_assistant_pos] = score["overall"]
            else:
                # Fallback: assign to the last response token if no assistant tokens found
                reward_tensor[i, cur_response_length - 1] = score["overall"]
            
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)

        multi_turn_mask = data.batch["multi_turn_mask"]

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["answer"][i],
            }

            # Add full conversation sequence if available
            # Use full_sequences if available (untruncated), otherwise fall back to raw_prompt_ids (truncated)
            if "full_sequences" in data.non_tensor_batch:
                try:
                    reward_input["sequence"] = data.non_tensor_batch["full_sequences"][i]
                except Exception:
                    pass
            elif "raw_prompt_ids" in data.non_tensor_batch:
                try:
                    raw_ids = data.non_tensor_batch["raw_prompt_ids"][i]
                    reward_input["sequence"] = self.tokenizer.decode(raw_ids, skip_special_tokens=False)
                except Exception:
                    pass

            # Add additional training data fields if available
            if "table_id" in data.non_tensor_batch:
                reward_input["table_id"] = data.non_tensor_batch["table_id"][i]
            
            if "question_id" in data.non_tensor_batch:
                reward_input["question_id"] = data.non_tensor_batch["question_id"][i]
            
            if "related_data" in data.non_tensor_batch:
                reward_input["related_data"] = data.non_tensor_batch["related_data"][i]
            
            if "meta_data_path" in data.non_tensor_batch:
                reward_input["meta_data_path"] = data.non_tensor_batch["meta_data_path"][i]
            
            if "environment" in data.non_tensor_batch:
                reward_input["environment"] = data.non_tensor_batch["environment"][i]
            if "answer" in data.non_tensor_batch:
                reward_input["answer"] = data.non_tensor_batch["answer"][i]
            if "action_seqs" in data.non_tensor_batch:
                reward_input["action_seqs"] = data.non_tensor_batch["action_seqs"][i]
            else:
                raise ValueError("action_seqs is required in non_tensor_batch for BatchFunctionRewardManager.")
            if "discount_factors" in data.non_tensor_batch:
                reward_input["discount_factors"] = data.non_tensor_batch["discount_factors"][i]
            if "helper_call_traces" in data.non_tensor_batch:
                reward_input["helper_call_traces"] = data.non_tensor_batch["helper_call_traces"][i]
            if "oracle_traces" in data.non_tensor_batch:
                reward_input["oracle_traces"] = data.non_tensor_batch["oracle_traces"][i]
            if "gold_answers" in data.non_tensor_batch:
                reward_input["gold_answers"] = data.non_tensor_batch["gold_answers"][i]
            if "pred_answers" in data.non_tensor_batch:
                reward_input["pred_answers"] = data.non_tensor_batch["pred_answers"][i]
            if "d_unit_list" in data.non_tensor_batch:
                reward_input["d_unit_list"] = data.non_tensor_batch["d_unit_list"][i]
            if "d_code_list" in data.non_tensor_batch:
                reward_input["d_code_list"] = data.non_tensor_batch["d_code_list"][i]
            if "response_list" in data.non_tensor_batch:
                reward_input["response_list"] = data.non_tensor_batch["response_list"][i]
            if "task_id" in data.non_tensor_batch:
                reward_input["task_id"] = data.non_tensor_batch["task_id"][i]
            if "unit_test_traces" in data.non_tensor_batch:
                reward_input["unit_test_traces"] = data.non_tensor_batch["unit_test_traces"][i]
            if "code_attempt_traces" in data.non_tensor_batch:
                reward_input["code_attempt_traces"] = data.non_tensor_batch["code_attempt_traces"][i]
            





            # if "crop_paths_data" in data.non_tensor_batch:
            #     reward_input["crop_paths_data"] = data.non_tensor_batch["crop_paths_data"][i]

            reward_inputs.append(reward_input)

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            # Find the last assistant token position for reward assignment
            assistant_token_positions = torch.where(multi_turn_mask[i] == 1)[0]
            if len(assistant_token_positions) > 0:
                # Assign reward to the last assistant token
                last_assistant_pos = assistant_token_positions[-1].item()
                reward_tensor[i, last_assistant_pos] = score["overall"]
            else:
                # Fallback: assign to the last response token if no assistant tokens found
                response_length = torch.sum(data.batch["response_mask"][i], dim=-1)
                cur_response_length = int(response_length.item())
                reward_tensor[i, cur_response_length - 1] = score["overall"]
            
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
