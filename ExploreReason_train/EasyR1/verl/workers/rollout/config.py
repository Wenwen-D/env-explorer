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
"""
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    val_override_config: dict[str, Any] = field(default_factory=dict)
    # Multi-turn specific parameters
    max_turns: int = 8
    num_llm_calls_available: int = 8
    single_turn_response_length: int = 500
    crop_size: int = 200
    # Tool-use cropping behavior
    multi_point_enclosing_crop: bool = False
    enclosing_padding: int = 50
    temp_dir: str = "temp_crops"
    # Out-of-bounds early stopping behavior
    oob_early_stop: bool = False
    # Provide corrective feedback text between turns
    provide_feedback: bool = False
    feedback_distance_threshold: int = -1
    # Suppress axis direction if |dx| or |dy| <= this epsilon
    feedback_axis_epsilon: int = 5
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)
    retrieved_dict_path: Optional[str] = field(default=None, init=False)
    all_task_base_path: Optional[str] = field(default=None, init=False)
    enable_thinking: bool = field(default=False, init=False)
    log_dir: Optional[str] = field(default=None, init=False)
    def to_dict(self):
        return asdict(self)
