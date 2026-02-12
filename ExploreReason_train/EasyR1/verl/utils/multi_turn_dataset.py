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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union, List, Dict

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
from .dataset import process_image, process_video


class MultiTurnRLHFDataset(Dataset):
    """
    Multi-turn RLHF dataset for table reading tasks with conversation history.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "problem",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        environment_key: str = "environment",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 5000,
        max_response_length: int = 2000,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        max_turns: int = 8,
        single_turn_response_length: int = 500,
        is_thinking_model: bool = False,
        enable_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.environment_key = environment_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_turns = max_turns
        self.single_turn_response_length = single_turn_response_length
        self.is_thinking_model = is_thinking_model
        self.enable_thinking = enable_thinking

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_from_disk(data_path)[data_split]
        else:
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )
        
    def _build_initial_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """Build initial messages for the conversation."""
        prompt_str: str = example[self.prompt_key]
        
        # Extract system prompt from format template if available
        system_prompt = None
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            # Get the system prompt part (everything after the content)
            rendered = format_prompt.render(content="")
            if rendered.strip():
                system_prompt = rendered.strip()

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content_list})
            return messages
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content_list})
            return messages
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_str})
            return messages

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        """Filter out prompts that are too long for multi-turn processing."""
        messages = self._build_initial_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            if self.is_thinking_model:
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=self.enable_thinking) # TODO: add thinking here!
            else:
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_initial_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            if self.is_thinking_model:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking) # TODO: add thinking here!
            else:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) 
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        # Handle position IDs for different models
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            from .dataset import get_rope_index
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise ValueError(f"Prompt too long: {len(raw_prompt_ids)} > {self.max_prompt_length}")

        # For multi-turn, we need to create a placeholder response that will be filled during rollout
        # The response will be generated dynamically during the multi-turn conversation
        placeholder_response_ids = torch.full(
            (self.max_response_length,), 
            self.tokenizer.pad_token_id, 
            dtype=input_ids.dtype
        )
        
        # Create a placeholder response mask
        placeholder_response_mask = torch.zeros(
            (self.max_response_length,), 
            dtype=attention_mask.dtype
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "multi_modal_data": example.get("multi_modal_data", None),
            "responses": placeholder_response_ids,
            "response_mask": placeholder_response_mask,
            "environment": example.get(self.environment_key, ""),   
            **{k: v for k, v in example.items() if k not in ["multi_modal_data"]},
        }


class MultiTurnRLHFDataset_csv_explorer(Dataset):
    """
    Multi-turn RLHF dataset for table reading tasks with conversation history.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "problem",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        environment_key: str = "environment",
        task_id_key: str = "task_id",
        discount_factor_key: str = "discount_factor",
        oracle_trace_key: str = "oracle_trace",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 5000,
        max_response_length: int = 2000,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        max_turns: int = 8,
        single_turn_response_length: int = 500,
        is_thinking_model: bool = False,
        enable_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.environment_key = environment_key
        self.task_id_key = task_id_key
        self.discount_factor_key = discount_factor_key
        self.oracle_trace_key = oracle_trace_key

        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_turns = max_turns
        self.single_turn_response_length = single_turn_response_length
        self.is_thinking_model = is_thinking_model
        self.enable_thinking = enable_thinking

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_from_disk(data_path)[data_split]
        else:
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )
        
    def _build_initial_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """Build initial messages for the conversation."""
        prompt_str: str = example[self.prompt_key]
        
        # Extract system prompt from format template if available
        system_prompt = None
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            # Get the system prompt part (everything after the content)
            rendered = format_prompt.render(content="")
            if rendered.strip():
                system_prompt = rendered.strip()

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content_list})
            return messages
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content_list})
            return messages
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_str})
            return messages

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        """Filter out prompts that are too long for multi-turn processing."""
        messages = self._build_initial_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            if self.is_thinking_model:
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=self.enable_thinking) # TODO: add thinking here!
            else:
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_initial_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            if self.is_thinking_model:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking) # TODO: add thinking here!
            else:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) 
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        # Handle position IDs for different models
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            from .dataset import get_rope_index
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise ValueError(f"Prompt too long: {len(raw_prompt_ids)} > {self.max_prompt_length}")

        # For multi-turn, we need to create a placeholder response that will be filled during rollout
        # The response will be generated dynamically during the multi-turn conversation
        placeholder_response_ids = torch.full(
            (self.max_response_length,), 
            self.tokenizer.pad_token_id, 
            dtype=input_ids.dtype
        )
        
        # Create a placeholder response mask
        placeholder_response_mask = torch.zeros(
            (self.max_response_length,), 
            dtype=attention_mask.dtype
        )
        print("🌐 MultiTurnRLHFDataset_csv_explorer Debug example output keys:",)
        print(example.keys())
        print("🌐 MultiTurnRLHFDataset_csv_explorer Debug example values:", example)


        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "multi_modal_data": example.get("multi_modal_data", None),
            "responses": placeholder_response_ids,
            "response_mask": placeholder_response_mask,
            "environment": example.get(self.environment_key, ""),
            "task_id": example.get(self.task_id_key, ""),
            "discount_factor": example.get(self.discount_factor_key, 1.0),
            "oracle_trace": example.get(self.oracle_trace_key, ""),
            **{k: v for k, v in example.items() if k not in ["multi_modal_data"]},
        }


class MultiTurnRLHFDataset_code_test_explorer(Dataset):
    """
    Multi-turn RLHF dataset for code test explorer tasks with conversation history.
    Extends csv_explorer with additional fields for d_unit and d_code discount factors.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "problem",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        task_id_key: str = "task_id",
        d_unit_key: str = "d_unit",
        d_code_key: str = "d_code",
        oracle_trace_key: str = "oracle_trace",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 5000,
        max_response_length: int = 2000,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        max_turns: int = 8,
        single_turn_response_length: int = 500,
        is_thinking_model: bool = False,
        enable_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.task_id_key = task_id_key
        self.d_unit_key = d_unit_key
        self.d_code_key = d_code_key
        self.oracle_trace_key = oracle_trace_key

        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_turns = max_turns
        self.single_turn_response_length = single_turn_response_length
        self.is_thinking_model = is_thinking_model
        self.enable_thinking = enable_thinking

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_from_disk(data_path)[data_split]
        else:
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_initial_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        """Build initial messages for the conversation."""
        prompt_str: str = example[self.prompt_key]

        # Extract system prompt from format template if available
        system_prompt = None
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            # Get the system prompt part (everything after the content)
            rendered = format_prompt.render(content="")
            if rendered.strip():
                system_prompt = rendered.strip()

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content_list})
            return messages
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content_list})
            return messages
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_str})
            return messages

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        """Filter out prompts that are too long for multi-turn processing."""
        messages = self._build_initial_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            if self.is_thinking_model:
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=self.enable_thinking)
            else:
                input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_initial_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            if self.is_thinking_model:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking)
            else:
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        # Handle position IDs for different models
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            from .dataset import get_rope_index
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise ValueError(f"Prompt too long: {len(raw_prompt_ids)} > {self.max_prompt_length}")

        # For multi-turn, we need to create a placeholder response that will be filled during rollout
        placeholder_response_ids = torch.full(
            (self.max_response_length,),
            self.tokenizer.pad_token_id,
            dtype=input_ids.dtype
        )

        # Create a placeholder response mask
        placeholder_response_mask = torch.zeros(
            (self.max_response_length,),
            dtype=attention_mask.dtype
        )

        # DEBUG: Check what's in the raw example before we return
        print(f"🌐 MultiTurnRLHFDataset_code_test_explorer Debug example keys: {example.keys()}")
        print(f"🌐 MultiTurnRLHFDataset_code_test_explorer Debug example values: {example}")

        # Check for required keys
        print(f"🔍 DEBUG Dataset __getitem__: Looking for key '{self.d_unit_key}' in example")
        print(f"🔍 DEBUG Dataset __getitem__: d_unit value = {example.get(self.d_unit_key, 'KEY_NOT_FOUND')}")
        print(f"🔍 DEBUG Dataset __getitem__: Looking for key '{self.d_code_key}' in example")
        print(f"🔍 DEBUG Dataset __getitem__: d_code value = {example.get(self.d_code_key, 'KEY_NOT_FOUND')}")
        print(f"🔍 DEBUG Dataset __getitem__: Looking for 'sampled_format' in example")
        print(f"🔍 DEBUG Dataset __getitem__: sampled_format value = {example.get('sampled_format', 'KEY_NOT_FOUND')}")

        # Extract all values we need
        d_unit_value = example.get(self.d_unit_key, 1.0)
        d_code_value = example.get(self.d_code_key, 1.0)
        sampled_format_value = example.get('sampled_format', None)

        # Build the return dict
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids,
            "multi_modal_data": example.get("multi_modal_data", None),
            "responses": placeholder_response_ids,
            "response_mask": placeholder_response_mask,
            "task_id": example.get(self.task_id_key, ""),
            "d_unit": d_unit_value,
            "d_code": d_code_value,
            "oracle_trace": example.get(self.oracle_trace_key, ""),
            **{k: v for k, v in example.items() if k not in ["multi_modal_data"]},
        }

        # Explicitly add sampled_format if it exists
        if sampled_format_value is not None:
            result["sampled_format"] = sampled_format_value
            print(f"✅ DEBUG Dataset __getitem__: Added sampled_format to result: {sampled_format_value}")
        else:
            print(f"⚠️ WARNING Dataset __getitem__: sampled_format not found in example! Will be None in result.")

        print(f"🔍 DEBUG Dataset __getitem__: Final result keys: {result.keys()}")
        print(f"🔍 DEBUG Dataset __getitem__: Final result d_unit = {result.get('d_unit')}")
        print(f"🔍 DEBUG Dataset __getitem__: Final result d_code = {result.get('d_code')}")
        print(f"🔍 DEBUG Dataset __getitem__: Final result sampled_format = {result.get('sampled_format')}")

        return result
