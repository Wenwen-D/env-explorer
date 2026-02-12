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

from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from ..utils.multi_turn_dataset import MultiTurnRLHFDataset, MultiTurnRLHFDataset_csv_explorer, MultiTurnRLHFDataset_code_test_explorer
# from ..utils.multi_turn_PopQA_dataset import MultiTurnPopQADataset
from .config import DataConfig


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    # Check if multi-turn dataset is requested
    
    if hasattr(config, 'use_multi_turn') and config.use_multi_turn: # TODO:
        if hasattr(config, 'multi_turn_data_loader_type') and config.multi_turn_data_loader_type == 'csv_explorer':
            print("Using MultiTurnRLHFDataset_csv_explorer for data loading.")
            train_dataset = MultiTurnRLHFDataset_csv_explorer(
                data_path=config.train_files,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key=config.prompt_key,
                answer_key=config.answer_key,
                image_key=config.image_key,
                video_key=config.video_key,
                environment_key=getattr(config, 'environment_key', None),
                task_id_key=getattr(config, 'task_id_key', 'task_id'),
                discount_factor_key=getattr(config, 'discount_factor_key', 'discount_factor'),
                oracle_trace_key=getattr(config, 'oracle_trace_key', 'oracle_trace'),
                image_dir=config.image_dir,
                video_fps=config.video_fps,
                max_prompt_length=config.max_prompt_length,
                max_response_length=config.max_response_length,
                truncation="right",
                format_prompt=config.format_prompt,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                filter_overlong_prompts=config.filter_overlong_prompts,
                filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
                max_turns=getattr(config, 'max_turns', 8),
                single_turn_response_length=getattr(config, 'single_turn_response_length', 500),
                is_thinking_model=getattr(config, 'is_thinking_model', False),
                enable_thinking=getattr(config, 'enable_thinking', False),
            )
        elif hasattr(config, 'multi_turn_data_loader_type') and config.multi_turn_data_loader_type == 'code_test_explorer':
            print("Using MultiTurnRLHFDataset_code_test_explorer for data loading.")
            train_dataset = MultiTurnRLHFDataset_code_test_explorer(
                data_path=config.train_files,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key=config.prompt_key,
                answer_key=config.answer_key,
                image_key=config.image_key,
                video_key=config.video_key,
                task_id_key=getattr(config, 'task_id_key', 'task_id'),
                d_unit_key=getattr(config, 'd_unit_key', 'd_unit'),
                d_code_key=getattr(config, 'd_code_key', 'd_code'),
                oracle_trace_key=getattr(config, 'oracle_trace_key', 'oracle_trace'),
                image_dir=config.image_dir,
                video_fps=config.video_fps,
                max_prompt_length=config.max_prompt_length,
                max_response_length=config.max_response_length,
                truncation="right",
                format_prompt=config.format_prompt,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                filter_overlong_prompts=config.filter_overlong_prompts,
                filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
                max_turns=getattr(config, 'max_turns', 8),
                single_turn_response_length=getattr(config, 'single_turn_response_length', 4096),
                is_thinking_model=getattr(config, 'is_thinking_model', False),
                enable_thinking=getattr(config, 'enable_thinking', False),
            )
        else:
            train_dataset = MultiTurnRLHFDataset(
                data_path=config.train_files,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key=config.prompt_key,
                answer_key=config.answer_key,
                image_key=config.image_key,
                video_key=config.video_key,
                environment_key=getattr(config, 'environment_key', None),
                image_dir=config.image_dir,
                video_fps=config.video_fps,
                max_prompt_length=config.max_prompt_length,
                max_response_length=config.max_response_length,
                truncation="right",
                format_prompt=config.format_prompt,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                filter_overlong_prompts=config.filter_overlong_prompts,
                filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
                max_turns=getattr(config, 'max_turns', 8),
                single_turn_response_length=getattr(config, 'single_turn_response_length', 500),
                is_thinking_model=getattr(config, 'is_thinking_model', False),
                enable_thinking=getattr(config, 'enable_thinking', False),
            )
        
    else:
        train_dataset = RLHFDataset(
            data_path=config.train_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            video_key=config.video_key,
            environment_key=getattr(config, 'environment_key', None),
            image_dir=config.image_dir,
            video_fps=config.video_fps,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
            filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        )
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )
    if hasattr(config, 'use_multi_turn') and config.use_multi_turn: # TODO:
        if hasattr(config, 'multi_turn_data_loader_type') and config.multi_turn_data_loader_type == 'csv_explorer':
            val_dataset = MultiTurnRLHFDataset_csv_explorer(
                data_path=config.val_files,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key=config.prompt_key,
                answer_key=config.answer_key,
                image_key=config.image_key,
                video_key=config.video_key,
                environment_key=getattr(config, 'environment_key', None),
                task_id_key=getattr(config, 'task_id_key', 'task_id'),
                discount_factor_key=getattr(config, 'discount_factor_key', 'discount_factor'),
                oracle_trace_key=getattr(config, 'oracle_trace_key', 'oracle_trace'),
                image_dir=config.image_dir,
                video_fps=config.video_fps,
                max_prompt_length=config.max_prompt_length,
                max_response_length=config.max_response_length,
                truncation="right",
                format_prompt=config.format_prompt,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                filter_overlong_prompts=config.filter_overlong_prompts,
                filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
                max_turns=getattr(config, 'max_turns', 8),
                single_turn_response_length=getattr(config, 'single_turn_response_length', 500),
                is_thinking_model=getattr(config, 'is_thinking_model', False),
                enable_thinking=getattr(config, 'enable_thinking', False),
            )
        elif hasattr(config, 'multi_turn_data_loader_type') and config.multi_turn_data_loader_type == 'code_test_explorer':
            val_dataset = MultiTurnRLHFDataset_code_test_explorer(
                data_path=config.val_files,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key=config.prompt_key,
                answer_key=config.answer_key,
                image_key=config.image_key,
                video_key=config.video_key,
                task_id_key=getattr(config, 'task_id_key', 'task_id'),
                d_unit_key=getattr(config, 'd_unit_key', 'd_unit'),
                d_code_key=getattr(config, 'd_code_key', 'd_code'),
                oracle_trace_key=getattr(config, 'oracle_trace_key', 'oracle_trace'),
                image_dir=config.image_dir,
                video_fps=config.video_fps,
                max_prompt_length=config.max_prompt_length,
                max_response_length=config.max_response_length,
                truncation="right",
                format_prompt=config.format_prompt,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                filter_overlong_prompts=config.filter_overlong_prompts,
                filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
                max_turns=getattr(config, 'max_turns', 8),
                single_turn_response_length=getattr(config, 'single_turn_response_length', 500),
                is_thinking_model=getattr(config, 'is_thinking_model', False),
                enable_thinking=getattr(config, 'enable_thinking', False),
            )
        else:
            val_dataset = MultiTurnRLHFDataset(
                data_path=config.val_files,
                tokenizer=tokenizer,
                processor=processor,
                prompt_key=config.prompt_key,
                answer_key=config.answer_key,
                image_key=config.image_key,
                video_key=config.video_key,
                environment_key=getattr(config, 'environment_key', None),
                image_dir=config.image_dir,
                video_fps=config.video_fps,
                max_prompt_length=config.max_prompt_length,
                max_response_length=config.max_response_length,
                truncation="right",
                format_prompt=config.format_prompt,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                filter_overlong_prompts=config.filter_overlong_prompts,
                filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
                max_turns=getattr(config, 'max_turns', 8),
                single_turn_response_length=getattr(config, 'single_turn_response_length', 500),
            )
            
    else:
        val_dataset = RLHFDataset(
            data_path=config.val_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=config.prompt_key,
            answer_key=config.answer_key,
            image_key=config.image_key,
            environment_key=getattr(config, 'environment_key', None),
            image_dir=config.image_dir,
            max_prompt_length=config.max_prompt_length,
            truncation="right",
            format_prompt=config.format_prompt,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            filter_overlong_prompts=config.filter_overlong_prompts,
        )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    print(f"Sample data point: {train_dataset[0]}")
    return train_dataloader, val_dataloader
