# scripts/run_evaluation.py
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from src.models.lite_llm import APIModel, LocalModel
from src.data.task_loader import TaskLoader
from src.data.data_types import TaskInstance, TaskResult
from src.conversation.conversation_manager_popqa import ConversationManager_popqa, RetrievedDict
from src.execute_code import CodeExecutor
from src.evaluation.task_evaluator import TaskEvaluator, compute_ece
import shutil
import os
import logging
import concurrent.futures
import threading
import yaml
import numpy as np
from collections import defaultdict



def setup_litellm_suppression():
    """Setup comprehensive litellm logging suppression"""
    
    # Custom filter to block litellm INFO messages
    class LiteLLMFilter(logging.Filter):
        def filter(self, record):
            if record.levelno == logging.INFO:
                message = record.getMessage().lower()
                if any(keyword in message for keyword in [
                    'litellm completion()',
                    'wrapper: completed call',
                    'calling success_handler',
                    'provider =',
                    'model=',
                    'litellm.completion',
                ]):
                    return False
            return True
    
    # Apply filter to root logger and known litellm loggers
    litellm_loggers = [
        "",  # root logger
        "litellm",
        "litellm.proxy", 
        "litellm.router",
        "litellm.utils",
        "litellm.integrations",
        "litellm.llms",
        "litellm.main"
    ]
    
    for logger_name in litellm_loggers:
        logger = logging.getLogger(logger_name)
        logger.addFilter(LiteLLMFilter())
        if logger_name != "":  # Don't change root logger level
            logger.setLevel(logging.WARNING)
            logger.propagate = False
    
    # Find and suppress any existing litellm loggers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if 'litellm' in logger_name.lower():
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            logger.propagate = False
            logger.addFilter(LiteLLMFilter())

def load_generation_config(config_path):
    """Load generation config from YAML or JSON file"""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    return config.get('generation_config', {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model name or path")
    parser.add_argument("-c", "--checkpoint", default=None, help="Checkpoint path for Lora model")
    parser.add_argument("--remote_model", action='store_true', help="Use remote model API")
    parser.add_argument("--url", default=None, help="API URL if applicable")
    parser.add_argument("-t", "--task_metadata", required=True, help="Task metadata file")
    parser.add_argument("--limit", type=int, default=None, help="Number of tasks to run")
    parser.add_argument("--max_turns", type=int, default=1, help="Maximum conversation turns")
    parser.add_argument("-o", "--output", required=True, help="Output file prefix")
    parser.add_argument("--thinking", action='store_true', help="Use thinking mode (Qwen3 model)")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--system_choice", type=str, default='default', choices=['default', 'extended', 'draft', 'draft_v1', 'cost', 'noprior'], help="System prompt choice")
    parser.add_argument("--config_file", type=str, default=None, help="Generation config file")
    parser.add_argument("--resume", action='store_true', help="Resume from existing output file if exists")
    parser.add_argument("--do_retrieve", action='store_true', help="Whether to use retrieval in multi-turn setting or single-turn setting")
    parser.add_argument("--num_passages", type=int, default=1, help="Number of retrieved passages to include in the context")
    parser.add_argument("--r_list", nargs="+", type=float)
    parser.add_argument("--confidence_field", type=str, default="confidence_calibrated_think_platt", help="Confidence field to use for evaluation")
    parser.add_argument("--p_ret", type=float, default=0.57, help="Retrieval accuracy probability estimate")


    args = parser.parse_args()

    if args.config_file:
        generation_config = load_generation_config(args.config_file)
    else:
        generation_config = None
    
    os.environ['LITELLM_LOG'] = 'ERROR'  # Set logging level for LiteLLM

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(f"logs/{args.output}_eval.log")])
    logger.info("Starting evaluation with arguments: %s", args)
    logger.info(f"Generation config: {generation_config}")
    
    setup_litellm_suppression()


    # executor = CodeExecutor()
    # evaluator = TaskEvaluator()
    if args.remote_model:
        logger.info(f"Using remote model API at {args.url}")
        model_class = APIModel
        model = model_class(model_name=args.model, logger=logger,
                            generation_config=generation_config, base_url=args.url)
    else:
        model_class = LocalModel
        model = model_class(model_path=args.model, device='cuda', logger=logger, think_mode=args.thinking, checkpoint=args.checkpoint)

    if args.max_turns > 1 or args.do_retrieve:
        logger.info(f"Using conversation manager for multi-turn interaction (max_turns={args.max_turns})")
        logger.info(f"Loading retrieved passages for PopQA...")
        # NOTE: for _target_ret, we use the target wiki source for retrieval, otherwise used the commented line of code.
        do_retrieve = True
    else:
        logger.info(f"Single-turn interaction mode (max_turns={args.max_turns})")
        logger.info(f"No retrieved passages will be used.")
        # retrieved_passages = None
        do_retrieve = False
    conversation_manager = ConversationManager_popqa(model, logger=logger,max_turns=args.max_turns, verbose=args.verbose, system_choice = args.system_choice, do_retrieve = do_retrieve, num_passages=args.num_passages, p_ret=args.p_ret, confidence_field=args.confidence_field) # TODO: add retrieved_dict

    output_file = f"../output_popqa/{args.output}/results.json"
    if Path(output_file).exists(): 
        raise ValueError(f"Output file {output_file} already exists. Please choose a different output name to avoid overwriting.")
        # logger.warning(f"Output file {output_file} already exists. Loading for eval only...")
        # with open(output_file, 'r') as f:
        #     results = json.load(f)
        if not args.resume:
            raise ValueError(f"Output file {output_file} already exists. Please choose a different output name to avoid overwriting.")
        else:
            tasks_all = json.load(open(args.task_metadata, 'r'))
            tasks_completed = json.load(open(output_file, 'r'))
            completed_ids = set([t['task_id'] for t in tasks_completed])
            tasks = [t for t in tasks_all if t['task_id'] not in completed_ids]
            logger.info(f"Resuming from {output_file}, {len(tasks_completed)} tasks already completed, {len(tasks)} remaining.")
            results = tasks_completed
            results_lock = threading.Lock()
        logger.info(f"Running evaluation on {len(tasks)} tasks with model {args.model}, num_workers={args.num_workers}")
        logger.info(f"Results will be saved to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        chunk_size = args.num_workers * 20 
        enable_thinking = 'qwen3' in args.model.lower() and args.thinking
        print(f"🧠 enable_thinking: {enable_thinking}")
        logger.info(f"Chunk size: {chunk_size}, enable_thinking: {enable_thinking}")
        for chunk_start in tqdm(range(0, len(tasks), chunk_size), desc="Processing task chunks"): # change 1 to len(tasks)
            chunk_tasks = tasks[chunk_start:chunk_start + chunk_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_task = {
                    executor.submit(conversation_manager.run_task, task=task, instruction=task['question'], enable_thinking=enable_thinking, num_passages=args.num_passages, r=task['discount_factor']): task
                    for task in chunk_tasks
                }

                for future in tqdm(concurrent.futures.as_completed(future_to_task),
                                total=len(chunk_tasks), desc="Processing tasks"):
                    # task = future_to_task[future]
                    result = future.result()
                    with results_lock:
                        if result is not None:
                            # print("✅ one result collected")
                            results.append(result)
                            # Save incrementally
                            if len(results) % 2 == 0:
                                with open(output_file, 'w') as f:
                                    json.dump(results, f, indent=4)
                                    # json.dump([r.model_dump() for r in results], f, indent=4)
                                    # print("⏳ Dump!")
        
        with open(output_file, 'w') as f:
            # json.dump([r.dict() for r in results], f, indent=4)
            json.dump(results, f, indent=4)

    else:

        tasks = json.load(open(args.task_metadata, 'r'))
        if args.limit is not None:
            tasks = tasks[:args.limit]
        logger.info(f"Loaded {len(tasks)} tasks from {args.task_metadata}")
        logger.info(f"Sample task: {tasks[0]}")

        results = []
        results_lock = threading.Lock()
        logger.info(f"Running evaluation on {len(tasks)} tasks with model {args.model}, num_workers={args.num_workers}")
        logger.info(f"Results will be saved to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if args.r_list is not None:
            r_values = args.r_list
            tasks_with_r = []
            for task in tasks:
                for r in r_values:
                    task_copy = task.copy()
                    task_copy['discount_factor'] = r
                    tasks_with_r.append(task_copy)
            tasks = tasks_with_r
            logger.info(f"Expanded tasks with r values {r_values}, total tasks now {len(tasks)}")
        else:
            logger.info(f"No r_list provided, using original tasks.")
            assert all('discount_factor' in task for task in tasks), "All tasks must have 'discount_factor' field if no r_list is provided."
        logger.info(f"Sample expanded task: {tasks[0]}")

        chunk_size = args.num_workers * 20 
        enable_thinking = 'qwen3' in args.model.lower() and args.thinking
        print(f"🧠 enable_thinking: {enable_thinking}")
        logger.info(f"Chunk size: {chunk_size}, enable_thinking: {enable_thinking}")
        for chunk_start in tqdm(range(0, len(tasks), chunk_size), desc="Processing task chunks"): # change 1 to len(tasks)
            chunk_tasks = tasks[chunk_start:chunk_start + chunk_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:

                future_to_task = {
                    executor.submit(
                        conversation_manager.run_task,
                        task=task,
                        instruction=task['question'],
                        enable_thinking=enable_thinking,
                        r=task['discount_factor']
                    )
                    for task in chunk_tasks
                }

                for future in tqdm(concurrent.futures.as_completed(future_to_task),
                                total=len(chunk_tasks), desc="Processing tasks"):
                    result = future.result()
                    with results_lock:
                        if result is not None:
                            results.append(result)
                            if len(results) % 2 == 0:
                                with open(output_file, 'w') as f:
                                    json.dump(results, f, indent=4)

        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
    
    if "accuracy" in results[0]:
        accuracies = [res['accuracy'] for res in results if 'accuracy' in res]
        overall_accuracy = sum(accuracies) / len(accuracies)
        logger.info(f"Overall accuracy ({sum(accuracies)} / {len(accuracies)}): {overall_accuracy:.4f}")
    if "discounted_reward" in results[0]:
        for key in ["discounted_reward", "num_retrieves", "oracle_retrieve", "oracle_match_rate"]:
            if key not in results[0]:
                continue
            values = [res[key] for res in results if key in res]
            overall_value = sum(values) / len(values)
            logger.info(f"Overall {key} ({sum(values)} / {len(values)}): {overall_value:.4f}")
        


if __name__ == "__main__":
    main()



