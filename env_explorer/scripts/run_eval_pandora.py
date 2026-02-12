# scripts/run_evaluation.py
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from src.models.lite_llm import APIModel, LocalModel
from src.data.task_loader import TaskLoader
from src.data.data_types import TaskInstance, TaskResult
from src.conversation.conversation_manager_pandora import ConversationManager_pandora
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
from datetime import datetime
from src.conversation.pandora_prompts import BANDIT_SYSTEM_PROMPT_t2, INSTRUCTION_TEMPLATE_t2

def form_instruction_t2(task):
    labels_str = ', '.join(task['env']['labels'])
    priors_str = ', '.join([f"{k}: {v}" for k, v in task['env']['priors'].items()])
    r = task['discount_factor']
    instruction = INSTRUCTION_TEMPLATE_t2.format(labels_str=labels_str, priors_str=priors_str, r=r)
    return instruction


def get_output_filename(args, logger):
    """Generate output filename based on model, thinking mode, turns, and optional identifier"""
    # Format model name (replace slashes and dashes with underscores)
    model_name = args.model.replace("/", "_").replace("-", "_")

    # Add thinking indicator
    think_indicator = "_think" if args.thinking else ""

    # Add max turns
    max_turns = f"_turns{args.max_turns}"

    # Add optional identifier
    output_id = f"_{args.output}" if args.output else ""

    # Construct base filename
    base_filename = f"{model_name}{max_turns}{think_indicator}{output_id}.json"
    output_file = f"../output_pandora/{base_filename}"

    # Handle existing file with timestamp
    if Path(output_file).exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../output_pandora/{model_name}{max_turns}{think_indicator}{output_id}_{timestamp}.json"
        logger.info(f"Output file already exists, using timestamped filename: {output_file}")

    return output_file


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
    parser.add_argument("--limit", type=int, default=20, help="Number of tasks to run")
    parser.add_argument("--max_turns", type=int, default=3, help="Maximum conversation turns")
    parser.add_argument("-o", "--output", default="", help="Optional identifier for output filename")
    parser.add_argument("--thinking", action='store_true', help="Use thinking mode (Qwen3 model)")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--config_file", type=str, default=None, help="Generation config file")

    args = parser.parse_args()

    if args.config_file:
        generation_config = load_generation_config(args.config_file)
    else:
        generation_config = None
    
    os.environ['LITELLM_LOG'] = 'ERROR'  # Set logging level for LiteLLM

    logger = logging.getLogger(__name__)
    os.makedirs("logs", exist_ok=True)

    # Generate log filename based on model and settings
    model_name = args.model.replace("/", "_").replace("-", "_")
    think_indicator = "_think" if args.thinking else ""
    max_turns = f"_turns{args.max_turns}"
    output_id = f"_{args.output}" if args.output else ""
    log_filename = f"logs/{model_name}{max_turns}{think_indicator}{output_id}_eval.log"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_filename)])
    logger.info("Starting evaluation with arguments: %s", args)
    logger.info(f"Generation config: {generation_config}")
    
    setup_litellm_suppression()



    if args.remote_model:
        logger.info(f"Using remote model API at {args.url}")
        model_class = APIModel
        model = model_class(model_name=args.model, logger=logger,
                            generation_config=generation_config, base_url=args.url)
    else:
        model_class = LocalModel
        model = model_class(model_path=args.model, device='cuda', logger=logger, think_mode=args.thinking, checkpoint=args.checkpoint)
 
    conversation_manager = ConversationManager_pandora(model, logger=logger,max_turns=args.max_turns, verbose=args.verbose)

    output_file = get_output_filename(args, logger)

    tasks = json.load(open(args.task_metadata, 'r'))
    # tasks = tasks[:1] # TODO: debugging now
    logger.info(f"Loaded {len(tasks)} tasks from {args.task_metadata}")
    logger.info(f"Sample task: {tasks[0]}")
    # print(tasks[0])
    # raise ValueError(f"Debugging stop here")
    

    # Run evaluation
    results = []
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
                executor.submit(conversation_manager.run_task, task, instruction=form_instruction_t2(task), enable_thinking=enable_thinking): task
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


if __name__ == "__main__":
    main()


# python run_eval_pandora.py -m Qwen/Qwen3-8B -t ../data/pandora_data/pandora_tasks.json  --remote_model --url "127.0.0.1:8000" 

# python run_eval_pandora.py -m Qwen/Qwen3-8B -t ../data/pandora_data/pandora_tasks.json  --remote_model --url "127.0.0.1:8000"  --thinking
