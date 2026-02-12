# scripts/run_evaluation.py
import json
import argparse
from pathlib import Path
from tqdm import tqdm

from src.models.lite_llm import APIModel, LocalModel
from src.data.task_loader import TaskLoader
from src.data.data_types import TaskInstance, TaskResult, ConversationTurn
from src.conversation.conversation_manager import ConversationManager_popqa, RetrievedDict
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
from src.conversation.popqa_prompts import POPQA_DIRECT, POPQA_ASK_CONFIDENCE_SYS, POPQA_ASK_CONFIDENCE_USER, POPQA_ASK_CONFIDENCE_SYS_RET, POPQA_ASK_CONFIDENCE_USER_RET
# from src.conversation.popqa_prompts import POPQA_DIRECT


# def compute_accuracy(results)
    # possible_answers = json.loads(row.possible_answers)        
    #     is_correct = False
    #     genread_has_answer = False
    #     for pa in possible_answers:
    #         if pa in pred or pa.lower() in pred or pa.capitalize() in pred:
    #             is_correct = True
    #         if args.eval_method == "genread" and pa in response or pa.lower() in response or pa.capitalize() in response:
    #             genread_has_answer = True
    #     accuracy.append(is_correct)
    #     if args.eval_method == "genread":
    #         has_answer.append(genread_has_answer)

# def form_instruction_t2(task):
#     labels_str = ', '.join(task['env']['labels'])
#     priors_str = ', '.join([f"{k}: {v}" for k, v in task['env']['priors'].items()])
#     r = task['discount_factor']
#     instruction = INSTRUCTION_TEMPLATE_t2.format(labels_str=labels_str, priors_str=priors_str, r=r)
#     # print("line28",type(instruction))
#     return instruction


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
    parser.add_argument("--system_choice", type=str, default='default', choices=['default', 'extended', 'draft', 'draft_v1', 'cost'], help="System prompt choice")
    parser.add_argument("--config_file", type=str, default=None, help="Generation config file")
    parser.add_argument("--resume", action='store_true', help="Resume from existing output file if exists")
    parser.add_argument("--do_retrieve", action='store_true', help="Use retrieval-augmented generation")
    # parser.add_argument("--num_passages", type=int, default=1, help="Number of retrieved passages to use")
    # parser.add_argument("--calibration_sample", type=int, default=0, help="Number of samples for calibration evaluation")
    # parser.add_argument("--config", default="configs/model_configs.yaml", help="Config file")
    
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

    # conversation_manager = ConversationManager_popqa(model, logger=logger,max_turns=args.max_turns, verbose=args.verbose, system_choice = args.system_choice)
    if args.do_retrieve:
        # NOTE: for _target_ret, we use the target wiki source for retrieval, otherwise used the commented line of code.
        do_retrieve = True
    else:
        # retrieved_dict = None
        do_retrieve = False
    output_file = f"../output_popqa/ask_confidence/{args.output}/results.json"

    def ask_confidence(task, enable_thinking, verbose=False,):
        if not do_retrieve:
            messages = [
                ConversationTurn(role="system", content=POPQA_ASK_CONFIDENCE_SYS),
                ConversationTurn(role="user", content=POPQA_ASK_CONFIDENCE_USER.format(question=task['question'])),
            ]
        else:
            raise NotImplementedError("Retrieval-augmented confidence asking not supported.")
            # context_passages = retrieved_dict.passages.get(task['task_id'], [])
            # # context_text = context_passages[0] if context_passages else "No relevant context found."
            # context_text = "\n".join(context_passages[:num_passages]).strip() if context_passages else "No relevant context found."
            # messages = [
            #     ConversationTurn(role="system", content=POPQA_ASK_CONFIDENCE_SYS_RET),
            #     ConversationTurn(role="user", content=POPQA_ASK_CONFIDENCE_USER_RET.format(question=task['question'], context=context_text)),
            # ]
        response = model.generate_response(messages, enable_thinking=enable_thinking)
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()
        try:
            confidence = float(response)
        except ValueError:
            confidence = 0.0  # default if parsing fails
        if verbose:
            logger.info(f"messages: {messages}")
            logger.info(f"Question: {task['question']}")
            logger.info(f"Model response: {response}")
            logger.info(f"Parsed confidence: {confidence}")
        tt = {'task_id': task['task_id'], 'question': task['question'], 'confidence': confidence, 'task': task}
        return tt

    if Path(output_file).exists(): # TODO
        # TODO:
        # raise ValueError(f"Output file {output_file} already exists. Please choose a different output name to avoid overwriting.")
        logger.warning(f"Output file {output_file} already exists. Loading for eval only...")
        with open(output_file, 'r') as f:
            results = json.load(f)
    else:
        # raise ValueError(f"Output file {output_file} does not exist. Please run the evaluation first.")
        # raise ValueError(f"Output file {output_file} does not exist. Please run the evaluation first.")

        tasks = json.load(open(args.task_metadata, 'r'))
        if args.limit is not None:
            tasks = tasks[:args.limit]
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
        # raise ValueError(f"Debugging stop here")
        # for task in tqdm(tasks, desc="Processing tasks"):
        #     result = conversation_manager.run_task(task, use_guide=args.guide,check_before_final=args.final_check)
        #     results.append(result)
            
        #     # Save incrementally
        #     output_file = f"../output/{args.output}_results.json"
        #     with open(output_file, 'w') as f:
        #         # json.dump([r.dict() for r in results], f, indent=4)
        #         json.dump([r.model_dump() for r in results], f, indent=4)

        chunk_size = args.num_workers * 20 
        enable_thinking = 'qwen3' in args.model.lower() and args.thinking
        print(f"🧠 enable_thinking: {enable_thinking}")
        logger.info(f"Chunk size: {chunk_size}, enable_thinking: {enable_thinking}")
        for chunk_start in tqdm(range(0, len(tasks), chunk_size), desc="Processing task chunks"): # change 1 to len(tasks)
            chunk_tasks = tasks[chunk_start:chunk_start + chunk_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_task = {
                    executor.submit(ask_confidence, task, enable_thinking=enable_thinking, verbose=args.verbose): task
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
