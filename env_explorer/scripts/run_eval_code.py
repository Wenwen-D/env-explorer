import json
import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import threading
import yaml
import torch
import csv
from datetime import datetime
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.models.lite_llm import APIModel, LocalModel
from src.conversation.conversation_manager_code import ConversationManagerCode
from src.train.csv_calibrator.light_bert_format_predictor import load_data, FilenameFormatDataset, FormatPredictorTinyBERT


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


def load_generation_config(path):
    if not path:
        return {}
    if path.endswith((".yaml", ".yml")):
        with open(path) as f:
            cfg = yaml.safe_load(f)
    else:
        with open(path) as f:
            cfg = json.load(f)
    return cfg.get("generation_config", {})


def main():
    parser = argparse.ArgumentParser(description="Evaluate CSV reasoning tasks.")
    parser.add_argument("-m", "--model", required=True, help="Model name or path")
    parser.add_argument("-c", "--checkpoint", default=None, help="Checkpoint path for LoRA model")
    parser.add_argument("--remote_model", action="store_true", help="Use remote model API")
    parser.add_argument("--url", default=None, help="Remote model API URL if applicable")
    parser.add_argument("-t", "--task_metadata", required=True, help="CSV task metadata JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--max_turns", type=int, default=8, help="Maximum turns per task")
    parser.add_argument("-o", "--output", required=True, help="Output file suffix")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--config_file", type=str, default='./csv_gen_config.yaml', help="Path to generation config file (YAML or JSON)")
    parser.add_argument("--thinking", action="store_true", help="Use thinking mode (Qwen3)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--calibrator", default=None, required=False, help="Path to CSV format calibrator model")
    parser.add_argument("--rho", default=-1, type=float, help=(
        "Relative cost exponent between CODE and UNIT_TEST actions, "
        "where d_code = d_unit_test ** rho. "
        "Larger rho penalizes CODE more heavily; smaller rho favors early CODE attempts."
        "If set to -1, use the rho value specified in each task."
    )) # 2, 0.5. TODO: [0.5, 1.2], or {0.3, 0.6, 1.0, 1.5, 2.0}
    args = parser.parse_args()



    def get_output_identifier(args):
        id_think = "_think" if args.thinking else ""
        id_model = args.model.replace("/", "_").replace("-", "_")
        id_calib = f"_calib_{Path(args.calibrator).stem}" if args.calibrator else "_calib_none"
        id_turns = f"_turns{args.max_turns}"
        id_limit = f"_lim{args.limit}" if args.limit else ""
        id_rho = f"_rho{args.rho}" if args.rho else ""
        return f"{id_model}{id_turns}{id_think}{id_calib}{id_limit}{id_rho}__{args.output}"

    args.output = get_output_identifier(args)

    os.environ["LITELLM_LOG"] = "ERROR"
    setup_litellm_suppression()
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(f"logs/{args.output}_eval.log")],
    )

    logger.info("🚀 Starting CSV evaluation with %s", args.model)

    generation_config = load_generation_config(args.config_file)
    if args.remote_model:
        model = APIModel(model_name=args.model, base_url=args.url, generation_config=generation_config, logger=logger)
    else:
        model = LocalModel(model_path=args.model, device="cuda",
                           think_mode=args.thinking, checkpoint=args.checkpoint, logger=logger)
    
    if args.calibrator == 'oracle':
        calibrator_model = 'oracle'

    elif args.calibrator is not None:
        calibrator_model = FormatPredictorTinyBERT("prajjwal1/bert-tiny")
        state = torch.load(args.calibrator, map_location=DEVICE)
        calibrator_model.load_state_dict(state)
        calibrator_model.to(DEVICE)
        calibrator_model.eval()
    else:
        calibrator_model = None

    conversation_manager = ConversationManagerCode(model, max_turns=args.max_turns, verbose=args.verbose, logger=logger, enable_thinking =args.thinking, calibrator_model=calibrator_model, rho = args.rho,
                                                   base_dir=str(Path(args.task_metadata).parent))

    output_file = f"../output_code/{args.output}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # create timestamp to avoid overwriting
    if Path(output_file).exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../output_code/{args.output}_{timestamp}.json"
        logger.info(f"Output file already exists, using timestamped filename: {output_file}")

    tasks_all = json.load(open(args.task_metadata))
    if args.limit:
        tasks_all = tasks_all[:args.limit]

    if args.resume and Path(output_file).exists():
        completed = json.load(open(output_file))
        done_ids = {t["task_id"] for t in completed}
        tasks = [t for t in tasks_all if t["task_id"] not in done_ids]
        results = completed
        logger.info(f"Resuming: {len(done_ids)} done, {len(tasks)} remaining")
    else:
        tasks, results = tasks_all, []

    results_lock = threading.Lock()
    chunk_size = args.num_workers * 20
    logger.info(f"Total {len(tasks)} tasks, saving to {output_file}")

    for chunk_start in tqdm(range(0, len(tasks), chunk_size), desc="Task chunks"):
        chunk = tasks[chunk_start:chunk_start + chunk_size]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(conversation_manager.run_task, task, max_turns=args.max_turns): task
                for task in chunk
            }
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(chunk), desc="Processing"):
                res = fut.result()
                if res is None:
                    continue
                with results_lock:
                    results.append(res)
                    if len(results) % 5 == 0:
                        with open(output_file, "w") as f:
                            json.dump(results, f, indent=2)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("✅ Done! Results saved to %s", output_file)

    # compute overall accuracy
    correct = sum(1 for r in results if r.get("success", 0) == 1)
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    # compute overall reward
    total_reward = sum(r.get("reward", 0.0) for r in results)
    logger.info("Overall total reward: %.4f", total_reward)
    avg_reward = total_reward / total if total > 0 else 0.0
    logger.info("Overall avg reward: %.4f", avg_reward)
    # compute overall match rate
    total_match = sum(r.get("oracle_trace_match_rate", 0.0) for r in results)
    overall_match_rate = total_match / total if total > 0 else 0.0
    logger.info("Overall oracle trace match rate: %.4f", overall_match_rate)

    # log num_turns, len(helper_call_trace), len(code_attempt_trace)
    avg_turns = sum(r.get("num_turns", 0) for r in results) / total if total > 0 else 0.0
    logger.info("Average num_turns: %.2f", avg_turns)

    avg_helper_calls = sum(len(r.get("helper_call_trace", [])) for r in results) / total if total > 0 else 0.0
    logger.info("Average helper calls: %.2f", avg_helper_calls)
    
    avg_code_attempts = sum(len(r.get("code_attempt_trace", [])) for r in results) / total if total > 0 else 0.0
    logger.info("Average code attempts: %.2f", avg_code_attempts)

    # Log results to CSV file
    csv_summary_path = "../output_code/performance_summary.csv"
    os.makedirs(os.path.dirname(csv_summary_path), exist_ok=True)

    # Get output identifier with timestamp from the output file name
    output_identifier_with_timestamp = Path(output_file).stem

    # Check if file exists and has content
    file_exists = Path(csv_summary_path).exists()
    is_empty = not file_exists or Path(csv_summary_path).stat().st_size == 0

    with open(csv_summary_path, 'a', newline='') as csvfile:
        fieldnames = ['output_identifier', 'accuracy', 'avg_reward', 'oracle_match_rate',
                      'avg_num_turns', 'avg_helper_calls', 'avg_code_attempts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is new or empty
        if is_empty:
            writer.writeheader()

        # Write the data row
        writer.writerow({
            'output_identifier': output_identifier_with_timestamp,
            'accuracy': f"{accuracy:.6f}",
            'avg_reward': f"{avg_reward:.6f}",
            'oracle_match_rate': f"{overall_match_rate:.4f}",
            'avg_num_turns': f"{avg_turns:.2f}",
            'avg_helper_calls': f"{avg_helper_calls:.2f}",
            'avg_code_attempts': f"{avg_code_attempts:.2f}"
        })

    logger.info(f"📊 Performance summary logged to {csv_summary_path}")



if __name__ == "__main__":
    main()



