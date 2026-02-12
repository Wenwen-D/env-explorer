from pathlib import Path
from typing import List, Optional
from src.data.data_types import TaskInstance, TaskResult, ConversationTurn, ExecutionResult
from src.models.base_model import BaseModel
from src.execute_code import CodeExecutor
from src.execute_code import CodeExtractor
from src.evaluation.task_evaluator import TaskEvaluator
from src.conversation.popqa_prompts import POPQA_DIRECT, POPQA_DIRECT_WITH_RET, POPQA_DIRECT_WITH_RET_USER
from src.conversation.popqa_prompts import POPQA_MULTI_SYS, POPQA_MULTI_T1, POPQA_MULTI_T2, POPQA_MULTI_SYS_NOPRIOR, POPQA_MULTI_T1_NOPRIOR, POPQA_MULTI_T2_NOPRIOR

import re
import random
import logging
import json
import os

class RetrievedDict:
    def __init__(self, passages_dir):
        """
        passages_dir: either a path to a JSONL file,
                      or a list of paths to JSONL files.
        Each line in the file should be a JSON object containing:
            {
              "task_id": ...,
              "top_passages": ...
            }
        """
        if isinstance(passages_dir, str):
            self.passages = self.load_passages_from_file(passages_dir)
        elif isinstance(passages_dir, (list, tuple)):
            self.passages = {}
            for path in passages_dir:
                partial = self.load_passages_from_file(path)
                # later files overwrite duplicate task_ids
                self.passages.update(partial)
        else:
            raise ValueError("passages_dir must be a string or a list of strings.")

    def load_passages_from_file(self, path):
        """
        Load one JSONL file and return a dict:
            {task_id: top_passages}
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        passages = {}
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_no} in {path}: {e}")

                task_id = obj.get("task_id")
                top_passages = obj.get("top_passages")
                if task_id is None:
                    raise ValueError(f"Missing 'task_id' at line {line_no} in {path}")
                passages[task_id] = top_passages
        return passages


class ConversationManager_popqa:
    def __init__(self, model: BaseModel,
                 max_turns: int = 3, verbose: bool = False, logger: Optional[logging.Logger] = None, system_choice = 'default', do_retrieve: bool = False, num_passages: Optional[int] = None, p_ret: float = 0.57,
                 confidence_field: str = "confidence_calibrated_think_platt"):
        self.model = model

        self.max_turns = max_turns
        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)
        self.system_choice = system_choice
        self.do_retrieve = do_retrieve
        self.num_passages = num_passages
        self.p_ret = p_ret
        self.confidence_field = confidence_field
        if max_turns == 1:
            if self.do_retrieve:
                self.system_prompt = POPQA_DIRECT_WITH_RET
            else:
                self.system_prompt = POPQA_DIRECT
        elif max_turns ==2:
            if system_choice == 'noprior':
                # print(" Setting POPQA_MULTI_SYS_NOPRIOR - self.p_ret:", self.p_ret)
                self.system_prompt = POPQA_MULTI_SYS_NOPRIOR.format(retrieval_accuracy=self.p_ret)
            else:
                self.system_prompt = POPQA_MULTI_SYS.format(retrieval_accuracy=self.p_ret)
        else:
            raise ValueError(f"Only max_turns 1 or 2 are supported in POPQA, got {max_turns}.")
        
    def extract_popqa_action(self, response: str, is_direct=False):
        # print("⚠️⚠️ is direct:", is_direct)
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()

        if is_direct:
            pred_action = 'ANSWER'
            pred_answer = response.strip()
            return pred_action, pred_answer
        else:
            if response.upper().startswith('RETRIEVE'):
                pred_action = 'RETRIEVE'
                pred_answer = ''
                return pred_action, pred_answer
            elif response.upper().startswith('ANSWER'):
                pred_action = 'ANSWER'
                pred_answer = response[len('ANSWER'):].strip()
                if ':' in pred_answer:
                    pred_answer = pred_answer.split(':', 1)[-1].strip()
                return pred_action, pred_answer
            else:
                return 'ANSWER', response.strip()

    def provide_feedback(self, task, pred_action, pred_choice, timestep):
        # print("provide_feedback line 597", pred_action, pred_choice)
        if pred_action is None or pred_choice is None:
            feedback = ("Invalid answer. Please respond with 'RETRIEVE: {query}' or 'ANSWER: {answer}'."
                        f"Please provide your next action.\n TIMESTEP: t={timestep+1}\n"
                    )
            return feedback, False
        if pred_action == "RETRIEVE":
            # context_passages = self.retrieved_dict.passages.get(task['task_id'], [])
            context_passages = self.get_retrieval_context(task, num_passages=self.num_passages)
            context_text = context_passages if isinstance(context_passages, str) else "No relevant context found." 
            feedback = POPQA_MULTI_T2.format(context=context_text, question=task['question'])
            return feedback, False
        elif pred_action == "ANSWER":
            return f"Your answer: {pred_choice}. Correct answer: {task['possible_answers']}. Task completed.", True
        else:
            print("UNKNOWN error line 541")
            raise NotImplementedError(f"provide feedback: Unknown action {pred_action} in POPQA.")
    def check_popqa_answer(self, task, model_action_history):
        """Check if the final answer in model_action_history matches the ground truth answers."""
        if not model_action_history:
            return 0
        final_action, final_answer = model_action_history[-1]
        if final_action != 'ANSWER':
            return 0
        possible_answers = task.get('possible_answers', [])
        for pa in possible_answers:
            if pa in final_answer or pa.lower() in final_answer.lower() or pa.capitalize() in final_answer or pa.lower() in final_answer:
                return 1
        return 0
    
    def get_retrieval_context(self, task, num_passages=None):
        ctxs = task.get('ctxs', [])
        if not ctxs:
            return "No relevant context found."
        if num_passages is None:
            num_passages = len(ctxs)
        sorted_ctxs = sorted(ctxs, key=lambda x: float(x["score"]), reverse=True)[:num_passages]
        context_with_title = [f"Title: {c['title']}\nContent: {c['text']}" for c in sorted_ctxs]
        return "\n\n".join(context_with_title).strip()



    def run_task(self, task, instruction, 
                 enable_thinking=False, r = 1):
        num_passages = self.num_passages
        if self.do_retrieve:
            # context_passages = self.retrieved_dict.passages.get(task['task_id'], [])
            context_passages = self.get_retrieval_context(task, num_passages=num_passages)
            context_text = context_passages if isinstance(context_passages, str) else "No relevant context found." 
            # # context_text = context_passages[0] if context_passages else "No relevant context found."
            # context_text = "\n".join(context_passages[:num_passages]).strip() if context_passages else "No relevant context found."
            instruction = POPQA_DIRECT_WITH_RET_USER.format(context=context_text, question=instruction)
        if self.max_turns == 1:
            res = self.run_task_direct(task, instruction, enable_thinking=enable_thinking)
        else:
            res = self.run_task_multi_turn(task, r=r, enable_thinking=enable_thinking)
        return res

    def run_task_direct(self, task, instruction, enable_thinking = False):
        """Run a single task with multi-turn conversation
        - every turn of conversation is associated with an execution result, including the final turn
        """
        # print("line411", type(self.system_prompt))
        messages = [
            ConversationTurn(role="system", content=self.system_prompt),
            ConversationTurn(role="user", content=instruction)
        ]

        # reasoning_history = []
        model_action_history = []
        if self.verbose:
            print(f"🌐 Running PopQA task {task['task_id']}:")
            print(f"--system prompt: {self.system_prompt}")
            print(f"--instruction: {instruction}")


        max_turns = self.max_turns
    
        for turn in range(max_turns):
            if self.verbose:
                print(f"--- Turn {turn} ---")
            # Generate response (with guide on first turn if requested)
            response = self.model.generate_response(messages, enable_thinking=enable_thinking)
            pred_action, pred_answer = self.extract_popqa_action(response, is_direct=(max_turns==1))
            # pred_action, pred_choice = self.extract_bandit_action(response, task['env']['labels'])
            user_feedback, is_completed = self.provide_feedback(task, pred_action, pred_answer, turn)      
            model_action_history.append( (pred_action, pred_answer) )      
            if self.verbose:
                print(f"Turn {turn} assistant response: {response}")
                print(f"Turn {turn} user feedback: {user_feedback}")
            

            messages.append(ConversationTurn(role="assistant", content=response))
            messages.append(ConversationTurn(role="user", content=user_feedback))
            if is_completed:
                break
        
        accuracy = self.check_popqa_answer(task, model_action_history)
        tt = dict(
            task_id=task['task_id'],
            conversation=[m.model_dump() for m in messages],
            num_turns=len([m for m in messages if m.role == "assistant"]),
            action_history=model_action_history,
            task=task,
            accuracy=accuracy,
            r = 1,
        )
        # print("🌐 popqa tt", tt)
        return tt

    def run_task_multi_turn(self, task, r, enable_thinking = False):
        """Run a single task with multi-turn conversation
        - every turn of conversation is associated with an execution result, including the final turn
        """
        # print("line411", type(self.system_prompt))
        p_ret = self.p_ret
        if self.system_choice == 'noprior':
            instruction_1 = POPQA_MULTI_T1_NOPRIOR.format(question=task['question'], r=r)
        else:
            # print("🌐🌐🌐🌐🌐🌐 900", task.keys())
            # print("🌐🌐🌐🌐🌐🌐 900", task[self.confidence_field])
            instruction_1 = POPQA_MULTI_T1.format(question=task['question'], r=r, 
                                            p_no_context=f"{task[self.confidence_field]:.2f}",
                                            # p_with_context=f"{p_ret:.2f}"
                                            )
        messages = [
            ConversationTurn(role="system", content=self.system_prompt),
            ConversationTurn(role="user", content=instruction_1)
        ]

        # reasoning_history = []
        model_action_history = []
        if self.verbose:
            print(f"🌐 Running PopQA task {task['task_id']}:")
            print(f"--system prompt: {self.system_prompt}")
            print(f"--instruction: {instruction_1}")
            print(f"--messages so far: {messages}")


        max_turns = self.max_turns
        
    
        for turn in range(max_turns):
            if self.verbose:
                print(f"--- Turn {turn} ---")
            # Generate response (with guide on first turn if requested)
            response = self.model.generate_response(messages, enable_thinking=enable_thinking)
            pred_action, pred_answer = self.extract_popqa_action(response, is_direct=(max_turns==1))
            # pred_action, pred_choice = self.extract_bandit_action(response, task['env']['labels'])
            user_feedback, is_completed = self.provide_feedback(task, pred_action, pred_answer, turn)      
            model_action_history.append( (pred_action, pred_answer) )      
            if self.verbose:
                print(f"Turn {turn} assistant response: {response}")
                print(f"Turn {turn} user feedback: {user_feedback}")
            

            messages.append(ConversationTurn(role="assistant", content=response))
            messages.append(ConversationTurn(role="user", content=user_feedback))
            if is_completed:
                break
        
        accuracy = self.check_popqa_answer(task, model_action_history)
        if task[self.confidence_field] <= p_ret * r:
            # should call retrieve
            oracle_retrieve = 1
        else:
            oracle_retrieve = 0
        
        num_model_retrieves = sum(1 for a, _ in model_action_history if a == 'RETRIEVE')

        tt = dict(
            task_id=task['task_id'],
            conversation=[m.model_dump() for m in messages],
            num_turns=len([m for m in messages if m.role == "assistant"]),
            action_history=model_action_history,
            task=task,
            accuracy=accuracy,
            r = r,
            discounted_reward = accuracy * (r ** (len(model_action_history) - 1)),
            num_retrieves = num_model_retrieves,
            oracle_retrieve = oracle_retrieve,
            oracle_match_rate = num_model_retrieves == oracle_retrieve,

        )
        # print("🌐 popqa tt", tt)
        return tt




