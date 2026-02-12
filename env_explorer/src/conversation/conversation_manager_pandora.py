from pathlib import Path
from typing import List, Optional
from src.data.data_types import TaskInstance, TaskResult, ConversationTurn, ExecutionResult
from src.models.base_model import BaseModel
from src.execute_code import CodeExecutor
from src.execute_code import CodeExtractor
from src.evaluation.task_evaluator import TaskEvaluator
from src.conversation.pandora_prompts import BANDIT_SYSTEM_PROMPT, BANDIT_SYSTEM_PROMPT_NO_EXPLAIN

import re
import random
import logging
import json
import os


class ConversationManager_pandora:
    def __init__(self, model: BaseModel,
                 max_turns: int = 3, verbose: bool = False, logger: Optional[logging.Logger] = None, system_choice = 'default'):
        self.model = model

        self.max_turns = max_turns
        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)
        self.system_choice = system_choice
        if self.system_choice == 'default':
            self.system_prompt = BANDIT_SYSTEM_PROMPT
        elif self.system_choice == 'no_explain':
            self.system_prompt = BANDIT_SYSTEM_PROMPT + BANDIT_SYSTEM_PROMPT_NO_EXPLAIN
        else:
            raise ValueError(f"Unknown system choice: {self.system_choice}")
    
    
    def extract_bandit_action(self, response: str, arm_labels):
        """Extract the bandit action from the model response.
        The action should be in the format "Action: VERIFY <action>" or "Action: GUESS <action>", where <action> is the arm label.
        """
        # find Action: in response and parse the action following it:
        if 'Action:' in response:
            response = response.split('Action:')[-1]
        if '</think>' in response:
            response = response.split('</think>')[-1]
        response = response.strip().upper()
        action_param = response.split(' ')
        if len(action_param) < 2:
            return None, None
        if action_param[0] not in ['VERIFY', 'GUESS']:
            return None, None
        if action_param[1] not in arm_labels:
            return None, None
        return action_param[0], action_param[1]

    def provide_feedback(self, task, pred_action, pred_choice, arm_labels, timestep):
        if pred_action is None or pred_choice is None:
            ABC_or = " or ".join(arm_labels).strip()
            return f"Invalid action. Please respond with 'Action: VERIFY {ABC_or}' or 'Action: GUESS {ABC_or}'.", False
        if pred_action == "VERIFY":
            if pred_choice == task['env']['true_arm']:
                return f"The verification result is: YES, {pred_choice} is correct. Given this, please provide your next action.\n TIMESTEP: t={timestep+1}\nChoose your action.", False
            else:
                return f"The verification result is: NO, {pred_choice} is incorrect. Given this, please provide your next action.\n TIMESTEP: t={timestep+1}\nChoose your action.", False
        elif pred_action == "GUESS":
            if pred_choice == task['env']['true_arm']:
                return f"Your guess {pred_choice} is CORRECT. Task completed.", True
            else:
                return f"Your guess {pred_choice} is INCORRECT. The correct answer was {task['env']['true_arm']}. Task completed.", True
        

    def run_task(self, task, instruction, max_turns = None, 
                 enable_thinking=False):
        # if enable_thinking:
            # print("line 396 🧠 enable thinking")
        res = self.run_task_default(task, instruction, max_turns, enable_thinking=enable_thinking)
        
        # if add_no_think:
        #     raw_conversation = res['conversation']
        #     for cc in raw_conversation:
        #         if cc['role'] == "user":
        #             cc['content'] += " /no_think"
        #     res['conversation'] = raw_conversation
        # else:
        #     print("🧠 enable thinking")
        return res

    def run_task_default(self, task, instruction, max_turns = None, enable_thinking = False):
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
            print(f"🎰 Running bandit task {task['task_id']}:")
            print(f"--system prompt: {self.system_prompt}")
            print(f"--instruction: {instruction}")

        if max_turns is None:
            max_turns = self.max_turns
        
        # print("line 432 max_turns", max_turns)
    
        for turn in range(max_turns):
            if self.verbose:
                print(f"--- Turn {turn} ---")
            # Generate response (with guide on first turn if requested)
            response = self.model.generate_response(messages, enable_thinking=enable_thinking)
            pred_action, pred_choice = self.extract_bandit_action(response, task['env']['labels'])
            user_feedback, is_completed = self.provide_feedback(task, pred_action, pred_choice, task['env']['labels'], turn)      
            model_action_history.append( (pred_action, pred_choice) )      
            if self.verbose:
                print(f"Turn {turn} assistant response: {response}")
                print(f"Turn {turn} user feedback: {user_feedback}")
            

            messages.append(ConversationTurn(role="assistant", content=response))
            messages.append(ConversationTurn(role="user", content=user_feedback))
            if is_completed:
                print(" Breaking as is_completed is True ")
                break
            else:
                print(" Continuing as is_completed is False ")
        
        # print("line 468 after turns")
        tt = dict(
            task_id=task['task_id'],
            conversation=[m.model_dump() for m in messages],
            num_turns=len([m for m in messages if m.role == "assistant"]),
            action_history=model_action_history,
            task=task
        )
        return tt


