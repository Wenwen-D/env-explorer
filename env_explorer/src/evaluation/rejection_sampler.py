# src/evaluation/rejection_sampler.py
from typing import List, Tuple
from src.data.data_types import TaskResult, TaskInstance
from src.conversation.conversation_manager import ConversationManager

class RejectionSampler:
    def __init__(self, conversation_manager: ConversationManager, num_samples: int = 5):
        self.conversation_manager = conversation_manager
        self.num_samples = num_samples
    
    def sample_good_traces(self, tasks: List[TaskInstance]) -> List[TaskResult]:
        """Sample good traces for training data"""
        good_traces = []
        
        for task in tasks:
            best_result = None
            best_score = -1
            
            # Generate multiple samples
            for _ in range(self.num_samples):
                result = self.conversation_manager.run_task(task)
                score = self._score_result(result)
                
                if score > best_score:
                    best_score = score
                    best_result = result
            
            if best_result and best_score > 0:  # Only keep successful traces
                good_traces.append(best_result)
        
        return good_traces
    
    def _score_result(self, result: TaskResult) -> float:
        """Score a task result for rejection sampling"""
        if not result.success:
            return 0.0
        
        # Higher score for fewer turns (more efficient)
        turn_penalty = max(0, result.num_turns - 2) * 0.1
        base_score = 1.0 - turn_penalty
        
        return max(0.0, base_score)