# src/models/base_model.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.data.data_types import ConversationTurn

class BaseModel(ABC):
    @abstractmethod
    def generate_response(self, messages: List[ConversationTurn]) -> str:
        """Generate a response given conversation history"""
        pass

