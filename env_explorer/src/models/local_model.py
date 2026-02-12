import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseModel
from typing import List, Dict, Any
from src.data.data_types import ConversationTurn

class VLLMModel(BaseModel):
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    def generate_response(self, messages: List[ConversationTurn]) -> str:
        # Convert to chat format
        chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        text = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids, 
            max_new_tokens=1024, 
            do_sample=True, 
            top_p=0.9, 
            temperature=0.7
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]