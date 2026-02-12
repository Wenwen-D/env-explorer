# src/models/lite_llm.py
import torch
import os
import litellm
litellm.verbose = False

from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseModel
from typing import List, Dict, Any, Optional
from src.data.data_types import ConversationTurn
import logging
# Get the litellm logger by its specific name
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("litellm.proxy").setLevel(logging.WARNING)
logging.getLogger("litellm.router").setLevel(logging.WARNING)
logging.getLogger("litellm.utils").setLevel(logging.WARNING)
from peft import PeftModel
# litellm.set_verbose(False)

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class LocalModel(BaseModel):
    def __init__(self, model_path: str, device: str = "cuda",
                 checkpoint: Optional[str] = None,
                 generation_config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None,
                 think_mode: bool = False):
        
        self.logger = logger or logging.getLogger(__name__)


        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.logger.info(f"Loading model from {model_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.checkpoint = checkpoint
        if checkpoint:
            try:
                self.model = PeftModel.from_pretrained(base_model, checkpoint)
                self.logger.info("Successfully loaded LoRA checkpoint")
            except Exception as e:
                self.logger.warning(f"Failed to load LoRA checkpoint: {e}")
                self.logger.info("Falling back to base model")
                self.model = base_model
        else:
            self.model = base_model
            
        if 'qwen3' in model_path.lower() and think_mode:
            self.default_generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 32768,
                "do_sample": True,

            }
        else:
            self.default_generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 2048,
                "do_sample": True,
            }
        
        self.think_mode = think_mode
        self.generation_config = self.default_generation_config
        if generation_config:
            self.generation_config.update(generation_config)
        
        self.logger.info(f"Initialized LocalModel with model {model_path} on device {device}")
        self.logger.info(f"Using generation config: {self.generation_config}")
        self.logger.info(f"(Apply to Qwen-3) Think mode is {'enabled' if think_mode else 'disabled'}")
    

    def generate_response(self, messages: List[ConversationTurn],
                          generation_config: Optional[Dict[str, Any]] = None) -> str:
        # Convert to chat format
        chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        if 'qwen3' in self.model.name_or_path.lower() and self.think_mode:

            text = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        elif 'qwen3' in self.model.name_or_path.lower() and not self.think_mode:
            # print("Using Qwen-3 model WITHOUT thinking mode")
            text = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            text = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            **(generation_config or self.generation_config)
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# src/models/api_model.py
# from openai import OpenAI
# from together import Together
# import os
# from IPython.display import display, Markdown, Latex
# from vllm import LLM, SamplingParams

# g_client = OpenAI(
#     base_url="http://129.114.17.34:8000/v1",
#     api_key="EMPTY",
# )

# t_client = Together(api_key="EMPTY",) #os.environ['TOGETHER_API_KEY'])

# d_client = OpenAI(
#     api_key="EMPTY", #os.environ['DEEPSEEK_API_KEY'],
#     base_url="https://api.deepseek.com",
# )

os.environ['LITELLM_LOG'] = 'ERROR'  # Set logging level for LiteLLM

class APIModel(BaseModel):
    def __init__(self, model_name: str,
                 generation_config: Optional[Dict[str, Any]],
                 base_url: Optional[str] = None, logger: Optional[logging.Logger] = None,
                 think_mode: bool = False):
        # Set dummy OPENAI_API_KEY if not already set (required for litellm)
        if 'OPENAI_API_KEY' not in os.environ:
            os.environ['OPENAI_API_KEY'] = 'EMPTY'

        self.model_name = model_name
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)
        # self.generation_config = generation_config
        if 'claude' in model_name.lower():
            self.default_generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048,
            }
        else:
            self.default_generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048,
                "do_sample": True,
            }
        self.generation_config = self.default_generation_config
        self.think_mode = think_mode
        if generation_config:
            self.generation_config.update(generation_config)
        print(f"Generation config: {self.get_generation_config()}")
    
        # If base_url is provided, configure litellm for custom endpoint
        if base_url:
            # You can configure litellm to use custom base URLs
            litellm.api_base = base_url
        self.logger.info(f"Initialized APIModel with model {model_name} at url='{base_url or 'default endpoint'}'")
        self.logger.info(f"Using generation config: {self.generation_config}")

    def update_generation_config(self, new_config: Dict[str, Any]):
        """Update generation configuration"""
        self.generation_config.update(new_config)
    def get_generation_config(self) -> Dict[str, Any]:
        """Get current generation configuration"""
        return self.generation_config.copy()


    def generate_response(self, messages: List[ConversationTurn], generation_config: Optional[Dict[str, Any]] = None, enable_thinking:bool = False) -> str:
        # print("🔥 generating response")
        if 'claude' in self.model_name.lower():
            # print("🔥 claude")
            return self.generate_response_with_claude(messages, generation_config)
        else:
            # print("🔥 not claude")
            return self.generate_response_url(messages, generation_config, enable_thinking=enable_thinking)
        
    def generate_response_url(self, messages: List[ConversationTurn],
                          generation_config: Optional[Dict[str, Any]] = None, enable_thinking:bool = False) -> str:
        try:
            chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        except Exception as e:
            print("🔥 Error converting messages to chat format:", e)
            print("🔥🔥🔥🔥", messages,"🔥🔥🔥🔥")
            raise e

        # add "/no_think" to the last user prompt in chat_messages if the model is Qwen-3 and think_mode is False
        if 'qwen3' in self.model_name.lower() and not enable_thinking:
            if chat_messages and chat_messages[-1]['role'] == 'user':
                chat_messages[-1]['content'] += " /no_think"
                # print(f"added '/no_think' to last user prompt: {chat_messages[-1]['content']}")
            else:
                print("🔥 Warning: No user message found to append '/no_think'")
                print(f"Current chat messages: {chat_messages}")
                # raise ValueError("No user message found to append '/no_think'")
        # else:
        #     print(f"🔥 enable_thinking is set to {enable_thinking} for model {self.model_name}")
        generation_config = generation_config or self.generation_config

        cur_gen_config = self.generation_config.copy() if generation_config is None else generation_config

        
        server_url = "http://{}/v1".format(self.base_url) if self.base_url else None
        try:
            response = litellm.completion(
                model='openai/'+self.model_name,  # Use the model name without path
                messages=chat_messages,
                api_base=server_url,  # Pass server_url if provided
                **cur_gen_config,
            )
            # print(f"enable_thinking: {enable_thinking}", response, '\n', f"response={response.choices[0].message.content}")
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error generating response with {self.model_name}: {e}")


    def generate_response_with_claude(self, messages: List[ConversationTurn],
                                    generation_config: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Claude models via Anthropic API"""
        
        chat_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Use provided config or fall back to instance config
        config = generation_config or self.generation_config
        
        cur_gen_config = self.generation_config.copy() if generation_config is None else generation_config

        try:
            # Use litellm to call Claude
            response = litellm.completion(
                model=self.model_name,
                messages=chat_messages,
                **cur_gen_config,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating response with Claude model {self.model_name}: {e}")
            raise RuntimeError(f"Error generating response with Claude {self.model_name}: {e}")