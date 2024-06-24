"""Contains classes for querying local large language models."""

import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from agent.LLM import LLM

class Local_LLM(LLM):
    def __init__(self, model_config, model = None, tokenizer = None):
        # self.model_name = model_config["name"]
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["model_config"]["pretrained_model_name_or_path"])
        else:
            self.tokenizer = tokenizer
        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(**model_config["model_config"])
        else:
            self.model = model
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        try:
            self.tokenizer.apply_chat_template([{"role":"system","content":""}])
            self.tokenizer_has_system_prompt = True
        except:
            self.tokenizer_has_system_prompt = False

        self.generation_config = model_config["generation_config"]
        self.system_prompt = model_config["sys_prompt"]

    def generate(self, chat : List[dict], **kwargs) -> List[str]:
        tokens = self.tokenizer.apply_chat_template(
            chat, 
            tokenize = True, add_generation_prompt = True, return_tensors = "pt", return_attention_mask = True, return_dict = True
            ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**tokens, **self.generation_config, **kwargs)
        output = output[:, tokens["input_ids"].shape[-1]:]    # Only return generated tokens
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return decoded_output