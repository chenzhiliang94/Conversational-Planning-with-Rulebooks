from reward.Base_Reward import Base_Reward
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from agent.Conversation import Conversation

# Use Meta-Llama-Guard-2-8B safe probability as reward model.
# The reward is the difference between the safe probability of the chat with the human response and the safe probability of the chat without the human response.
class Llama_2_Guard_Reward(Base_Reward):
    def __init__(self, device_map : int = 0) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-Guard-2-8B",
            torch_dtype = torch.bfloat16,
            device_map=device_map,
            )
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-Guard-2-8B")

        # Get the indices for the safe and unsafe tokens
        self.safe_unsafe_indices = self.tokenizer("safeunsafe", return_attention_mask=False)["input_ids"]

        # Unsafe prompt to force generation of an unsafe category
        self.unsafe_prompt = self.tokenizer("unsafe\nS", return_tensors="pt", return_attention_mask=False)['input_ids'].to(self.model.device)

        # Get the indices for the categories (1 to 11)
        category_indices = self.tokenizer(list(map(str, range(1,12))), return_attention_mask=False)["input_ids"]
        self.category_indices = [i[0] for i in category_indices]

    # Get the probability of the chat being safe
    def get_safe_prob(self, chat : List[dict] | List[List[dict]]) -> float | list[float]:
        is_list_of_chat = isinstance(chat[0], list)
        if not is_list_of_chat:
            chat = [chat]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)
        safe_unsafe_logits = self.model(input_ids=input_ids, return_dict=True)['logits'][:,-1, self.safe_unsafe_indices].detach().cpu()
        # safe_unsafe_logits = safe_unsafe_logits
        safe_unsafe_probs = torch.nn.Softmax(dim=-1)(safe_unsafe_logits)
        if not is_list_of_chat:
            return safe_unsafe_probs[0,0].item()
        return safe_unsafe_probs[:,0]

    # Get the probabilities of the unsafe categories
    def get_unsafe_categories_probs(self, chat : List[dict] | List[List[dict]]) -> float | list[float]:
        is_list_of_chat = isinstance(chat[0], list)
        if not is_list_of_chat:
            chat = [chat]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)

        # Append unsafe prompt to the input_ids
        input_ids_unsafe = torch.cat((input_ids, self.unsafe_prompt.repeat(input_ids.shape[0],1)), dim=1)

        # Generate unsafe category probabilities
        unsafe_logits = self.model(input_ids=input_ids_unsafe, return_dict=True)['logits'][:,-1].detach().cpu()[:,self.category_indices]
        unsafe_probs = torch.nn.Softmax(dim=-1)(unsafe_logits)
        if not is_list_of_chat:
            return unsafe_probs[0,:]
        return unsafe_probs[:,:]

    def get_reward(self, prev_state : Conversation, action : str | None, human_response : str | None) -> float:
        return self.get_safe_prob((prev_state + action + human_response).create_chat()) - self.get_safe_prob(prev_state.create_chat())

    def get_embedding(self, chat : List[dict] | List[List[dict]]) -> torch.tensor:
        is_list_of_chat = isinstance(chat[0], list)
        if not is_list_of_chat:
            chat = [chat]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)

        # Append unsafe prompt to the input_ids
        input_ids_unsafe = torch.cat((input_ids, self.unsafe_prompt.repeat(input_ids.shape[0],1)), dim=1)

        # Generate unsafe category probabilities
        last_embeddings = self.model(input_ids=input_ids_unsafe, output_hidden_states = True)['hidden_states'][-1].detach().cpu().float()
        embedding = torch.cat((last_embeddings[:,-4,:], last_embeddings[:,-1,:]), dim=-1)   # Get embedding for safe/unsafe and unsafe category
        if not is_list_of_chat:
            embedding = embedding[0,:]
        return embedding

    def get_safe_prob_from_embedding(self, embedding : torch.tensor) -> float:
        embedding = embedding.to(self.model.device).bfloat16()
        safe_unsafe_embedding = embedding[:self.model.lm_head.in_features]
        safe_unsafe_logits = self.model.lm_head(safe_unsafe_embedding)[self.safe_unsafe_indices].detach().cpu().float()
        safe_unsafe_probs = torch.nn.Softmax(dim=-1)(safe_unsafe_logits)
        return safe_unsafe_probs[0].item()

    def get_unsafe_categories_probs_from_embedding(self, embedding : torch.tensor) -> float:
        embedding = embedding.to(self.model.device).bfloat16()
        unsafe_embedding = embedding[self.model.lm_head.in_features:]
        unsafe_logits = self.model.lm_head(unsafe_embedding)[self.category_indices].detach().cpu().float()
        unsafe_probs = torch.nn.Softmax(dim=-1)(unsafe_logits)
        return unsafe_probs