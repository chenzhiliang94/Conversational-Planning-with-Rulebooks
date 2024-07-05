from reward.Base_Reward import Base_Reward
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from agent.Conversation import Conversation

# Use Meta-Llama-Guard-2-8B safe probability as reward model.
# The reward is the difference between the safe probability of the chat with the human response and the safe probability of the chat without the human response.
class Llama_2_Guard_Reward(Base_Reward):
    def __init__(self, model = None, device_map : int = 0, random_projection : int | None = 1024, random_proj_seed = 42) -> None:
        try:
            assert model.name_or_path == "meta-llama/Meta-Llama-Guard-2-8B"
            self.model = model
        except:
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

        head_weights = self.model.lm_head.weight.detach().float()

        # Create diagonal block matrix
        safe_unsafe_matrix = head_weights[self.safe_unsafe_indices, :].t().cpu()
        category_matrix = head_weights[self.category_indices, :].t().cpu()

        zero_pad_A = torch.zeros((self.model.lm_head.in_features, len(self.safe_unsafe_indices)))
        zero_pad_B = torch.zeros((self.model.lm_head.in_features, len(self.category_indices)))
        self.proj_A = torch.cat((safe_unsafe_matrix, zero_pad_A), dim=0)
        self.proj_B = torch.cat((zero_pad_B, category_matrix), dim=0)

        if random_projection is not None and random_projection != self.model.lm_head.in_features * 2:
            assert random_projection < self.model.lm_head.in_features * 2, f"projection dimension must be smaller than {self.model.lm_head.in_features * 2}"
            B = torch.cat((self.proj_A, self.proj_B), dim=1)
            torch.manual_seed(random_proj_seed)
            A = torch.randn(random_projection, B.shape[1])
            self.projection_mat = torch.linalg.lstsq(A.T, B.T, driver="gelsd").solution.t().to(self.model.device)
            self.proj_A = A[:, :len(self.safe_unsafe_indices)]
            self.proj_B = A[:, len(self.safe_unsafe_indices):]
        torch.cuda.empty_cache()

    # Get the probability of the chat being safe
    def get_safe_prob(self, chat : List[dict] | List[List[dict]]) -> float | list[float]:
        embedding = self.embed(chat)
        return self.get_safe_prob_from_embedding(embedding)

    # Get the probabilities of the unsafe categories
    def get_unsafe_categories_probs(self, chat : List[dict] | List[List[dict]]) -> float | list[float]:
        embedding = self.embed(chat)
        return self.get_unsafe_categories_probs_from_embedding(embedding)

    # Get the reward delta for the action and human response
    def get_reward(self, prev_state : Conversation | tuple, action : str | tuple, human_response : str | tuple) -> float:
        if isinstance(prev_state, Conversation):
            print(human_response)
            score_after = self.get_safe_prob((prev_state + action + human_response).create_chat())
            score_before =  self.get_safe_prob(prev_state.create_chat())
            reward = (score_after - score_before)
            print("before multipling reward is ", reward)
            return 10 * reward
        else:
            reward = self.get_safe_prob_from_embedding(torch.FloatTensor((human_response))) - self.get_safe_prob_from_embedding(torch.FloatTensor((prev_state)))
            return 10 * reward
    
    # # Get the reward delta for the action and human response if they are in embedding space
    # def get_reward(self, prev_state : tuple, action : tuple, resulting_state : tuple) -> float:
    #     return 10 * self.get_safe_prob_from_embedding(torch.FloatTensor((resulting_state))) - self.get_safe_prob_from_embedding(torch.FloatTensor((prev_state)))
    
    # Use the model to embed the chat into a 8192 dim vector
    def embed(self, chat : Conversation | List[Conversation] | List[dict] | List[List[dict]]) -> torch.tensor:
        is_list_of_chat = isinstance(chat, list) and (isinstance(chat[0], list) or isinstance(chat[0], Conversation))
        if not is_list_of_chat:
            chat = [chat]
        #print("chat input to embedding model: ", chat)
        chat = [(i.create_chat() if isinstance(i, Conversation) else i) for i in chat]
        #print("after formatting: ", chat)
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)

        # Append unsafe prompt to the input_ids
        input_ids_unsafe = torch.cat((input_ids, self.unsafe_prompt.repeat(input_ids.shape[0],1)), dim=1)

        # Generate unsafe category probabilities
        with torch.no_grad():
            last_embeddings = self.model.model(input_ids=input_ids_unsafe).last_hidden_state.detach().float()
        embedding = torch.cat((last_embeddings[:,-4,:], last_embeddings[:,-1,:]), dim=-1)   # Get embedding for safe/unsafe and unsafe category
        if hasattr(self, 'projection_mat'):
            embedding = embedding @ self.projection_mat
        embedding = embedding.cpu()
        if not is_list_of_chat:
            embedding = embedding[0,:]
        return embedding

    def get_safe_prob_from_embedding(self, embedding : torch.tensor) -> float:
        safe_unsafe_logits = embedding @ self.proj_A
        safe_unsafe_probs = torch.nn.Softmax(dim=-1)(safe_unsafe_logits)
        if len(safe_unsafe_probs.shape) == 1:
            return safe_unsafe_probs[0].item()
        return safe_unsafe_probs[:,0]

    def get_unsafe_categories_probs_from_embedding(self, embedding : torch.tensor) -> float:
        unsafe_logits = embedding @ self.proj_B
        unsafe_probs = torch.nn.Softmax(dim=-1)(unsafe_logits)
        return unsafe_probs