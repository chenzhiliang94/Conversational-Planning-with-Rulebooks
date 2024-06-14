
import yaml

from typing import List, Tuple
from tqdm import tqdm
from agent.Local_LLM import Local_LLM
from agent.Online_LLM import Online_LLM
from agent.Conversation import Conversation, HUMAN, LLM, get_role

DEBUG = False

class Agent:
    """Abstract base class for large language models."""

    def __init__(self, role, config, needs_confirmation=False, disable_tqdm=True, 
                 model=None, tokenizer=None):
        """Initializes the model."""
        self.role = role
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm

        # Initialise human model
        if self.config["type"] == "local":
            LLM_class = Local_LLM
        else:
            LLM_class = Online_LLM
        self.model = LLM_class(
            self.config,
            model = model,
            tokenizer = tokenizer,
            )
        tqdm.write(f"Initialized {get_role(role)} as {self.config["type"]} model: {self.config["name"]}.")

    def sample_actions(self, prompt : str) -> List[str]:
        # convo = Conversation.from_delimited_string(prompt)
        convo = prompt
        return self.generate_text(convo)

    def generate_text(self, convos : Conversation | List[Conversation], batch = False) -> List[str] | List[List[str]]:
        """Generates text from the model.
        Parameters:
            convos: The prompt to use. List of Conversation.
        Returns:
            A list of list of strings.
        """

        convos_is_list = isinstance(convos, list)
        if not convos_is_list:
            convos = [convos]

        chats : List[List[dict]] = []
        # Create prompts from converstation histories
        for convo in convos:
            chat = self.model.apply_chat_format(convo)
            chats.append(chat)

        if DEBUG:
            print("generated prompts")
            print(chats)

        generated_text = []

        if batch:
            raise NotImplementedError
        else:
            if not self.disable_tqdm:
                chats = tqdm(chats)
            for chat in chats:
                output = self.model.generate(chat)
                generated_text.append(output)
        if not convos_is_list:
            generated_text = generated_text[0]
        return generated_text

def create_human_and_llm(**kwargs) -> Tuple[Agent, Agent]:
    with open("agent/llm_config.yaml", "r") as f:
        llm_config = yaml.full_load(f)

    human_agent = Agent(HUMAN, llm_config["human_model"], **kwargs)
    llm_agent = Agent(LLM, llm_config["llm_model"], **kwargs)
    return human_agent, llm_agent