"""Contains classes for querying large language models."""

from abc import ABC, abstractmethod
from agent.Conversation import Conversation
from typing import List

class LLM(ABC):
    system_prompt = ""

    @abstractmethod
    def generate(self, chat : List[dict]) -> List[str]:
        pass

    # Create chat and add add system prompt
    def apply_chat_format(self, convo : Conversation) -> List[dict]:
        chat = convo.create_chat()

        if len(self.system_prompt) > 0:
            if chat[0]["role"] == "assistant":
                chat[0]["content"] = self.system_prompt + "\n\n" + chat[0]["content"]
                chat = chat[1:]
            elif self.tokenizer_has_system_prompt:
                    chat.insert(0,{"role": "system", "content":self.system_prompt})
            # else:
            #     chat.insert(0,{"role": "assistant", "content":self.system_prompt})

        return chat