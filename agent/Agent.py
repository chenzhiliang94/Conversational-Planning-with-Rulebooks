
from agent.LLM import *
import yaml

class agent:

    def __init__(self, agent : GPT_Forward, config):
        self.agent = agent
        self.config = config
    
    @abstractmethod
    def generate_prompt(self):
        pass
    
    def sample_actions(self, historical_conversation):
        input_text = self.generate_prompt() + historical_conversation
        output = self.agent.generate_text(input_text)
        print("LLM raw response: ")
        print("\n\n RAW:", output)
        output = output[0].split("\n")
        output = [o[3:] for o in output]
        return output

# human simulator
class human_agent(agent):
    to_print = True
    def __init__(self, agent : GPT_Forward, config):
        super().__init__(agent, config)
    
    def generate_prompt(self):
        prompt = "You are a 23 year old young adult. Please continue the following conversation by giving a random response. Do not add any other additional text and ignore any [YOU] or [THEM] in outputs. Keep your responses not too long. Conversation: ".format(self.config["action_sample_count"])
        return prompt
    
    def toggle_print(self, to_print):
        if to_print:
            self.to_print=to_print
        else:
            self.to_print=to_print
    
    def sample_actions(self, historical_conversation):
        output = []
        for i in range(0, self.config["action_sample_count"]):
            input_text = self.generate_prompt() + historical_conversation
            #print("response shown to human: ", input_text)
            response = self.agent.generate_text(input_text)[0].strip()
            print("response by human: ", response)
            if not response.startswith('[YOU]:'):
                response ='[YOU]: ' + response
            output.append(response)
        output = list(set(output)) # remove duplicates
        if self.to_print:
            print("by human: ", output)
        return output

# actual LLM to generate and select responses.
class llm_agent(agent):
    to_print = True
    def __init__(self, agent : GPT_Forward, config):
        super().__init__(agent, config)
    
    def generate_prompt(self):
        prompt = "You are an AI companion trying to converse with another human being. Please continue the following conversation by giving a random response. Do not add any other additional text and ignore any [YOU] or [THEM] in outputs. Keep your responses not too long. Conversation: ".format(self.config["action_sample_count"])
        return prompt
    
    def toggle_print(self, to_print):
        if to_print:
            self.to_print=to_print
        else:
            self.to_print=to_print
    
    def sample_actions(self, historical_conversation):
        output = []
        for i in range(0, self.config["action_sample_count"]):
            input_text = self.generate_prompt() + historical_conversation
            #print("response shown to human: ", input_text)
            response = self.agent.generate_text(input_text)[0].strip()
            print("response by LLM: ", response)
            if not response.startswith('[THEM]:'):
                response ='[THEM]: ' + response
            output.append(response)
        output = list(set(output)) # remove duplicates
        if self.to_print:
            print("by llm: ", output)
        return output

def create_human_and_llm():
    with open("agent/llm_config.yaml", "r") as f:
        llm_config = yaml.full_load(f)
        
    with open("agent/agent_config.yaml", "r") as f:
        agent_config = yaml.full_load(f)
        
    llm_agent_config = model_from_config(llm_config["model"])
    human_config = model_from_config(llm_config["model"])
    llm = llm_agent(llm_agent_config, agent_config)
    human = human_agent(human_config, agent_config)
    return human, llm