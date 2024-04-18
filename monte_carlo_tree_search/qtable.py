from collections import defaultdict
from monte_carlo_tree_search.qfunction import QFunction

class QTable(QFunction):
    def __init__(self, default=0.0):
        self.qtable = defaultdict(lambda: default)

    def update(self, state, action, delta, visits, reward):
        self.qtable[(state, action)] = self.qtable[(state, action)] + delta

    def get_q_value(self, state, action):
        return self.qtable[(state, action)]
    
import torch
import torch.nn as nn
from monte_carlo_tree_search.qfunction import QFunction
from torch.optim import Adam
from monte_carlo_tree_search.deep_agent import DeepAgent
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class DeepQFunction(QFunction, DeepAgent):
    """ A neural network to represent the Q-function.
        This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(
        self, alpha=0.001
    ) -> None:
        self.alpha = alpha
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.q_network = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", 
                                                           num_labels = 1)
        self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)

    def merge(self, state, action):
        # merge conversation, and LLM response together.
        return state.conversation + " " + action
    
    def update(self, state, action, delta, visits, reward):
        optimiser = Adam(self.q_network.parameters(), lr=0.0005 * (1/visits)**2)
        optimiser.zero_grad()  # Reset gradients to zero
        merged_convo = self.merge(state, action)
        if len(merged_convo) > 1000:
            merged_convo = merged_convo[-999:]
        encoded_input = self.tokenizer(merged_convo, return_tensors='pt')
        if len(encoded_input) > 512:
            encoded_input = encoded_input[:512]
        output = self.q_network(**encoded_input)
        print("output of network before update: ", output.logits)
        for x in range(30):
            optimiser.zero_grad()  # Reset gradients to zero
            output = self.q_network(**encoded_input, labels = torch.tensor(reward, dtype=torch.float))
            output.loss.backward()
            optimiser.step()  # Do a gradient descent step with the optimiser
        print("output of network after update: ", output.logits)
        
    def get_q_value(self, state, action):
        
        merged_convo = self.merge(state, action)
        if len(merged_convo) > 1000:
            merged_convo = merged_convo[-999:]
        # Convert the state into a tensor
        encoded_input = self.tokenizer(merged_convo, return_tensors='pt')
        output = self.q_network(**encoded_input)
        return output.logits[0][0]

    def get_max_q(self, state, actions):
        
        best_action = None
        best_reward = float("-inf")
        for action in actions:
            merged_convo = self.merge(state, action)
            if len(merged_convo) > 1000:
                merged_convo = merged_convo[-999:]
            encoded_input = self.tokenizer(merged_convo, return_tensors='pt')
            reward_estimate = self.q_network(**encoded_input).logits[0][0]
            if reward_estimate > best_reward:
                best_action = action
                best_reward = reward_estimate
        return (best_action, best_reward)
            
            
