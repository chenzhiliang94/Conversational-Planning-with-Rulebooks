from collections import defaultdict
from monte_carlo_tree_search.qfunction import QFunction
import time

def combine_encoded_inputs(input1, input2):
    new_encoding = {}
    for k in input1.keys():
        # padding first 
        i1_size = input1[k].shape[1]
        i2_size = input2[k].shape[1]
        i1 = input1[k]
        i2 = input2[k]
        if i2_size > i1_size:
            i1 = nn.functional.pad(input1[k], (0, i2_size-i1_size), 'constant', 0)
        elif i2_size < i1_size:
            i2 = nn.functional.pad(input2[k], (0, i1_size-i2_size), 'constant', 0)
        new_encoding[k] = torch.cat((i1,i2), 0)
    return new_encoding


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
        self, alpha=0.001, steps_update=100, cuda = torch.device('cuda:2')
    ) -> None:
        self.alpha = alpha
        self.steps_update = steps_update
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.q_network = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", 
                                                           num_labels = 1).to(cuda)
        self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)
        self.cuda = cuda

    def merge(self, state, action):
        # merge conversation, and LLM response together.
        return state.conversation + action
    
    def update(self, state, action, delta, visits, reward):
        optimiser = Adam(self.q_network.parameters(), lr=0.01 * (1/visits)**2)
        optimiser.zero_grad()  # Reset gradients to zero
        merged_convo = self.merge(state, action)
        merged_convo = str(merged_convo)
        if len(merged_convo) > 1000:
            merged_convo = merged_convo[-999:]
        encoded_input = self.tokenizer(merged_convo, return_tensors='pt')
        if len(encoded_input) > 512:
            encoded_input = encoded_input[:512]
        # if len(merged_convo) > 1000:
        #     merged_convo = merged_convo[-999:]
        encoded_input = self.tokenizer(merged_convo, truncation=True, max_length=512,  return_tensors='pt').to(self.cuda)

        #print("output of network before update: ", output.logits)
        for x in range(self.steps_update):
            optimiser.zero_grad()  # Reset gradients to zero
            output = self.q_network(**encoded_input, labels = torch.tensor(reward, dtype=torch.float).to(self.cuda))
            output.loss.backward()
            optimiser.step()  # Do a gradient descent step with the optimiser
        #print("output of network after update: ", output.logits)
        
    def get_q_value(self, state, action):
        
        merged_convo = self.merge(state, action)
        merged_convo = str(merged_convo)
        if len(merged_convo) > 1000:
            merged_convo = merged_convo[-999:]
        # if len(merged_convo) > 1000:
        #     merged_convo = merged_convo[-999:]
        # Convert the state into a tensor
        encoded_input = self.tokenizer(merged_convo, truncation=True, max_length=512,  return_tensors='pt').to(self.cuda)
        output = self.q_network(**encoded_input)
        return output.logits[0][0]

    def get_max_q(self, state, actions):
        
        best_action = None
        best_reward = float("-inf")
        for action in actions:
            merged_convo = self.merge(state, action)
            # if len(merged_convo) > 1000:
            #     merged_convo = merged_convo[-999:]
            encoded_input = self.tokenizer(merged_convo, truncation=True, max_length=512, return_tensors='pt').to(self.cuda)
            reward_estimate = self.q_network(**encoded_input).logits[0][0]
            if reward_estimate > best_reward:
                best_action = action
                best_reward = reward_estimate
        return (best_action, best_reward)
            
# class DeepQSemanticFunction(QFunction, DeepAgent):
#     """ A neural network to represent the Q-function for semantic space
#         This class uses PyTorch for the neural network framework (https://pytorch.org/).
#     """

#     def __init__(
#         self, dim, alpha=0.001
#     ) -> None:
#         self.alpha = alpha
#         self.dim = dim
#         self.q_network = nn.Sequential(
#             nn.Linear(dim * 2, 128),
#             nn.ReLU(),
#             nn.Linear(128, 24),
#             nn.ReLU(),
#             nn.Linear(24, 12),
#             nn.ReLU(),
#             nn.Linear(12, 1)
#         )
#         self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)

#     def merge(self, state, action):
#         # merge conversation, and LLM response together.
#         merged_convo = list(state.conversation) + list(action)
#         return torch.Tensor([merged_convo])
    
#     def update(self, state, action, delta, visits, reward):
#         self.optimiser.lr=0.0005 * (1/visits)**2
#         merged_convo = self.merge(state, action)
#         for x in range(30):
#             self.optimiser.zero_grad()  # Reset gradients to zero
#             loss_fn = nn.MSELoss()
#             y_pred = self.q_network(merged_convo)
#             loss = loss_fn(y_pred, torch.tensor([reward],requires_grad=True))
#             loss.backward()
#             self.optimiser.step()
        
#     def get_q_value(self, state, action):
#         merged_convo = self.merge(state, action)
#         output = self.q_network(merged_convo)
#         return output[0][0]

#     def get_max_q(self, state, actions):
            
#         best_action = None
#         best_reward = float("-inf")
#         for action in actions:
#             merged_convo = self.merge(state, action)
#             reward_estimate = self.q_network(merged_convo)[0][0]
#             if reward_estimate > best_reward:
#                 best_action = action
#                 best_reward = reward_estimate
#         return (best_action, best_reward)
    
    
class DeepQSemanticFunction(QFunction, DeepAgent):
    """ A neural network to represent the Q-function for semantic space
        This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(
        self, dim, cuda, steps_update, alpha=0.001
    ) -> None:
        self.alpha = alpha
        self.dim = dim
        self.update_steps = steps_update
        self.q_network = nn.Sequential(
            nn.Linear(dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1)
        )
        self.cuda = cuda
        self.replay_buffer = None
        self.past_rewards = []
        self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)

    def merge(self, state, action):
        # merge conversation, and LLM response together.
        merged_convo = list(state.conversation) + list(action)
        return torch.Tensor([merged_convo])
    
    def update_buffer(self, input, reward):
        self.past_rewards.append(reward)
        if self.replay_buffer is None:
            self.replay_buffer = input
        else:
            self.replay_buffer = torch.cat((self.replay_buffer, input), 0)
            
    def update(self, state, action, delta, visits, reward):
        self.optimiser.lr=self.alpha * (1/visits)**2
        merged_convo = self.merge(state, action)
        for x in range(self.update_steps):
            self.optimiser.zero_grad()  # Reset gradients to zero
            loss_fn = nn.MSELoss()
            y_pred = self.q_network(merged_convo)
            loss = loss_fn(y_pred, torch.tensor(reward,requires_grad=True).view(1,1))
            loss.backward()
            self.optimiser.step()
        self.update_buffer(merged_convo, reward)
        
        for x in range(self.update_steps):
            self.optimiser.zero_grad()  # Reset gradients to zero
            loss_fn = nn.MSELoss()
            y_pred = self.q_network(self.replay_buffer)
            loss = loss_fn(y_pred, torch.tensor(self.past_rewards,requires_grad=True).view(y_pred.shape[0],1))
            loss.backward()
            self.optimiser.step()
        
    def get_q_value(self, state, action):
        merged_convo = self.merge(state, action)
        output = self.q_network(merged_convo)
        return output[0][0]

    def get_max_q(self, state, actions):
            
        best_action = None
        best_reward = float("-inf")
        for action in actions:
            merged_convo = self.merge(state, action)
            reward_estimate = self.q_network(merged_convo)[0][0]
            if reward_estimate > best_reward:
                best_action = action
                best_reward = reward_estimate
        return (best_action, best_reward)
    
    def reset(self):
        self.q_network = nn.Sequential(
            nn.Linear(self.dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1)
        )
        self.replay_buffer = None
        self.past_rewards = []
    
    
class ReplayBufferDeepQFunction(QFunction, DeepAgent):
    """ A neural network to represent the Q-function.
        This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(
        self, alpha=0.1, steps_update=100, cuda = torch.device('cuda:2')
    ) -> None:
        self.alpha = alpha
        self.steps_update = steps_update
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.q_network = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", 
                                                           num_labels = 1).to(cuda)
        self.optimiser = Adam(self.q_network.parameters(), lr=self.alpha)
        self.cuda = cuda
        self.replay_buffer = None
        self.past_rewards = []

    def reset(self):
        self.q_network = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", 
                                                           num_labels = 1).to(self.cuda)
        self.replay_buffer = None
        self.past_rewards = []
    
    def merge(self, state, action):
        # merge conversation, and LLM response together.
        return state.conversation + action
    
    def update_buffer(self, input, reward):
        self.past_rewards.append(reward)
        if self.replay_buffer is None:
            self.replay_buffer = input
        else:
            self.replay_buffer = combine_encoded_inputs(self.replay_buffer, input)

    def update(self, state, action, delta, visits, reward):
        optimiser = Adam(self.q_network.parameters(), lr=self.alpha * (1/visits)**2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.1, patience=100, threshold=200)
        criterion = torch.nn.MSELoss()
        merged_convo = self.merge(state, action)
        merged_convo = str(merged_convo)
        
        # update replay buffer
        encoded_input = self.tokenizer(merged_convo, truncation=True, max_length=512,  padding=True, return_tensors='pt').to(self.cuda)
        self.update_buffer(encoded_input, reward)
        
        self.q_network.train()
        # update based on this specific experience
        start_time = time.time()
        optimiser = Adam(self.q_network.parameters(), lr= self.alpha * (1/visits)**2)
        for x in range(self.steps_update):
            optimiser.zero_grad()  # Reset gradients to zero
            output = self.q_network(**encoded_input, labels = torch.tensor(reward, dtype=torch.float).to(self.cuda))
            if output.loss == torch.tensor(float('nan')): # if loss becomes nan, reduce LR
                optimiser = Adam(self.q_network.parameters(), lr= 0.1 * self.alpha * (1/visits)**2)
                continue
            output.loss.backward()
            optimiser.step()  # Do a gradient descent step with the optimiser
            print("loss in standard update: ", output.loss)
        print("time taken for update Q", time.time()-start_time)
        start_time = time.time()
        
        # update based on replay buffer
        optimiser = Adam(self.q_network.parameters(), lr= 0.3* self.alpha * (1/visits)**2)
        for x in range(self.steps_update):
            optimiser.zero_grad()  # Reset gradients to zero
            output = self.q_network(**self.replay_buffer, labels = torch.tensor(self.past_rewards, dtype=torch.float).view(len(self.past_rewards), 1).to(self.cuda))
            if output.loss.detach() == torch.tensor(float('nan')): # if loss becomes nan, reduce LR
                optimiser = Adam(self.q_network.parameters(), lr= 0.5 * self.alpha * (1/visits)**2)
                continue
            output.loss.backward()
            print("loss in replay buffer: ", output.loss)
            optimiser.step()  # Do a gradient descent step with the optimiser
        print("time taken for update Q with replay buffer: ", time.time()-start_time)
    
    def update_with_replay_buffer(self):
        optimiser = Adam(self.q_network.parameters(), lr=self.alpha)
        
        # update based on replay buffer
        for x in range(self.steps_update):
            optimiser.zero_grad()  # Reset gradients to zero
            output = self.q_network(**self.replay_buffer, labels = torch.tensor(self.past_rewards, dtype=torch.float).to(self.cuda))
            output.loss.backward()
            print(output.loss)
            optimiser.step()  # Do a gradient descent step with the optimiser
        
    def get_q_value(self, state, action):
        print("getting q value of merged convo:")
        
        merged_convo = self.merge(state, action)
        merged_convo = str(merged_convo)
        print(merged_convo)
        encoded_input = self.tokenizer(merged_convo, truncation=True, max_length=512,  return_tensors='pt').to(self.cuda)
        #print(encoded_input)
        with torch.no_grad():
            output = self.q_network(**encoded_input)
            q_value = output.logits[0][0]
            print("Q value is: ")
            print('{0:.8f}'.format(q_value))
            return q_value

    def get_max_q(self, state, actions):
        
        best_action = None
        best_reward = float("-inf")
        for action in actions:
            reward_estimate = self.get_q_value(state, action)
            if reward_estimate > best_reward:
                best_action = action
                best_reward = reward_estimate
        return (best_action, best_reward)