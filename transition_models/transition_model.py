import random
import numpy as np
import torch
from transition_models.MoE_regression import RegressionModel
'''
input and output data are all in tuple format (due to the need for dict hashing in MCTS procedure).
'''
class TransitionModel:
    def __init__(self, samples=5) -> None:
        self.samples = samples
        pass
    
    # given a state (n-dim embedding), acton (n-dim directional vector), account for human stochastic response and return a new state
    def transit(self, state, action):
        
        # convert tuple to np array
        state = np.array(list(state))
        action = np.array(list(action))
        
        intermediate_state = state + action # action is a directional vector, so we can add them directly
        
        # mimic transition for now randomly
        # convert back to tuple format
        new_states = [tuple(intermediate_state * random.gauss(0, 1)) for x in range(self.samples)]
        
        return new_states
    
    # given a state (n-dim embedding), return a LLM action (n-dimensional vector)
    def sample_actions(self, state):
        
        # convert tuple to np array
        state = np.array(list(state))
        
        # mimic action for now randomly
        dim = state.shape[0]
        return [tuple(np.random.normal(0, 1, dim)) for x in range(self.samples)]
    
class TransitionModelMOE:
    def __init__(self, samples=5, noise=0.05) -> None:
        self.samples = samples
        self.std = noise
        self.llm_model = RegressionModel() # used to generate actions
        self.human_model = RegressionModel() # used to generate transition to next state
        models_dir = "transition_models/human_llm_seed1"
        self.llm_model.load_state_dict(torch.load(f"{models_dir}/model_min_train.pth")["model_state_dict"])
        models_dir = "transition_models/llm_human_seed2"
        self.human_model.load_state_dict(torch.load(f"{models_dir}/model_min_train.pth")["model_state_dict"])
        pass
    
    # given a state (n-dim embedding), acton (n-dim directional vector), account for human stochastic response and return a new state
    def transit(self, state, action):
        
        # convert tuple to np array
        state = np.array(list(state))
        action = np.array(list(action))
        
        intermediate_state = state + action # action is a directional vector, so we can add them directly
        input = (torch.FloatTensor(intermediate_state).unsqueeze(0))
        with torch.no_grad():
            next_state = self.llm_model.forward(input)[0]
        
        perturbed_state = [tuple(list(next_state.numpy()))]
        for s in range(self.samples):
            noise = torch.randn(next_state.size()) * self.std
            perturbed_state.append(tuple(list((next_state + noise).numpy())))
        return perturbed_state

    
    # given a state (n-dim embedding), return a LLM action (n-dimensional vector)
    def sample_actions(self, state):
        
        # convert tuple to np array
        input = np.array(state)
        input = (torch.FloatTensor(input).unsqueeze(0))
        with torch.no_grad():
            next_state = self.llm_model.forward(input)[0]
        
        perturbed_state = [tuple(list(next_state.numpy()))]
        for s in range(self.samples):
            noise = torch.randn(next_state.size()) * self.std
            perturbed_state.append(tuple(((next_state + noise).numpy())))
            
        perturbed_state = [tuple(np.subtract(np.array((x)), np.array((state)))) for x in perturbed_state]
        return perturbed_state