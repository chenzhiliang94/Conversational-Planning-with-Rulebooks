import random
import numpy as np
import torch
from transition_models.regression_wrapper import RegressionWrapper
from mixture_of_experts import HeirarchicalMoE
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
    def __init__(self, samples=5, noise=0.05, cuda=torch.device("cpu")) -> None:
        self.samples = samples
        self.std = noise
        self.llm_models = [] # used to generate actions
        self.human_models = [] # used to generate transition to next state
        main_dir = "models/deterministic/"
        self.cuda = cuda
        for i in range(1):
            models_dir = f"{main_dir}/seed_{i}_batch_2048/human_llm"
            self.llm_models.append(RegressionWrapper(HeirarchicalMoE(1024)))
            self.llm_models[i].load_state_dict(torch.load(f"{models_dir}/model_min_train.pth", map_location=torch.device("cpu"))["model_state_dict"])
            self.llm_models[i].to(cuda)
            models_dir = f"{main_dir}/seed_{i}_batch_2048/llm_human"
            self.human_models.append(RegressionWrapper(HeirarchicalMoE(1024)))
            self.human_models[i].load_state_dict(torch.load(f"{models_dir}/model_min_train.pth", map_location=torch.device("cpu"))["model_state_dict"])
            self.human_models[i].to(cuda)
    
    def forward(self, input, models):
        input = input.to(self.cuda)
        next_state = []
        for model in models:
            with torch.no_grad():
                next_state.append(model.forward(input)[0].cpu())

        if len(next_state) == 1:
            next_state = next_state[0]
            perturbed_state = [tuple(next_state.numpy())]
            for s in range(self.samples):
                noise = torch.randn(next_state.size()) * self.std
                perturbed_state.append(tuple((next_state + noise).numpy()))
        else:
            perturbed_state = [tuple(next_state[i].numpy()) for i in range(len(next_state))]

        return perturbed_state

    # given a state (n-dim embedding), acton (n-dim directional vector), account for human stochastic response and return a new state
    def transit(self, state, action):
        
        # convert tuple to np array
        state = np.array(list(state))
        action = np.array(list(action))
        
        intermediate_state = state + action # action is a directional vector, so we can add them directly
        input = (torch.FloatTensor(intermediate_state).unsqueeze(0))

        perturbed_state = self.forward(input, self.llm_models)

        return perturbed_state

    
    # given a state (n-dim embedding), return a LLM action (n-dimensional vector)
    def sample_actions(self, state):
        
        # convert tuple to np array
        input = np.array(state)
        input = (torch.FloatTensor(input).unsqueeze(0))

        perturbed_state = self.forward(input, self.llm_models)

        return perturbed_state