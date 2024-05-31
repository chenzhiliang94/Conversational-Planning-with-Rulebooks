import random
import numpy as np

from transition_models.transition_model import TransitionModel
from botorch.models import SingleTaskGP, HeteroskedasticSingleTaskGP, KroneckerMultiTaskGP, MultiTaskGP, FixedNoiseMultiTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood, MarginalLogLikelihood
from botorch.optim import optimize_acqf

'''
input and output data are all in tuple format (due to the need for dict hashing in MCTS procedure).
'''
class GPTransitionModel(TransitionModel):
    def __init__(self, samples=5) -> None:
        self.samples = samples
        pass
    
    def fit_env_transitions(X, y):
        model = KroneckerMultiTaskGP(X, y)
        pass
    
    def fit_llm_transitions(X, y):
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