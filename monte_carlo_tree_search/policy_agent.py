from monte_carlo_tree_search.gridworld import *
from graph_visualisation import GraphVisualisation
from monte_carlo_tree_search.qtable import QTable, DeepQFunction
from monte_carlo_tree_search.single_agent_mcts import SingleAgentMCTS
from monte_carlo_tree_search.conversation_env import conversation_environment, conversation_state
from monte_carlo_tree_search.semantic_conversation_env import semantic_conversation_environment, conversation_semantic_state
from monte_carlo_tree_search.ucb import UpperConfidenceBounds
import numpy as np
from scipy import stats
import torch

from abc import abstractmethod

class LearntAgent():
    def __init__(self) -> None:
        pass
    @abstractmethod
    def generate_action(self, state):
        pass

# an agent that just greedily returns the best action during runtime. Infer next response by human and choose greedily.
class GreedyAgent(LearntAgent):
    
    def __init__(self, reward_calculator, action_generator) -> None:
        self.reward_calculator = reward_calculator
        self.action_generator = action_generator

    def generate_action(self, state):
        possible_actions = self.action_generator.sample_actions(state.conversation) # maybe add an argument to choose number of actions
        return self.reward_calculator.select(state, possible_actions)

# greedy reward functions to be used in GreedyAgent
def len_reward_function(human_response):
    return len(human_response)

class greedy_reward_generator():
    def __init__(self, human_agent, reward_function) -> None:
        self.human = human_agent
        self.reward_function = reward_function
    
    # greedy reward: infer multiple human responses. take average reward from them.
    def select(self, state, possible_actions):
        convo = state.conversation
        action_reward = []
        for action in possible_actions:
            human_responses = self.human.sample_actions(convo + " " + action)
            
            reward_to_be_averaged = []
            for response in human_responses:
                reward_to_be_averaged.append(self.reward_function(response))
            action_reward.append(np.mean(reward_to_be_averaged))
        best_action_idx = action_reward.index(min(action_reward))
        return possible_actions[best_action_idx]
            
# An agent with a pretrained Q function used to find best action during runtime. No searching is done.
class OfflineAgent(LearntAgent):
    
    def __init__(self, qfunction : DeepQFunction, llm_agent) -> None:
        self.qfunction = qfunction
        self.llm_agent = llm_agent

    def generate_action(self, state):
        possible_actions = self.llm_agent.sample_actions(state.conversation) # maybe add an argument to choose number of actions
        best_action, best_reward = self.qfunction.get_max_q(state, possible_actions)
        return best_action

# An agent which performs MCTS during runtime. Takes in a Q functon during initialization (possibly pretrained)
class OnlineAgent(LearntAgent):
    
    def __init__(self, qfunction : DeepQFunction, search_depth, mcts_time_limit, llm_agent, human_simulator, search_space="response_space", terminating_heuristic_q_function=None, transition_model=None, tokenizer=None, embedding_model=None) -> None:
        self.search_depth = search_depth
        self.mcts_time_limit = mcts_time_limit
        self.llm_agent = llm_agent
        self.human_simulator = human_simulator
        self.qfunction = qfunction
        self.terminating_heuristic_q_function = terminating_heuristic_q_function
        self.search_space = search_space
        self.transition_model = transition_model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
    
    def generate_action(self, state):
        
        # perform mcts
        if self.search_space=="response_space":
            conversation_env = conversation_environment(self.human_simulator, self.llm_agent, state.conversation, max_depth=self.search_depth)
        elif self.search_space=="semantic_space":
            conversation_env = semantic_conversation_environment(tokenizer=self.tokenizer, model=self.embedding_model, transition_model=self.transition_model, initial_state=state.conversation, max_depth=self.search_depth)
        mcts = SingleAgentMCTS(conversation_env, self.qfunction, UpperConfidenceBounds(), terminating_heuristic_q_function=self.terminating_heuristic_q_function)
        mcts.mcts(timeout=self.mcts_time_limit)
        self.qfunction = mcts.qfunction # qfunction learnt after performing mcts
        
        # get best action from learnt q function after mcts
        possible_actions = self.llm_agent.sample_actions(state.conversation)
        if self.search_space=="response_space":
            best_action, best_reward = self.qfunction.get_max_q(state, possible_actions)
            
        # if semantic space used, some semantic projection is needed
        elif self.search_space=="semantic_space":
            encoded_input = self.tokenizer(state.conversation, return_tensors='pt')
            output = self.model(**encoded_input).last_hidden_state
            conversation_semantics = tuple(torch.mean(output[0],0).detach().numpy())
            
            action_semantics = []
            for action in possible_actions:
                encoded_input = self.tokenizer(state.conversation + " " + action, return_tensors='pt')
                output = self.model(**encoded_input).last_hidden_state
                action_semantics.append(tuple(torch.mean(output[0],0).detach().numpy()))
            state.conversation = conversation_semantics
            best_action, best_reward = self.qfunction.get_max_q(state, possible_actions)
            best_idx = possible_actions.index(best_action)
            best_action = possible_actions[best_idx]
            
        return best_action
    
    # util function for resetting q function
    def reset(self):
        self.qfunction = DeepQFunction()
    
def evaluate_agent(agent : LearntAgent, env, starting_state, number_replies):
    
    cumulative_reward = 0.0
    for r in range(number_replies):
        
        # get best action based on starting_state
        action = agent.generate_action(starting_state)
        
        # go to next state
        next_state, reward = env.execute_in_simulation(starting_state, action)
        
        starting_state = next_state
        cumulative_reward += reward
    return cumulative_reward

# evaluate an agent with the mdp.
def run_evaluations(agent, type, env, evaluation_starters, number_replies):
    result_row = []
    for evaluation_starter in evaluation_starters:
        initial_state = conversation_state(evaluation_starter, evaluation_starter)
        initial_state.depth = 1
        
        # repeated trials
        rewards = []
        for x in range(10):
            print("trial: ", x, " of evaluation for agent of type:  ", type)
            cumulative_reward = evaluate_agent(agent, env, initial_state, number_replies)
            print("cumulative reward for this trial: ", cumulative_reward)
            rewards.append(cumulative_reward)
        print(evaluation_starter)
        print("all rewards from trials: ", rewards)
        print("mean: ", np.mean(rewards))
        print("std error: ", stats.sem(rewards))
        result_row.append((int(np.mean(rewards)), int(stats.sem(rewards))))
    return result_row
