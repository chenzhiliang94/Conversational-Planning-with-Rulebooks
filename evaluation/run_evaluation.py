import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from monte_carlo_tree_search.policy_agent import *
from monte_carlo_tree_search.qtable import QTable, DeepQSemanticFunction
from monte_carlo_tree_search.conversation_env import *
from agent.Agent import *
from transformers import AutoTokenizer, BertModel
from transition_models.transition_model import TransitionModel
import torch
from scipy import stats
import numpy as np
import torch
import os.path

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument("--evaluation_data", help="evaluation_data", default="evaluation_starters_simple.txt")
parser.add_argument("--evaluation_depth",  help="number of sequential actions to evaluate", default=3)
parser.add_argument("--mcts_search_depth", help="mcts search depth; only applies to mcts approaches", default=5)
parser.add_argument("--mcts_time",  help="mcts search time budget", default=100)
parser.add_argument("--pretrained_q_function",  help="pre-learnt q function for heuristic or initialization", default="model_pretrained_qfn")
parser.add_argument("--result_file",  help="result_file_name", default="evaluation_results")
parser.add_argument("--agent",  help="agent type")
args = vars(parser.parse_args())

evaluation_output = args["result_file"]
evaluation_data = args["evaluation_data"]
evaluation_action_depth = int(args["evaluation_depth"])
runtime_mcts_search_depth = int(args["mcts_search_depth"])
runtime_mcts_timeout = int(args["mcts_time"])
model = torch.load(args["pretrained_q_function"])
agent_ = args["agent"]

# get the convo starters for evaluation
with open('evaluation/' + str(evaluation_data)) as f:
    evaluation_starters = f.readlines()

# create the llm and human simulator
human, llm_agent = create_human_and_llm()

# create agents for evaluation
greedy_agent = GreedyAgent(greedy_reward_generator(human, len_reward_function), llm_agent) # infer human's next response and choose best one
pure_offline_agent = OfflineAgent(model, llm_agent) # use pretrained q functon, don't do any mcts
pure_online_mcts_agent = OnlineAgent(DeepQFunction(), runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human, reward_human_response_length, search_space="response_space") # use a brand new q function and do mcts during runtime
online_mcts_terminal_heuristic = OnlineAgent(DeepQFunction(), runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human, reward_human_response_length, model) # use a brand new q function and do mcts during runtime
pretrained_offline_online_mcts_agent = OnlineAgent(model, runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human, reward_human_response_length) # use pretrained q function and perform mcts

# semantic space agent. WIP
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model_bert = BertModel.from_pretrained("google-bert/bert-base-uncased")

def reward_function_dummy(a,b,c):
    return random.randint(0,100)

transition_model = TransitionModel()
semanticqfunction = DeepQSemanticFunction(dim=768)
pure_online_agent_semantic_agent = OnlineAgent(semanticqfunction, runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human, reward_function_dummy, search_space="semantic_space", transition_model=transition_model, tokenizer=tokenizer, embedding_model=model_bert) # online SEMANTIC space agent

agents = []
agent_type = []

if agent_ == "greedy":
    agent_type.append(agent_)
    agents.append(greedy_agent)

if agent_ == "pure_offline":
    agent_type.append(agent_)
    agents.append(pure_offline_agent)
    
if agent_ == "pure_online":
    agent_type.append(agent_)
    agents.append(pure_online_mcts_agent)

if agent_ == "offline_online_mixed":
    agent_type.append(agent_)
    agents.append(pretrained_offline_online_mcts_agent)

if agent_ == "semantic_online_agent":
    agent_type.append(agent_)
    agents.append(pure_online_agent_semantic_agent)

# create the mdp environment for evaluation
evaluation_conversation_env = conversation_environment(human, llm_agent, "", max_depth=evaluation_action_depth*2)
human.toggle_print(False)
llm_agent.toggle_print(False)

all_results = []
all_results.append(evaluation_starters)
for agent,type in zip(agents, agent_type):
    start = time.time()
    result_row = run_evaluations(agent, type, evaluation_conversation_env, evaluation_starters, evaluation_action_depth)
    all_results.append(result_row)
    print(type)
    print("time taken for 10 trials: ", time.time()-start)
    
all_results = [list(i) for i in zip(*all_results)] # transpose
import csv

with open(evaluation_output+'.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_results)