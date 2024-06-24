import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from monte_carlo_tree_search.policy_agent import *
from monte_carlo_tree_search.qtable import QTable, DeepQSemanticFunction, ReplayBufferDeepQFunction
from monte_carlo_tree_search.conversation_env import *
from agent.Agent import *
from transformers import AutoTokenizer, BertModel, AutoModel
from transition_models.transition_model import TransitionModel
from transition_models.embedding_model import embedding_model_mistral, embedding_model_nomic
from reward.Embedding_Dummy_Reward import Embedding_Dummy_Reward
from reward.Human_Length_Reward import Human_Length_Reward

import torch
from scipy import stats
import numpy as np
import torch
import os.path
import time

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
parser.add_argument("--embedding", default="mistral")
parser.add_argument("--cuda",  help="cuda")
parser.add_argument("--reward_decay",  default=0.9)
args = vars(parser.parse_args())

evaluation_output = args["result_file"]
evaluation_data = args["evaluation_data"]
evaluation_action_depth = int(args["evaluation_depth"])
runtime_mcts_search_depth = int(args["mcts_search_depth"])
runtime_mcts_timeout = int(args["mcts_time"])
model = torch.load(args["pretrained_q_function"])
agent_ = args["agent"]
embedding_type = args["embedding"]
cuda_ = int(args["cuda"])
reward_decay = float(args["reward_decay"])

# get the convo starters for evaluation
with open('evaluation/' + str(evaluation_data)) as f:
    evaluation_starters = f.readlines()

# create the llm and human simulator
human, llm_agent = create_human_and_llm()

# create agents for evaluation
random_agent = RandomAgent(llm_agent)
greedy_agent = GreedyAgent(greedy_reward_generator(human, len_reward_function), llm_agent) # infer human's next response and choose best one
pure_offline_agent = OfflineAgent(model, llm_agent) # use pretrained q functon, don't do any mcts
reward_function = Human_Length_Reward()
online_mcts_terminal_heuristic = OnlineAgent(DeepQFunction(), runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human, reward_function, model) # use a brand new q function and do mcts during runtime
pretrained_offline_online_mcts_agent = OnlineAgent(model, runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human, reward_function) # use pretrained q function and perform mcts

agents = []
agent_type = []

if agent_ == "greedy":
    agent_type.append(agent_)
    agents.append(greedy_agent)
    
if agent_ == "random":
    agent_type.append(agent_)
    agents.append(random_agent)

if agent_ == "pure_offline":
    agent_type.append(agent_)
    agents.append(pure_offline_agent)
    
if agent_ == "pure_online":
    pure_online_mcts_agent = OnlineAgent(ReplayBufferDeepQFunction(alpha=0.1, steps_update=100, cuda=torch.device('cuda:'+str(cuda_))), runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human, reward_function, search_space="response_space", reward_decay=reward_decay) # use a brand new q function and do mcts during runtime
    agent_type.append(agent_)
    agents.append(pure_online_mcts_agent)

if agent_ == "offline_online_mixed":
    agent_type.append(agent_)
    agents.append(pretrained_offline_online_mcts_agent)

if agent_ == "semantic_online":
    
    embed_model=None
    dim = None
    if embedding_type == "mistral":
        dim=4096
        # load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
        model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral').to(torch.device('cuda:'+str(cuda_)))
        print("finished loading from pretrained")
        embed_model = embedding_model_mistral(tokenizer, model, False, torch.device('cuda:'+str(cuda_)))

    if embedding_type == "nomic":
        dim=768
        # load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True).to(torch.device('cuda:'+str(cuda_)))
        print("finished loading from pretrained")
        embed_model = embedding_model_nomic(tokenizer, model, False, torch.device('cuda:'+str(cuda_))) 

    reward_function = Embedding_Dummy_Reward()
    transition_model = TransitionModel()
    semanticqfunction = DeepQSemanticFunction(dim=dim, cuda=torch.device('cuda:'+str(cuda_)), steps_update=50)
    pure_online_agent_semantic_agent = OnlineAgent(semanticqfunction, runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human, reward_function, search_space="semantic_space", transition_model=transition_model, embedding_model=embed_model) # online SEMANTIC space agent

    agent_type.append(agent_)
    agents.append(pure_online_agent_semantic_agent)

# create the mdp environment for evaluation
evaluation_conversation_env = conversation_environment(human, llm_agent, "", max_depth=evaluation_action_depth*2)

all_results = []
all_results.append(evaluation_starters)
for agent,type in zip(agents, agent_type):
    start = time.time()
    result_row = run_evaluations(agent, type, evaluation_conversation_env, evaluation_starters, evaluation_action_depth)
    all_results.append(result_row)
    print(type)
    print("time taken for 5 trials: ", time.time()-start)
    
all_results = [list(i) for i in zip(*all_results)] # transpose
import csv

with open(evaluation_output+'.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(all_results)