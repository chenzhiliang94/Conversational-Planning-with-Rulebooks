import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from monte_carlo_tree_search.policy_agent import *
from monte_carlo_tree_search.qtable import QTable
from monte_carlo_tree_search.conversation_env import *
from agent.Agent import *
from scipy import stats
import numpy as np
import torch
import os.path

# get the convo starters for evaluation
with open('evaluation/evaluation_starters.txt') as f:
    evaluation_starters = f.readlines()

runtime_mcts_timeout = 1000 # how long to run mcts during runtime
runtime_mcts_search_depth = 8 # mcts how deep to search
evaluation_action_depth = 5 # how many rounds of evaluation to run (i.e., number of action picked in total)

# create the llm and human simulator
human, llm_agent = create_human_and_llm()

# helper to train a q function offline
def offline_train_q_function(conversation_starters, human, llm_agent, timeout=100, search_depth=5):
    qfunction = DeepQFunction()
    for conversation_starter in conversation_starters:
        conversation_env = conversation_environment(human, llm_agent, conversation_starter, max_depth=search_depth)
        mcts = SingleAgentMCTS(conversation_env, qfunction, UpperConfidenceBounds())
        mcts.mcts(timeout=timeout)
        qfunction = mcts.qfunction
    return qfunction

# check if Q-function pretrained) exists, if not, train one offline with some conversation starters,
pretrained_q_function_name = "model_pretrained_qfn"
if not os.path.exists(pretrained_q_function_name):
    conversation_starters = [
        "School sucks today.",
        "Can you tell me something about donald trum.",
        "how is solar eclipse calculated?",
        "Hi! How are you today?",
        "Tell me something controversial about relationship between Singapore and Malaysia",
        "Singapore is friends with USA or China?",
        "I'm a bit sad. I miss my cat."
    ]

    pretraining_mcts_timeout = 500 # how long to run simulation
    pretraining_depth = 8 # how deep to run mcts

    q_function_offline_learnt = offline_train_q_function(conversation_starters, human, llm_agent, timeout=pretraining_mcts_timeout, search_depth=pretraining_depth)
    torch.save(q_function_offline_learnt, "model_pretrained_qfn")

# create agents for evaluation
greedy_agent = GreedyAgent(greedy_reward_generator(human, len_reward_function), llm_agent) # infer human's next response and choose best one
pure_offline_agent = OfflineAgent(torch.load("model_pretrained_qfn"), llm_agent) # use pretrained q functon, don't do any mcts
pure_online_mcts_agent = OnlineAgent(DeepQFunction(), runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human) # use a brand new q function and do mcts during runtime
pretrained_offline_online_mcts_agent = OnlineAgent(torch.load("model_pretrained_qfn"), runtime_mcts_search_depth, runtime_mcts_timeout, llm_agent, human) # use pretrained q function and perform mcts

agents = [greedy_agent, pure_offline_agent, pure_online_mcts_agent, pretrained_offline_online_mcts_agent]
agent_type = ["greedy", "pure offline", "pure_online", "offline_online_mixed"]

# create the mdp environment for evaluation
conversation_env = conversation_environment(human, llm_agent, "", max_depth=evaluation_action_depth*2)
human.toggle_print(False)
llm_agent.toggle_print(False)

for agent,type in zip(agents, agent_type):
    start = time.time()
    run_evaluations(agent, type,conversation_env, evaluation_starters, evaluation_action_depth)
    print(type)
    print("time taken for 10 trials: ", time.time()-start)