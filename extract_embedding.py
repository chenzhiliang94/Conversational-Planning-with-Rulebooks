import random
import torch
import numpy as np
import pandas as pd

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from transition_models.embedding_model import *

def extract_semantic_embedding(conversation_data, output_name, embedding_model):
    data = []
    for idx_convo, convo in enumerate(conversation_data.dialog):
        print("convo idx: ", idx_convo)
        if len(convo) < 2:
            continue
        s1 = ""
        for idx in range(len(convo)-1):
            row_data = []
            s1 = s1 + " " + convo[idx]
            s2 = s1 + " " + convo[idx+1]
            
            s1_embedding = embedding_model.embed(s1).detach().cpu().numpy()
            s2_embedding = embedding_model.embed(s2).detach().cpu().numpy()
            
            transition_embedding = np.concatenate([s1_embedding, s2_embedding])
            
            row_data.append(s1)
            row_data.append(s2)
            row_data += list(transition_embedding)
            data.append(row_data)

    import csv
    with open(output_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
        
def extract_semantic_embedding_LMSYS_dialogue(conversation_data, output_name, embedding_model):
    llm_transition_data = [] # response of LLM
    env_transition_data = [] # response of human
    moderation_score_data = []
    count = 0
    for convo, moderation_score in zip(conversation_data.conversation, conversation_data.openai_moderation):
        if len(convo) < 2:
            continue
        s1 = ""
        for idx in range(len(convo)-1):
            row_data = []
            row_data_llm = []
            row_data_env = []
            row_data_moderation = []

            s1 = s1 + " " + convo[idx]["content"]
            s2 = s1 + " " + convo[idx+1]["content"]
            
            s1_embedding = embedding_model.embed(s1).detach().cpu().numpy()
            s2_embedding = embedding_model.embed(s2).detach().cpu().numpy()
            
            transition_embedding = np.concatenate([s1_embedding, s2_embedding])
            
            row_data.append(s1)
            row_data.append(s2) # sentence
            
            if idx % 2 == 0: # if human is s1, llm is s2
                row_data_llm = row_data + list(transition_embedding)
            if idx % 2 == 1: # llm is s1, human is s2
                row_data_env = row_data + list(transition_embedding)
            
                
            harmful_score = moderation_score[idx]['category_scores']
            print(harmful_score)
            row_data_moderation = row_data + [harmful_score["harassment"]] + [harmful_score["hate"]] + [harmful_score["sexual"]] + [harmful_score["violence"]] + list(transition_embedding)
                
            llm_transition_data.append(row_data_llm)
            env_transition_data.append(row_data_env)
            moderation_score_data.append(row_data_moderation)
        count += 1

    import csv
    with open(output_name + "_LLM_TRANSITIONS.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(llm_transition_data)

    with open(output_name + "_HUMAN_TRANSITIONS.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(env_transition_data)

    with open(output_name + "_MODERATION_SCORES.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(moderation_score_data)

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument("--convo_data", help="convo_data")
parser.add_argument("--embedding_model",  help="embedding_model")
parser.add_argument("--device",  help="to_normalize")
parser.add_argument("--normalize",  help="to_normalize")
args = vars(parser.parse_args())

convo_data = args["convo_data"]
embedding_model_type = args["embedding_model"]
to_normalize = bool(int(args["normalize"]))
device_num = int(args["device"])

conversation_data = None
# read from dataset
if convo_data == "daily_dialogue":
    conversation_data = pd.read_parquet('daily_dialogue.parquet', engine='auto')
if convo_data == "LMSYS":
    conversation_data = pd.read_parquet('human_llm_convo.parquet', engine='auto')
    conversation_data = conversation_data.loc[conversation_data['language'] == "English"]
    
cuda4 = torch.device('cuda:'+str(device_num))

from transition_models.embedding_model import *

if embedding_model_type == "mistral":
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral').to(cuda4)
    print("finished loading from pretrained")
    embed_model = embedding_model_mistral(tokenizer, model, to_normalize, cuda4)

if embedding_model_type == "nomic":
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True).to(cuda4)
    print("finished loading from pretrained")
    embed_model = embedding_model_nomic(tokenizer, model, to_normalize, cuda4) 

output = convo_data + "_" + embedding_model_type + ".csv"

if convo_data == "daily_dialogue":
    extract_semantic_embedding(conversation_data, output, embed_model)
if convo_data == "LMSYS":
    extract_semantic_embedding_LMSYS_dialogue(conversation_data, output, embed_model)