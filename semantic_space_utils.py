import torch
import numpy as np
import matplotlib.pyplot as plt

def get_conversation(conversation_data, idx):
    # returns numpy array of sentences
    convo = conversation_data.iloc[idx]["conversation"]
    convo = [x['content'] for x in convo]
    return np.array(convo)

def get_scores(conversation_data, idx):
    # returns score
    convo_score = conversation_data.iloc[idx]["openai_moderation"]
    return convo_score

def get_all_conversation(conversation_data):
    # returns numpy array of all data
    result = []
    for idx in range(len(conversation_data)):
        result+=list(get_conversation(conversation_data, idx))
    return np.array(result)

def get_embeddings(tokenizer, model, convo):
    # returns numpy array of embedding
    
    def get_embedding(sentence):
        with torch.no_grad():
            inputs = tokenizer(str(sentence), return_tensors="pt")
            outputs = model(**inputs)
            outputs = outputs.last_hidden_state[0]
            sentence_embedding = torch.mean(outputs, dim=0).flatten()
            return np.array(sentence_embedding)
    result = []
    for sentence in convo:
        try:
            result.append(get_embedding(sentence))
        except:
            continue
    return np.array(result)

def get_convo_embedding(pca, conversation_data, model, tokenizer, idx):
    convo = get_conversation(conversation_data, idx)
    trajectory = get_embeddings(tokenizer, model, convo)
    return  pca.transform(trajectory)

def scatter(all_embedding_reduced_dim):
    human_response = []
    LLM_response = []
    for idx in range(0,len(all_embedding_reduced_dim)):
        response_embedding = all_embedding_reduced_dim[idx]

        #plt.annotate(idx, (response_embedding[0], response_embedding[1]),fontsize=15)
        # human response
        if idx % 2 == 0:
            human_response.append(response_embedding)
        
        # LLM response
        if idx % 2 == 1:
            LLM_response.append(response_embedding)
    human_response = np.array(human_response)
    LLM_response = np.array(LLM_response)
    plt.scatter(human_response[:,0],human_response[:,1],s=3, marker="s",c="black",label="human response")
    plt.scatter(LLM_response[:,0],LLM_response[:,1],s=3, marker="^",c="lime", label="LLM response")
    
def visualize_convo(idx, all_embedding_reduced_dim, conversation_data, model_bert, tokenizer, pca):    
    plt.figure(figsize=(13,8))
    # all embedding
    plt.scatter(all_embedding_reduced_dim[:,0], all_embedding_reduced_dim[:,1],s=1)

    plt.plot(get_convo_embedding(pca, conversation_data, model_bert, tokenizer, idx)[:, 0], get_convo_embedding(pca, conversation_data, model_bert, tokenizer, idx)[:,1], c="r", linestyle='dashed',linewidth=0.5, alpha=0.6)
    scatter(get_convo_embedding(pca, conversation_data, model_bert, tokenizer, idx))


    plt.legend()
    plt.title("embedding space of responses")
    plt.show()

    convo = get_conversation(conversation_data, idx)
    score = get_scores(conversation_data, idx)
    print(convo)