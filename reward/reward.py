
def length_convo(convo):
    '''
    assume convo is a list of sentence strings, more than size 2
    '''
    
    cumulative_reward = 0.0  
    for idx, sentence in enumerate(convo):
        if idx % 2 == 0:
            cumulative_reward += len(sentence)
    return cumulative_reward

'''
Each reward function here receives a (convo_state, action, human_response)
input and returns the IMMEDIATE reward/cost for performing a certain action and observing the human response.
The cumulative sum should be derived by the function user.
'''

def reward_human_response_length(prev_state , action : str, human_response : str) -> float:
    print("prev state: ", prev_state)
    print("action by us (LLM): ", action)
    print("one step human response: ", human_response)
    return len(human_response)

def reward_llm_toxicity(prev_state , action : str, human_response : str) -> float:
    # TBC
    pass

    # return toxic_cost(action)

def reward_human_toxicity(prev_state, action : str, human_response : str) -> float:
    # TBC
    pass

    # return toxic_cost(human_response)
    
def rulebook_reward(prev_state, action : str, human_response : str):
    
    return [0.0, 0.0, 0.0]
    
    
    
    