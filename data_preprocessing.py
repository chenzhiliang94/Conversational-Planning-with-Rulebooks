import numpy as np
import pandas as pd

def retrieve_openai_moderation(conversation_data : pd.DataFrame):
    # takes in dataframe and spits out [sentence, moderation_score]
    # with columns:
    # 'harassment', 'harassment/threatening',
    # 'hate', hate/threatening' 'self-harm'
    # 'self-harm/instructions': 5.6314562e-05,
    # 'self-harm/intent': 0.00021206646,
    # 'sexual': 0.006361129,
    # 'sexual/minors': 0.00035667635,
    # 'violence': 0.006479414,
    # 'violence/graphic': 8.6505745e-05
    data = []
    for idx, row in conversation_data.iterrows():
        conversation = row.conversation
        moderation = row.openai_moderation
        assert len(conversation) == len(moderation)
        
        for sentence_data, moderation_score_data in zip(conversation, moderation):
            row_data = []
            sentence = sentence_data["content"]
            category_scores = moderation_score_data["category_scores"]
            harassment = category_scores["harassment"]
            harassment_threatening = category_scores["harassment/threatening"]
            hate = category_scores["hate"]
            hate_threatening = category_scores["hate/threatening"]
            selfharm = category_scores["self-harm"]
            selfharm_instructions = category_scores["self-harm/instructions"]
            selfharm_intent = category_scores["self-harm/intent"]
            sexual = category_scores["sexual"]
            sexual_minors = category_scores["sexual/minors"]
            violence = category_scores["violence"]
            violence_graphic = category_scores["violence/graphic"]
            row_data+=[sentence,
                            harassment,
                            harassment_threatening,
                            hate,
                            hate_threatening,
                            selfharm,
                            selfharm_instructions,
                            selfharm_intent,
                            sexual,
                            sexual_minors,
                            violence,
                            violence_graphic]
            row_data = [str(x) for x in row_data]
            data.append(row_data)
    data = np.array(data)
    # ignore headers for now
    np.savetxt('moderation_data.csv', data, delimiter=",", fmt="%s")
            
            
            
        
        
        
# conversation_data['contains_h'] = np.where(
#     df['A'] == df['B'], 0, np.where(
#     df['A'] >  df['B'], 1, -1)) 