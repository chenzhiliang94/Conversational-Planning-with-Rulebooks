
from transition_models.transition_data import DailyDialogueTransitionData
from transition_models.deterministic_transition_model import DeterministicTransitionModel
from torch.utils.data import DataLoader, Dataset

import pickle
import matplotlib.pyplot as plt
import torch

with open('daily_dialogue_transition_dataset_train.pkl', 'rb') as f:    
    train_dataset = pickle.load(f)
print("finished loading training dataset")    
with open('daily_dialogue_transition_dataset_validation.pkl', 'rb') as f:    
    val_dataset = pickle.load(f)
print("finished loading val dataset") 

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

dim = 4096
device = torch.device('cuda:5')
transition = DeterministicTransitionModel(dim, device)
train_loss,validation_loss = transition.train_llm_transition(train_loader, val_loader, lr=0.001, num_epochs=300)


