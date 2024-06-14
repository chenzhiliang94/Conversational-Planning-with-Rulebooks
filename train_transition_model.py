
from transition_models.transition_data import DailyDialogueTransitionData
from transition_models.deterministic_transition_model import DeterministicTransitionModel
from torch.utils.data import DataLoader, Dataset

import pickle
import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def train(train_data_path, val_data_path, dim, cuda, save_path):
    with open(train_data_path, 'rb') as f:    
        train_dataset = pickle.load(f)
        print(train_dataset[0][0].shape)
    print("finished loading training dataset")    
    with open(val_data_path, 'rb') as f:    
        val_dataset = pickle.load(f)
    print("finished loading val dataset") 

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    #device = torch.device('cuda:5')
    device = cuda
    transition = DeterministicTransitionModel(dim, cuda=device)
    transition.train_llm_transition(train_loader, val_loader, path_to_save=save_path, lr=0.001, num_epochs=300)

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument("--train_data", help="train_data")
parser.add_argument("--val_data",  help="val_data")
parser.add_argument("--device",  help="device")
parser.add_argument("--embed",  help="embed")
parser.add_argument("--model_name_to_save",  help="model_name")
args = vars(parser.parse_args())

train_path = str(args["train_data"])
val_data = str(args["val_data"])
model_save = str(args["model_name_to_save"])
device_num = int(args["device"])
embed = str(args["embed"])

dim = -1
if embed == "nomic":
    dim = 768
if embed == "mistral":
    dim = 4096

# example usage:
# python3 train_transition_model.py --train_data=data/mistral/daily_dialogue_transition_dataset_train_MISTRAL.pkl --val_data=data/mistral/daily_dialogue_transition_dataset_validation_MISTRAL.pkl
# --device=0 --embed=mistral --model_name_to_save=mistral_daily_dialogue
train(train_path, val_data, dim, torch.device("cuda:"+str(device_num)), model_save)


