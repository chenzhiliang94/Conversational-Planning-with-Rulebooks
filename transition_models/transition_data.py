import os
import torch

from torch.utils.data import Dataset, DataLoader


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from csv import reader   

class DailyDialogueTransitionData(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]