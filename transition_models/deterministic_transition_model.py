import random
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from transition_models.transition_model import TransitionModel

# class MultiOutputModel(nn.Module):
#     def __init__(self, dim=4096):
#         super(MultiOutputModel, self).__init__()
        
#         # shared
#         self.fc1 = nn.Linear(dim, int(dim/2))
#         self.fc2 = nn.Linear(int(dim/2), int(dim/4))
#         self.fc3 = nn.Linear(int(dim/4), int(dim/8))
        
#         # number of independent network equals to dim
#         self.separate_network = []
#         for o in range(dim):
#             self.separate_network.append(nn.Sequential(nn.Linear(int(dim/8), int(dim/16)),
#                                                        nn.ReLU(), 
#                                                        nn.Linear(int(dim/16), int(dim/32)),
#                                                        nn.ReLU(),
#                                                        nn.Linear(int(dim/32), 1)).to(torch.device('cuda:0')))
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
        
#         output = []
#         for network in self.separate_network:
            
#             y = network(x)
#             output.append(y)
#         concatenated_tensor = torch.cat(output, dim=1)
#         return concatenated_tensor

def mean_abs_percentage_loss_fn(output, target):
# MAPE loss, no percentage just absolute
    return torch.mean(torch.abs((target - output)))

class MultiOutputModel(nn.Module):
    def __init__(self, embedding_dimension=4096):
        super(MultiOutputModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dimension, int(embedding_dimension/2)),
            nn.ReLU(),
            #nn.BatchNorm1d(num_features=int(embedding_dimension/2)),
            nn.Linear(int(embedding_dimension/2), int(embedding_dimension/4)),
            nn.ReLU(),
            nn.Linear(int(embedding_dimension/4), int(embedding_dimension/8)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(int(embedding_dimension/8), int(embedding_dimension/16)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(int(embedding_dimension/16), int(embedding_dimension/16)),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(int(embedding_dimension/16), int(embedding_dimension/8)),
            nn.ReLU(),
            nn.Linear(int(embedding_dimension/8), int(embedding_dimension/4)),
            nn.ReLU(),
            nn.Linear(int(embedding_dimension/4), int(embedding_dimension/2)),
            nn.ReLU(),
            nn.Linear(int(embedding_dimension/2), int(embedding_dimension))
        )
    def forward(self, x):
        return self.model(x)
    

'''
input and output data are all in tuple format (due to the need for dict hashing in MCTS procedure).
'''
class DeterministicTransitionModel(TransitionModel):
    def __init__(self, embedding_dimension, samples=1, cuda = torch.device('cuda:0')) -> None:
        self.embedding_dimension = embedding_dimension
        self.samples = samples
        self.cuda = cuda
        self.env_transition_model = MultiOutputModel(embedding_dimension).to(self.cuda)
        self.llm_transition_model = MultiOutputModel(embedding_dimension).to(self.cuda)
        
        #self.env_independent_transition_model = MultiOutputModel(embedding_dimension).to(self.cuda)
        pass
    
    def train_llm_transition(self, train_loader, val_loader, path_to_save=None, num_epochs=500, lr=0.001):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.llm_transition_model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        t_loss_all = []
        v_loss_all = []
        # Training loop
        for epoch in range(num_epochs):
            
            self.llm_transition_model.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.llm_transition_model(inputs.to(self.cuda))
                train_loss = criterion(outputs, targets.to(self.cuda))
                
                # Backward pass and optimize
                train_loss.backward()
                optimizer.step()

            scheduler.step()
            print(scheduler.get_last_lr())
            # Validation phase
            self.llm_transition_model.train().eval()  # Set the model to evaluation mode
            val_loss = 0.0
            percentage_loss = 0.0

            with torch.no_grad():  # Disable gradient computation for validation
                for inputs, labels in val_loader:
                    # Move inputs and labels to device (GPU/CPU)
                    inputs, labels = inputs.to(self.cuda), labels.to(self.cuda)
                    
                    # Forward pass
                    outputs = self.llm_transition_model(inputs)
                    loss = criterion(outputs, labels)
                    percentage_loss += mean_abs_percentage_loss_fn(outputs, labels).item()
                    # Accumulate the validation loss
                    val_loss += loss.item()
            
            # Calculate the average validation loss
            val_loss /= len(val_loader)
            percentage_loss /= len(val_loader)
            
            t_loss_all.append(train_loss)
            v_loss_all.append(val_loss)
            # Print statistics
            print(f'Epoch {epoch+1}/{num_epochs}, '
                f'Train Loss: {train_loss:.8f}, '
                f'Validation Loss: {val_loss:.8f},'
                f'Validation Percentage Loss: {percentage_loss:.8f},')
        if path_to_save is not None:
            torch.save(self.llm_transition_model, "models/deterministic/"+str(path_to_save))
        return t_loss_all, v_loss_all
        
    def train_env_transition(self, train_loader, val_loader, path_to_save=None, num_epochs=500, lr=0.001):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.env_transition_model.parameters(), lr=lr)

        # Training loop
        for epoch in range(num_epochs):
            self.env_transition_model.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.env_transition_model(inputs.to(self.cuda))
                train_loss = criterion(outputs, targets.to(self.cuda))
                
                # Backward pass and optimize
                train_loss.backward()
                optimizer.step()

        
            # Validation phase
            self.env_transition_model.train().eval()  # Set the model to evaluation mode
            val_loss = 0.0
            
            with torch.no_grad():  # Disable gradient computation for validation
                for inputs, labels in val_loader:
                    # Move inputs and labels to device (GPU/CPU)
                    inputs, labels = inputs.to(self.cuda), labels.to(self.cuda)
                    
                    # Forward pass
                    outputs = self.env_transition_model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Accumulate the validation loss
                    val_loss += loss.item()
            
            # Calculate the average validation loss
            val_loss /= len(val_loader)
            
            # Print statistics
            print(f'Epoch {epoch+1}/{num_epochs}, '
                f'Train Loss: {train_loss:.8f}, '
                f'Validation Loss: {val_loss:.8f}')
        if path_to_save is not None:
            torch.save(self.env_transition_model, "models/deterministic/"+str(path_to_save))
    
    def load_llm_transition_model(self, path):
        self.llm_transition_model = MultiOutputModel(self.embedding_dimension)
        self.llm_transition_model.load_state_dict(torch.load(path))
        
        
    def load_env_transition_model(self, path):
        self.env_transition_model = MultiOutputModel(self.embedding_dimension)
        self.env_transition_model.load_state_dict(torch.load(path))
        
    # given a state (n-dim embedding), action (n-dim directional vector), account for human stochastic response and return a new state
    def transit(self, state, action):
        
        # convert tuple to np array
        state = np.array(list(state))
        action = np.array(list(action))
        
        intermediate_state = state + action # action is a directional vector, so we can add them directly
        
        # mimic transition for now randomly
        # convert back to tuple format
        new_states = [self.env_transition__model(torch.Tensor(intermediate_state).to(self.cuda))]
        
        return new_states
    
    # given a state (n-dim embedding), return a LLM action (n-dimensional vector)
    def sample_actions(self, state):
        
        # convert tuple to np array
        state = np.array(list(state))
        
        # mimic action for now randomly
        dim = state.shape[0]
        return [self.llm_transition_model(torch.Tensor(state).to(self.cuda))]