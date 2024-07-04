from torch import nn
import torch
from mixture_of_experts import HeirarchicalMoE

class RegressionModel(nn.Module):
    def __init__(self, embedding_size = 1024):
        super(RegressionModel, self).__init__()
        self.add_module("model", HeirarchicalMoE(dim = embedding_size))
        self.input_mean = nn.Parameter(torch.zeros(embedding_size), requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(embedding_size), requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(embedding_size), requires_grad=False)
        self.output_std = nn.Parameter(torch.ones(embedding_size), requires_grad=False)
        self.use_residuals = nn.Parameter(torch.tensor(True, dtype=bool), requires_grad=False)

    def set_parameters(self, input_mean, input_std, output_mean, output_std, use_residuals):
        self.input_mean = nn.Parameter(input_mean, requires_grad=False)
        self.input_std = nn.Parameter(input_std, requires_grad=False)
        self.output_mean = nn.Parameter(output_mean, requires_grad=False)
        self.output_std = nn.Parameter(output_std, requires_grad=False)
        self.use_residuals = nn.Parameter(torch.tensor(use_residuals, dtype=bool), requires_grad=False)

    def forward(self, x):
        y = (x - self.input_mean) / self.input_std
        y = self.model(y[:,None])[0][:,0,:]
        y = y * self.output_std + self.output_mean
        if self.use_residuals:
            y = x + y
        return y