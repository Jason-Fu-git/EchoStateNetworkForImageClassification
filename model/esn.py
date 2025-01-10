import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
from torchattacks import PGD


class ESN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1, spectrum_radius=0.9, input_connectivity=1,
                 hidden_connectivity=1, num_iters=100, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.spectrum_radius = spectrum_radius
        self.hidden_connectivity = hidden_connectivity
        self.input_connectivity = input_connectivity
        self.num_iters = num_iters
        self.device = device

        # Input layer, Win is initialized with uniform distribution
        self.Win = torch.rand(input_size, hidden_size) * 0.2 - 0.1
        # sparse connectivity
        mask = torch.rand(input_size, hidden_size)
        mask = mask < input_connectivity
        self.Win = self.Win * mask.float()
        self.Win = nn.Parameter(self.Win)

        # Hidden layer, W is initialized with normal distribution
        self.W = torch.randn(hidden_size, hidden_size) * 0.1
        # sparse connectivity
        mask = torch.rand(hidden_size, hidden_size)
        mask = mask < hidden_connectivity
        self.W = self.W * mask.float()

        # Scaling the spectral radius of W
        self.W = self.W * (self.spectrum_radius / torch.max(torch.abs(torch.linalg.eig(self.W)[0])))
        self.W = nn.Parameter(self.W)

        # linear output layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, w_requires_grad=False):
        """
        Forward pass of the ESN
        :param w_requires_grad: whether to require gradient for W and Win
        :param x: a tensor of shape (batch_size, num_iters, input_size)
        :return: y is a tensor of shape (batch_size, num_iters, output_size)
        """
        self.W.requires_grad = w_requires_grad
        self.Win.requires_grad = w_requires_grad
        h = torch.randn(x.size(0), self.hidden_size).to(self.device)
        y = torch.zeros(x.size(0), self.num_iters, self.output_size).to(self.device)
        for i in range(self.num_iters):
            h = torch.tanh(torch.matmul(x[:, i], self.Win) + torch.matmul(h, self.W))
            y[:, i] = self.linear(h)
        return y

    def forward_with_record(self, x):
        """
        Forward pass of the ESN, record the hidden states at each iteration
        :param x: a tensor of shape (batch_size, input_size)
        :return: hidden states of the ESN layer after num_iters iterations
        """
        # Initialize the hidden states
        h = torch.randn(x.size(0), self.hidden_size).to(self.device)
        h_record = torch.zeros(x.size(0), self.num_iters, self.hidden_size).to(self.device)
        for i in range(self.num_iters):
            h = torch.tanh(torch.matmul(x[:, i], self.Win) + torch.matmul(h, self.W))
            h_record[:, i] = h
        return h_record
