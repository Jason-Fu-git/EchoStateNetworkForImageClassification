import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the MLP
        :param x: a tensor of shape (batch_size, input_size)
        :return: y is a tensor of shape (batch_size, output_size)
        """
        h = self.tanh(self.fc1(x))
        y = self.softmax(self.fc2(h))
        return y