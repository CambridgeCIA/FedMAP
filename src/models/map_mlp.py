import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with two hidden layers.
    """
    def __init__(self, input_dim):
        super(MLP, self).__init__()

        # Define the layers of the MLP
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.layers(x)
