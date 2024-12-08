import torch
import torch.nn as nn
import torch.optim as optim

from .nn import NeuralNetworkModel

class AdvNN1(NeuralNetworkModel):
    
    def __init__(self, input_dim, hidden_dim, num_epochs):
        super().__init__(input_dim, hidden_dim, num_epochs)
        
        # Define the neural network architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)


class AdvNN2(NeuralNetworkModel):
    
    def __init__(self, input_dim, hidden_dim, num_epochs):
        super().__init__(input_dim, hidden_dim, num_epochs)
        
        # Define the neural network architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)