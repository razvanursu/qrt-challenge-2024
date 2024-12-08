import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score

from .base import BinaryClassificationModel 

class NeuralNetworkModel(BinaryClassificationModel):
    """
    A binary classification model using a neural network.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_epochs: int):
        """
        Initialize the neural network model.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
        """
        super().__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.num_epochs = num_epochs
        
        # Define the neural network architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the neural network on the given data.
        
        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels (binary: 0 or 1).
            epochs (int): Number of training epochs.
        """
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        for epoch in range(self.num_epochs):
            # Forward pass
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given data.
        
        Args:
            X (np.ndarray): Input features.
            
        Returns:
            np.ndarray: Predicted binary labels (0 or 1).
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int)
        return predictions.flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model's performance.
        
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True labels.
            
        Returns:
            float: Accuracy of the model.
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)