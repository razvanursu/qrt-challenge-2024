from abc import ABC, abstractmethod
import numpy as np

class BinaryClassificationModel(ABC):
    """
    Abstract base class for binary classification models.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the given data.
        
        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels (binary: 0 or 1).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given data.
        
        Args:
            X (np.ndarray): Input features.
            
        Returns:
            np.ndarray: Predicted binary labels (0 or 1).
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model's performance.
        
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True labels.
            
        Returns:
            float: Evaluation metric (e.g., accuracy).
        """
        pass