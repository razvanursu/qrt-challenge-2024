import numpy as np

from base import BinaryClassificationModel

class ConstantTrueModel(BinaryClassificationModel):
    """
    A model that always predicts the constant value `1` (true).
    """

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        No training is required for the constant true model.
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the constant value `1` for all inputs.
        
        Args:
            X (np.ndarray): Input features.
            
        Returns:
            np.ndarray: An array of ones with the same length as the input.
        """
        return np.ones(X.shape[0], dtype=int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the model's performance by comparing constant predictions (all 1s) to true labels.
        
        Args:
            X (np.ndarray): Input features (not used).
            y (np.ndarray): True labels.
            
        Returns:
            float: Accuracy of the constant true model.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy