"""
Abstract base class for NIDS models.
Provides a unified interface for RF, SVM, and DNN.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class AbstractNIDS(ABC):
    """
    Abstract interface for all NIDS models.
    
    All implementations must support:
      - fit(X_train, y_train): training
      - predict(X_test): class predictions
      - predict_proba(X_test): class probabilities (if available)
      - evaluate(X_test, y_test): compute metrics
      - get_activations(X): internal layer outputs (for neuron coverage baseline)
    """
    
    def __init__(self, config: dict[str, Any], random_state: int = 42):
        self.config = config
        self.random_state = random_state
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "AbstractNIDS":
        """Train the NIDS model."""
        pass
    
    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass
    
    @abstractmethod
    def predict_proba(self, X_test: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities, or None if not supported."""
        pass
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        average: str = "macro",
    ) -> dict[str, float]:
        """
        Evaluate the model on a test set.
        
        Returns:
            Dict with precision, recall, f1-score, accuracy.
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )
        
        if not self.is_trained:
            raise RuntimeError("Model must be fit before evaluation")
        
        y_pred = self.predict(X_test)
        
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=average)),
            "recall": float(recall_score(y_test, y_pred, average=average)),
            "f1": float(f1_score(y_test, y_pred, average=average)),
        }
    
    def get_activations(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Get internal layer activations.
        
        Returns:
            Flattened activation vector, or None for non-neural models.
        """
        return None
