"""
SVM NIDS implementation.
"""

from typing import Optional

import numpy as np
from sklearn.svm import SVC

from tata.models.nids.base import AbstractNIDS


class SVMNIDS(AbstractNIDS):
    """
    Support Vector Machine classifier for NIDS.
    
    Paper uses this as one of three evaluation models.
    """
    
    def __init__(self, config: dict = None, random_state: int = 42):
        if config is None:
            config = {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "class_weight": "balanced",
                "probability": True,
                "max_iter": 10000,
            }
        super().__init__(config, random_state)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SVMNIDS":
        """Train SVM."""
        self.model = SVC(
            C=self.config.get("C", 1.0),
            kernel=self.config.get("kernel", "rbf"),
            gamma=self.config.get("gamma", "scale"),
            class_weight=self.config.get("class_weight", "balanced"),
            probability=self.config.get("probability", True),
            random_state=self.random_state,
            max_iter=self.config.get("max_iter", 10000),
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X_test)
