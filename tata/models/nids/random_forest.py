"""
Random Forest NIDS implementation.
"""

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from tata.models.nids.base import AbstractNIDS


class RandomForestNIDS(AbstractNIDS):
    """
    Random Forest classifier for NIDS.
    
    Paper uses this as one of three evaluation models.
    """
    
    def __init__(self, config: dict = None, random_state: int = 42):
        if config is None:
            config = {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "n_jobs": -1,
            }
        super().__init__(config, random_state)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestNIDS":
        """Train Random Forest."""
        self.model = RandomForestClassifier(
            n_estimators=self.config.get("n_estimators", 200),
            max_depth=self.config.get("max_depth", None),
            min_samples_split=self.config.get("min_samples_split", 2),
            min_samples_leaf=self.config.get("min_samples_leaf", 1),
            max_features=self.config.get("max_features", "sqrt"),
            random_state=self.random_state,
            n_jobs=self.config.get("n_jobs", -1),
            class_weight=self.config.get("class_weight", "balanced"),
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
