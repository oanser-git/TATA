"""
DNN NIDS implementation using PyTorch.
"""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tata.models.nids.base import AbstractNIDS


class DNNClassifier(nn.Module):
    """
    Simple feedforward DNN for multi-class classification.
    
    Architecture: input -> hidden layers -> softmax output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DNNNIDS(AbstractNIDS):
    """
    Deep Neural Network classifier for NIDS.
    
    Paper uses this as one of three evaluation models (DNN).
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None, random_state: int = 42):
        if config is None:
            config = {
                "hidden_dims": [128, 128, 128],
                "dropout": 0.2,
                "epochs": 100,
                "batch_size": 256,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "early_stopping_patience": 10,
                "device": "cpu",
            }
        super().__init__(config, random_state)
        self.device = self.config.get("device", "cpu")
        self._activations: Optional[np.ndarray] = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DNNNIDS":
        """Train DNN classifier."""
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
        
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        self.model = DNNClassifier(
            input_dim=input_dim,
            hidden_dims=self.config.get("hidden_dims", [128, 64, 32]),
            num_classes=num_classes,
            dropout=self.config.get("dropout", 0.2),
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 0.001),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 256),
            shuffle=True,
        )
        
        epochs = self.config.get("epochs", 100)
        patience = self.config.get("early_stopping_patience", 10)
        best_loss = float("inf")
        patience_counter = 0
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        self.is_trained = True
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device)
            outputs = self.model(tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds
    
    def predict_proba(self, X_test: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X_test, dtype=torch.float32, device=self.device)
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs
    
    def get_activations(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract penultimate layer activations for neuron coverage baseline.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        self.model.eval()
        activations = []
        
        def hook(module, input, output):
            activations.append(output.detach().cpu().numpy())
        
        # Register hook on second-to-last layer
        # For DNNClassifier, the penultimate layer is the linear before output
        # Our architecture is: Linear -> ReLU -> Dropout -> ... -> Linear
        # We want the output of the last ReLU before final Linear
        # That's the output of the layer at index -2 in Sequential (the last Linear is -1)
        # Actually, our Sequential is [Linear, ReLU, Dropout, Linear, ReLU, Dropout, ..., Linear]
        # The penultimate activations are the ones right before the final Linear
        # That's the output of the ReLU just before the last Linear
        # In our sequential, that's the element at index -3 (ReLU) or -2 (Dropout)
        # Let's grab the output of the layer at index -3
        handle = self.model.net[-3].register_forward_hook(hook)
        
        try:
            with torch.no_grad():
                tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
                _ = self.model(tensor)
        finally:
            handle.remove()
        
        if activations:
            return np.concatenate(activations, axis=0)
        return None
