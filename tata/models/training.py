"""
Training utilities for the contrastive autoencoder.
Handles pair mining, training loop, and checkpointing.
"""

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn

from tata.models.autoencoder import ContrastiveAutoencoder, TataLoss


def sample_pairs(
    labels: np.ndarray,
    n_pairs: int,
    strategy: str = "random",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample pairs of indices for contrastive loss.
    
    Args:
        labels: Array of integer labels (n_samples,).
        n_pairs: Number of pairs to sample.
        strategy: 'random' or 'hard' (hard mining not implemented yet).
    
    Returns:
        idx_i: (n_pairs,) first indices.
        idx_j: (n_pairs,) second indices.
        pair_labels: (n_pairs,) 0 = same class, 1 = different class.
    """
    n_samples = len(labels)
    idx_i = []
    idx_j = []
    pair_labels = []
    
    for _ in range(n_pairs):
        # Randomly decide positive or negative pair
        is_negative = random.random() > 0.5
        
        if is_negative:
            # Sample two different classes
            classes = np.unique(labels)
            if len(classes) < 2:
                # Fallback to positive pair
                is_negative = False
            else:
                c1, c2 = random.sample(classes.tolist(), 2)
                idx_c1 = np.where(labels == c1)[0]
                idx_c2 = np.where(labels == c2)[0]
                i = random.choice(idx_c1)
                j = random.choice(idx_c2)
                idx_i.append(i)
                idx_j.append(j)
                pair_labels.append(1)
                continue
        
        # Positive pair: same class
        c = random.choice(np.unique(labels).tolist())
        idx_c = np.where(labels == c)[0]
        if len(idx_c) < 2:
            # If only one sample, pair with itself
            i = j = idx_c[0]
        else:
            i, j = random.sample(idx_c.tolist(), 2)
        idx_i.append(i)
        idx_j.append(j)
        pair_labels.append(0)
    
    return (
        np.array(idx_i),
        np.array(idx_j),
        np.array(pair_labels, dtype=np.float32),
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: TataLoss,
    optimizer: torch.optim.Optimizer,
    pair_sampling_strategy: str = "random",
    pairs_per_batch: int = 64,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dict of average losses.
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_contrastive = 0.0
    n_batches = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.numpy()
        
        # Forward pass for reconstruction
        z, x_recon = model(batch_x)
        
        # Sample pairs for contrastive loss
        idx_i, idx_j, pair_labels = sample_pairs(
            batch_y, n_pairs=pairs_per_batch, strategy=pair_sampling_strategy
        )
        
        z_i = z[idx_i]
        z_j = z[idx_j]
        pair_labels_t = torch.tensor(pair_labels, dtype=torch.float32, device=device)
        
        # Compute loss
        loss, loss_dict = loss_fn(batch_x, x_recon, z_i, z_j, pair_labels_t)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss_dict["total"]
        total_recon += loss_dict["reconstruction"]
        total_contrastive += loss_dict["contrastive"]
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "reconstruction": total_recon / n_batches,
        "contrastive": total_contrastive / n_batches,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: TataLoss,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Validation pass (no training, just reconstruction loss).
    
    Returns:
        Dict with validation MSE.
    """
    model.eval()
    total_mse = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            z, x_recon = model(batch_x)
            mse = nn.functional.mse_loss(x_recon, batch_x)
            total_mse += mse.item()
            n_batches += 1
    
    return {"mse": total_mse / n_batches}


def train_autoencoder(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    epochs: int = 250,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    margin: float = 10.0,
    lambda_contrastive: float = 0.1,
    pair_sampling_strategy: str = "random",
    pairs_per_batch: int = 64,
    device: str = "cpu",
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict]:
    """
    Full training loop for contrastive autoencoder.
    
    Args:
        model: ContrastiveAutoencoder instance.
        X_train: Training features.
        y_train: Training labels (integer-encoded).
        X_val: Validation features (optional).
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        margin: Contrastive loss margin.
        lambda_contrastive: Weight for contrastive loss.
        pair_sampling_strategy: Pair sampling strategy.
        pairs_per_batch: Number of pairs per batch.
        device: 'cpu' or 'cuda'.
        checkpoint_dir: Directory to save checkpoints.
        verbose: Print progress.
    
    Returns:
        Trained model, history dict.
    """
    model.to(device)
    
    # Datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if X_val is not None:
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.zeros(len(X_val), dtype=torch.long),  # dummy labels
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss and optimizer
    loss_fn = TataLoss(margin=margin, lambda_contrastive=lambda_contrastive)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    history = {"train_loss": [], "train_recon": [], "train_contrastive": [], "val_mse": []}
    best_val_mse = float("inf")
    
    for epoch in range(epochs):
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer,
            pair_sampling_strategy=pair_sampling_strategy,
            pairs_per_batch=pairs_per_batch,
            device=device,
        )
        
        history["train_loss"].append(train_metrics["loss"])
        history["train_recon"].append(train_metrics["reconstruction"])
        history["train_contrastive"].append(train_metrics["contrastive"])
        
        val_str = ""
        if val_loader is not None:
            val_metrics = validate(model, val_loader, loss_fn, device=device)
            history["val_mse"].append(val_metrics["mse"])
            val_str = f" | Val MSE: {val_metrics['mse']:.6f}"
            
            if val_metrics["mse"] < best_val_mse and checkpoint_dir is not None:
                best_val_mse = val_metrics["mse"]
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), Path(checkpoint_dir) / "best_model.pt")
        
        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {train_metrics['loss']:.6f} | "
                f"Recon: {train_metrics['reconstruction']:.6f} | "
                f"Contrastive: {train_metrics['contrastive']:.6f}{val_str}"
            )
    
    return model, history
