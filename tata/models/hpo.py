"""
Optuna hyperparameter optimization for the contrastive autoencoder.
Searches encoder architecture, latent dim, learning rate, lambda, margin, etc.
Selects best model based on silhouette-guided k-means accuracy.
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import optuna
import torch

from tata.embedding.clustering import run_clustering_pipeline
from tata.models.autoencoder import ContrastiveAutoencoder
from tata.models.training import train_autoencoder


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    config: Dict,
    device: str = "cpu",
) -> float:
    """
    Optuna objective: train AE with trial hyperparams, return negative KM-accuracy
    (we minimize, so lower = better; thus we return -accuracy).
    
    Args:
        trial: Optuna trial.
        X_train: Training data.
        y_train: Training labels.
        X_val: Validation data (optional).
        config: HPO search space configuration.
        device: 'cpu' or 'cuda'.
    
    Returns:
        Negative k-means accuracy (to minimize).
    """
    # Sample hyperparameters
    arch_choice = trial.suggest_categorical(
        "encoder_arch", config.get("encoder_dims_options", [[64, 32, 16]])
    )
    latent_dim = trial.suggest_int("latent_dim", *config.get("latent_dim", [2, 5]))
    learning_rate = trial.suggest_float("learning_rate", *config.get("learning_rate", [1e-4, 1e-2]), log=True)
    lambda_contrastive = trial.suggest_float("lambda", *config.get("lambda", [0.01, 1.0]), log=True)
    margin = trial.suggest_float("margin", *config.get("margin", [5.0, 20.0]))
    batch_size = trial.suggest_categorical("batch_size", config.get("batch_size", [64, 128, 256]))
    epochs = trial.suggest_int("epochs", *config.get("epochs", [100, 300]))
    
    # Derive decoder dims by reversing encoder dims (excluding latent)
    encoder_dims = arch_choice[:-1] if len(arch_choice) > 1 else arch_choice
    if latent_dim in encoder_dims:
        encoder_dims = [d for d in encoder_dims if d != latent_dim]
    
    # Ensure decoder mirrors encoder
    decoder_dims = list(reversed(encoder_dims))
    
    # Create model
    model = ContrastiveAutoencoder(
        input_dim=X_train.shape[1],
        encoder_dims=encoder_dims,
        latent_dim=latent_dim,
        decoder_dims=decoder_dims,
    )
    
    # Train
    trained_model, history = train_autoencoder(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        margin=margin,
        lambda_contrastive=lambda_contrastive,
        device=device,
        checkpoint_dir=None,
        verbose=False,
    )
    
    # Encode training set
    trained_model.eval()
    with torch.no_grad():
        train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        embeddings = trained_model.encode(train_tensor).cpu().numpy()
    
    # Clustering
    k_range = config.get("k_range", list(range(2, min(21, len(X_train)))))
    best_k, kmeans, metadata = run_clustering_pipeline(
        embeddings=embeddings,
        true_labels=y_train,
        k_range=k_range,
        random_state=42,
        n_init=10,
    )
    
    # Target metric: maximize KM accuracy -> minimize negative accuracy
    km_accuracy = metadata["km_accuracy"]
    
    # Also store silhouette score as user attr
    trial.set_user_attr("silhouette", metadata["silhouette_score"])
    trial.set_user_attr("best_k", best_k)
    trial.set_user_attr("val_mse", history["val_mse"][-1] if history["val_mse"] else float("inf"))
    
    return -km_accuracy  # minimize negative = maximize accuracy


def run_hpo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    n_trials: int = 20,
    study_name: str = "tata_ae_hpo",
    storage: Optional[str] = None,
    config: Optional[Dict] = None,
    device: str = "cpu",
    output_dir: str = "data/artifacts/hpo",
) -> Dict:
    """
    Run Optuna hyperparameter optimization for contrastive autoencoder.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional).
        n_trials: Number of Optuna trials.
        study_name: Optuna study name.
        storage: Optuna storage URL (e.g., sqlite path).
        config: HPO search space config dict.
        device: 'cpu' or 'cuda'.
        output_dir: Directory to save best config and study artifacts.
    
    Returns:
        Dict with best params, best model path, and study summary.
    """
    if config is None:
        config = {
            "encoder_dims_options": [[128, 64, 32], [64, 32, 16], [128, 64, 32, 16]],
            "latent_dim": [2, 5],
            "learning_rate": [1e-4, 1e-2],
            "lambda": [0.01, 1.0],
            "margin": [5.0, 20.0],
            "batch_size": [64, 128, 256],
            "epochs": [100, 300],
            "k_range": list(range(2, 21)),
        }
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, config, device),
        n_trials=n_trials,
    )
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    print(f"\n[HPO Complete] Best trial: {best_trial.number}")
    print(f"  KM Accuracy: {-best_trial.value:.4f}")
    print(f"  Silhouette: {best_trial.user_attrs['silhouette']:.4f}")
    print(f"  Best k: {best_trial.user_attrs['best_k']}")
    print(f"  Params: {json.dumps(best_params, indent=2)}")
    
    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    with open(out / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    with open(out / "study_summary.json", "w") as f:
        json.dump({
            "best_trial": best_trial.number,
            "best_km_accuracy": -best_trial.value,
            "best_silhouette": best_trial.user_attrs["silhouette"],
            "best_k": best_trial.user_attrs["best_k"],
            "n_trials": len(study.trials),
        }, f, indent=2)
    
    return {
        "best_params": best_params,
        "best_trial": best_trial.number,
        "best_km_accuracy": -best_trial.value,
        "best_silhouette": best_trial.user_attrs["silhouette"],
        "study": study,
        "output_dir": str(out),
    }
