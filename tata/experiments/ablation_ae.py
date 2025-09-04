"""
Ablation study: Contrastive Autoencoder vs Vanilla Autoencoder.
Section 4.2 of the paper.

Trains both encoder variants on N random train/test splits,
computes D/P/S assessment scores for each test set,
and compares average performance.
"""

import json
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tata.models.autoencoders import ContrastiveAutoencoder, VanillaAutoencoder
from tata.models.training import train_autoencoder
from tata.embedding.clustering import build_cluster_metadata, run_clustering_pipeline
from tata.metrics.assessment import assess_test_set


def run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    test_size: float = 0.2,
    encoder_dims: list[int] = [64, 32],
    latent_dim: int = 3,
    decoder_dims: list[int] = [32, 64],
    epochs: int = 250,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    lambda_contrastive: float = 0.1,
    margin: float = 10.0,
    k_range: range = range(2, 11),
    device: str = "cpu",
    random_state: int = 42,
    save_dir: str = "data/artifacts/ablation",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run ablation study comparing contrastive vs vanilla autoencoder.
    
    For each of n_splits random train/test splits:
      1. Fit scaler on train, transform both sets
      2. Train contrastive AE on train set
      3. Train vanilla AE on train set (same architecture)
      4. Fit k-means on contrastive train embeddings, assess test
      5. Fit k-means on vanilla train embeddings, assess test
    
    Args:
        X: Features (n_samples, n_features).
        y: Labels (n_samples,).
        n_splits: Number of random train/test splits.
        test_size: Fraction for test set.
        encoder_dims: Hidden dims for encoder.
        latent_dim: Latent dimension.
        decoder_dims: Hidden dims for decoder.
        epochs: Training epochs.
        batch_size: Batch size.
        learning_rate: Adam LR.
        lambda_contrastive: Weight for contrastive loss.
        margin: Contrastive margin.
        k_range: Range of k values for k-means search.
        device: 'cpu' or 'cuda'.
        random_state: Base random seed.
        save_dir: Where to save results.
        verbose: Print progress.
    
    Returns:
        Results dict with scores per split and averages.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    results: dict[str, Any] = {
        "contrastive": [],
        "vanilla": [],
        "splits": [],
    }
    
    for split_idx in range(n_splits):
        split_seed = random_state + split_idx
        if verbose:
            print(f"\n=== Split {split_idx + 1}/{n_splits} (seed={split_seed}) ===")
        
        # Paper: 60%-Dtrain splits (ablation has no val set)
        _X_train, _X_test, _y_train, _y_test = train_test_split(
            X, y, test_size=0.4, random_state=split_seed, stratify=y
        )
        X_train: np.ndarray = np.asarray(_X_train)
        X_test: np.ndarray = np.asarray(_X_test)
        y_train: np.ndarray = np.asarray(_y_train)
        y_test: np.ndarray = np.asarray(_y_test)
        
        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # --- Contrastive Autoencoder ---
        if verbose:
            print("  Training contrastive autoencoder...")
        
        cae = ContrastiveAutoencoder(
            input_dim=X_train_s.shape[1],
            encoder_dims=encoder_dims,
            latent_dim=latent_dim,
            decoder_dims=decoder_dims,
        )
        cae, _ = train_autoencoder(
            cae, X_train_s, y_train,
            epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate,
            lambda_contrastive=lambda_contrastive,
            margin=margin,
            device=device,
            verbose=False,
        )
        cae = cast(ContrastiveAutoencoder, cae)
        
        # Get embeddings
        cae.eval()
        with torch.no_grad():
            train_emb_c = cae.encode(torch.tensor(X_train_s, dtype=torch.float32, device=device)).cpu().numpy()
            test_emb_c = cae.encode(torch.tensor(X_test_s, dtype=torch.float32, device=device)).cpu().numpy()
        
        # Clustering
        best_k_c, kmeans_c, metadata_c = run_clustering_pipeline(
            train_emb_c, y_train, k_range=list(k_range), random_state=split_seed
        )
        
        scores_c = assess_test_set(
            train_emb_c, test_emb_c, y_train, y_test,
            cluster_metadata=metadata_c,
        )
        
        # --- Vanilla Autoencoder ---
        if verbose:
            print("  Training vanilla autoencoder...")
        
        vae = VanillaAutoencoder(
            input_dim=X_train_s.shape[1],
            encoder_dims=encoder_dims,
            latent_dim=latent_dim,
            decoder_dims=decoder_dims,
        )
        vae, _ = train_autoencoder(
            vae, X_train_s, y_train,
            epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate,
            lambda_contrastive=0.0,
            margin=margin,
            device=device,
            verbose=False,
        )
        vae = cast(VanillaAutoencoder, vae)
        
        # Get embeddings
        vae.eval()
        with torch.no_grad():
            train_emb_v = vae.encode(torch.tensor(X_train_s, dtype=torch.float32, device=device)).cpu().numpy()
            test_emb_v = vae.encode(torch.tensor(X_test_s, dtype=torch.float32, device=device)).cpu().numpy()
        
        best_k_v, kmeans_v, metadata_v = run_clustering_pipeline(
            train_emb_v, y_train, k_range=list(k_range), random_state=split_seed
        )
        
        scores_v = assess_test_set(
            train_emb_v, test_emb_v, y_train, y_test,
            cluster_metadata=metadata_v,
        )
        
        results["contrastive"].append({
            "split": split_idx,
            "scores": scores_c,
        })
        results["vanilla"].append({
            "split": split_idx,
            "scores": scores_v,
        })
        results["splits"].append({
            "n_train": len(X_train),
            "n_test": len(X_test),
        })
        
        if verbose:
            print(f"    Contrastive  -> D: {scores_c['global']['diversity']:.4f}, P: {scores_c['global']['proximity']:.4f}, S: {scores_c['global']['scarcity']:.4f}")
            print(f"    Vanilla      -> D: {scores_v['global']['diversity']:.4f}, P: {scores_v['global']['proximity']:.4f}, S: {scores_v['global']['scarcity']:.4f}")
    
    # Compute averages
    def _avg(metric: str, key: str) -> float:
        vals = [r["scores"]["global"][metric] for r in results[key]]
        return float(np.mean(vals))
    
    results["summary"] = {
        "contrastive": {
            "diversity_mean": _avg("diversity", "contrastive"),
            "proximity_mean": _avg("proximity", "contrastive"),
            "scarcity_mean": _avg("scarcity", "contrastive"),
        },
        "vanilla": {
            "diversity_mean": _avg("diversity", "vanilla"),
            "proximity_mean": _avg("proximity", "vanilla"),
            "scarcity_mean": _avg("scarcity", "vanilla"),
        },
    }
    
    # Save results
    with open(save_path / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print("\n=== Ablation Summary ===")
        print(f"Contrastive -> D: {results['summary']['contrastive']['diversity_mean']:.4f}, "
              f"P: {results['summary']['contrastive']['proximity_mean']:.4f}, "
              f"S: {results['summary']['contrastive']['scarcity_mean']:.4f}")
        print(f"Vanilla     -> D: {results['summary']['vanilla']['diversity_mean']:.4f}, "
              f"P: {results['summary']['vanilla']['proximity_mean']:.4f}, "
              f"S: {results['summary']['vanilla']['scarcity_mean']:.4f}")
    
    return results
