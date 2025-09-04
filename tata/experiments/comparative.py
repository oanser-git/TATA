"""
Multi-dataset comparative analysis (Section 4.5).

Trains the contrastive autoencoder and computes D/P/S metrics
for multiple NIDS datasets plus image benchmarks (MNIST, CIFAR-10).
Generates Figure 10: comparative bar chart across datasets.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tata.datasets.loaders import load_dataset
from tata.models.autoencoders import ContrastiveAutoencoder
from tata.models.training import train_autoencoder
from tata.embedding.encoder import Encoder
from tata.embedding.clustering import run_clustering_pipeline
from tata.metrics.assessment import assess_test_set


# Default dataset list from Table 2 in the paper
DEFAULT_DATASETS = [
    "nsl-kdd",
    "ctu-13",
    "iscx-ids2012",
    "unsw-nb15",
    "cic-unsw",
    "iscx-tor",
    "vpn-nonvpn",
    "ids2017",
    "ids2018",
    "bot-iot",
    "cic-ddos2019",
    "ton-iot",
    "mnist",
    "cifar10",
]


def evaluate_dataset(
    dataset_name: str,
    data_dir: str = "data",
    test_size: float = 0.2,
    ae_epochs: int = 100,
    ae_batch_size: int = 256,
    latent_dim: int = 3,
    device: str = "cpu",
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Run full TATA pipeline on a single dataset and return D/P/S scores.
    
    Args:
        dataset_name: Name of dataset to load.
        data_dir: Root data directory.
        test_size: Fraction for test split.
        ae_epochs: Autoencoder training epochs.
        ae_batch_size: AE batch size.
        latent_dim: Latent dimension.
        device: 'cpu' or 'cuda'.
        random_state: Random seed.
        verbose: Print progress.
    
    Returns:
        Dict with diversity, proximity, scarcity, km_accuracy, silhouette_score.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'='*60}")
    
    # Load dataset
    try:
        X, y = load_dataset(dataset_name, data_dir=data_dir)
    except FileNotFoundError as e:
        if verbose:
            print(f"  SKIPPED (not found): {e}")
        return {}
    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        return {}
    
    if verbose:
        print(f"  Loaded: X={X.shape}, classes={len(np.unique(y))}")
    
    # Paper-standard 60/20/20 split
    _X_train, _X_temp, _y_train, _y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    _X_val, _X_test, _y_val, _y_test = train_test_split(
        _X_temp, _y_temp, test_size=0.5, random_state=random_state, stratify=_y_temp
    )
    X_train = np.asarray(_X_train)
    X_val = np.asarray(_X_val)
    X_test = np.asarray(_X_test)
    y_train = np.asarray(_y_train)
    y_val = np.asarray(_y_val)
    y_test = np.asarray(_y_test)
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train contrastive autoencoder
    model = ContrastiveAutoencoder(
        input_dim=X_train_s.shape[1],
        encoder_dims=[128, 64, 32],
        latent_dim=latent_dim,
        decoder_dims=[32, 64, 128],
    )
    
    trained, _ = train_autoencoder(
        model, X_train_s, y_train,
        epochs=ae_epochs, batch_size=ae_batch_size,
        device=device, verbose=False,
    )
    
    encoder = Encoder(trained, device=device)
    train_emb = encoder.encode(X_train_s)
    test_emb = encoder.encode(X_test_s)
    
    # Clustering
    k_range = list(range(2, min(21, len(X_train) // 100 + 1)))
    best_k, kmeans, metadata = run_clustering_pipeline(
        train_emb, y_train, k_range=k_range, random_state=random_state
    )
    
    # Assessment
    scores = assess_test_set(
        train_emb, test_emb, y_train, y_test,
        cluster_metadata=metadata,
    )
    
    result = {
        "dataset": dataset_name,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
        "diversity": float(scores["global"]["diversity"]),
        "proximity": float(scores["global"]["proximity"]),
        "scarcity": float(scores["global"]["scarcity"]),
        "km_accuracy": float(metadata["km_accuracy"]),
        "silhouette_score": float(metadata["silhouette_score"]),
        "best_k": int(best_k),
    }
    
    if verbose:
        print(f"  D={result['diversity']:.3f} P={result['proximity']:.3f} S={result['scarcity']:.3f}")
        print(f"  KM-Acc={result['km_accuracy']:.3f} Sil={result['silhouette_score']:.3f}")
    
    return result


def run_comparative_analysis(
    datasets: Optional[list[str]] = None,
    data_dir: str = "data",
    test_size: float = 0.2,
    ae_epochs: int = 100,
    n_splits: int = 10,
    device: str = "cpu",
    random_state: int = 42,
    save_dir: str = "data/artifacts/comparative",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run comparative analysis across multiple datasets (Section 4.5).
    
    For each dataset:
      1. Load data
      2. Train/test split (multiple times for std)
      3. Train AE + cluster + assess
      4. Average D/P/S across splits
    
    Args:
        datasets: List of dataset names. Defaults to all in Table 2.
        data_dir: Root data directory.
        test_size: Fraction for test split.
        ae_epochs: Autoencoder training epochs.
        n_splits: Number of random train/test splits per dataset.
        device: 'cpu' or 'cuda'.
        random_state: Base random seed.
        save_dir: Where to save results.
        verbose: Print progress.
    
    Returns:
        DataFrame with one row per dataset and columns for D/P/S means/stds.
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for dataset_name in datasets:
        split_results = []
        
        for split_idx in range(n_splits):
            split_seed = random_state + split_idx
            res = evaluate_dataset(
                dataset_name=dataset_name,
                data_dir=data_dir,
                test_size=test_size,
                ae_epochs=ae_epochs,
                device=device,
                random_state=split_seed,
                verbose=(verbose and split_idx == 0),
            )
            if res:
                split_results.append(res)
        
        if not split_results:
            continue
        
        # Aggregate across splits
        diversity_vals = [r["diversity"] for r in split_results]
        proximity_vals = [r["proximity"] for r in split_results]
        scarcity_vals = [r["scarcity"] for r in split_results]
        
        results.append({
            "dataset": dataset_name,
            "diversity_mean": float(np.mean(diversity_vals)),
            "diversity_std": float(np.std(diversity_vals)),
            "proximity_mean": float(np.mean(proximity_vals)),
            "proximity_std": float(np.std(proximity_vals)),
            "scarcity_mean": float(np.mean(scarcity_vals)),
            "scarcity_std": float(np.std(scarcity_vals)),
            "n_samples": split_results[0]["n_samples"],
            "n_features": split_results[0]["n_features"],
            "n_classes": split_results[0]["n_classes"],
        })
    
    df = pd.DataFrame(results)
    
    # Save
    df.to_csv(save_path / "comparative_results.csv", index=False)
    with open(save_path / "comparative_results.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Comparative Analysis Summary")
        print(f"{'='*60}")
        print(df.to_string(index=False))
    
    return df
