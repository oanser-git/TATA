"""
End-to-end script for Phase 1: Test Set Assessment.
Loads trained encoder and clusters, computes D, P, S metrics.

Artifact requirements (in data/artifacts/ or --artifacts-dir):
  - splits.npz          (train/test data splits)
  - encoder.pt          (trained contrastive autoencoder with embedded config)
  - kmeans.pkl          (k-means model)
  - cluster_metadata.json
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tata.embedding.encoder import Encoder
from tata.metrics.assessment import assess_test_set


def _check_artifacts(artifacts_dir: Path):
    """Verify all required artifacts exist."""
    required = {
        "splits.npz": "Run scripts/run_preliminary.py first to generate data splits.",
        "encoder.pt": "Run scripts/run_preliminary.py first to train the encoder.",
        "kmeans.pkl": "Run scripts/run_preliminary.py first to compute clusters.",
        "cluster_metadata.json": "Run scripts/run_preliminary.py first to save cluster metadata.",
    }
    missing = []
    for fname, hint in required.items():
        if not (artifacts_dir / fname).exists():
            missing.append(f"  - {fname}: {hint}")
    if missing:
        raise FileNotFoundError(
            f"Missing required artifacts in {artifacts_dir}:\n" + "\n".join(missing)
        )


def run_assessment_phase(
    data_config_path: str = "configs/data.yaml",
    metrics_config_path: str = "configs/metrics.yaml",
    artifacts_dir: str = "data/artifacts",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run Phase 1 assessment on the test set.
    
    Args:
        data_config_path: Path to data config.
        metrics_config_path: Path to metrics config.
        artifacts_dir: Directory with trained encoder and clusters.
        seed: Random seed (must match preliminary phase).
    
    Returns:
        Dict with assessment results.
    """
    artifacts = Path(artifacts_dir)
    
    # Validate artifacts exist before loading anything
    _check_artifacts(artifacts)
    
    # Load configs
    with open(metrics_config_path) as f:
        metrics_cfg = yaml.safe_load(f)
    
    # Load splits (preprocessed, scaled data from preliminary phase)
    print("[Assessment] Loading data splits...")
    splits = np.load(artifacts / "splits.npz")
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    
    # Load cluster metadata
    with open(artifacts / "cluster_metadata.json") as f:
        cluster_meta_json = json.load(f)
    
    # Reconstruct cluster metadata dict
    cluster_metadata: dict[str, Any] = {
        "n_clusters": cluster_meta_json["n_clusters"],
        "centroids": np.array(cluster_meta_json["centroids"]),
        "cluster_majority_labels": cluster_meta_json["cluster_majority_labels"],
        "cluster_member_indices": {},  # Will be populated below
        "km_accuracy": cluster_meta_json["km_accuracy"],
        "silhouette_score": cluster_meta_json["silhouette_score"],
    }
    
    # Load encoder (new format with embedded config — no need for model_config.json)
    print("[Assessment] Loading encoder...")
    encoder = Encoder.from_checkpoint(artifacts / "encoder.pt")
    
    # Encode train and test
    print("[Assessment] Encoding datasets...")
    train_embeddings = encoder.encode(X_train)
    test_embeddings = encoder.encode(X_test)
    
    # Build cluster member indices from train embeddings
    with open(artifacts / "kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    cluster_labels = kmeans.labels_
    n_clusters = int(cluster_metadata["n_clusters"])
    for c in range(n_clusters):
        cluster_metadata["cluster_member_indices"][c] = np.where(cluster_labels == c)[0].tolist()
    
    # Run assessment
    print("[Assessment] Computing metrics...")
    results = assess_test_set(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        train_labels=y_train,
        test_labels=y_test,
        cluster_metadata=cluster_metadata,
    )
    
    # Save results
    results_path = artifacts / "assessment_results.json"
    with open(results_path, "w") as f:
        json.dump(results["global"], f, indent=2)
    
    per_cluster_path = artifacts / "assessment_per_cluster.json"
    with open(per_cluster_path, "w") as f:
        json.dump(results["per_cluster"], f, indent=2)
    
    print("\n[Assessment] Phase 1 complete!")
    print(f"  Diversity (D): {results['global']['diversity']:.4f}")
    print(f"  Proximity (P): {results['global']['proximity']:.4f}")
    print(f"  Scarcity (S):  {results['global']['scarcity']:.4f}")
    
    return results["global"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TATA Phase 1 Assessment")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--metrics-config", default="configs/metrics.yaml")
    parser.add_argument("--artifacts-dir", default="data/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_assessment_phase(
        data_config_path=args.data_config,
        metrics_config_path=args.metrics_config,
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
    )
