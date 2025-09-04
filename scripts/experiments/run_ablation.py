#!/usr/bin/env python3
"""
Runner script for Section 4.2: Autoencoder Ablation Study.
Compares Contrastive Autoencoder vs Vanilla Autoencoder.

Usage:
    python scripts/experiments/run_ablation.py --dataset ids2017 --n-splits 10
"""

import argparse
from pathlib import Path

import numpy as np

from tata.datasets.loaders import load_dataset
from tata.experiments.ablation_ae import run_ablation


def main():
    parser = argparse.ArgumentParser(description="Run autoencoder ablation study")
    parser.add_argument("--dataset", type=str, default="ids2017",
                        help="Dataset name (e.g., ids2017, ids2018)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Optional: path to processed dataset .npz file")
    parser.add_argument("--n-splits", type=int, default=10,
                        help="Number of random train/test splits")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="data/artifacts/ablation")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.data_path is not None:
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        print(f"Loading dataset from {data_path}...")
        data = np.load(data_path)
        X = data["X"]
        y = data["y"]
    else:
        print(f"Loading dataset: {args.dataset}...")
        X, y = load_dataset(args.dataset, data_dir=args.data_dir)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    print(f"\nRunning ablation study with {args.n_splits} splits...")
    results = run_ablation(
        X=X,
        y=y,
        n_splits=args.n_splits,
        test_size=args.test_size,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        device=args.device,
        random_state=args.random_state,
        save_dir=args.save_dir,
        verbose=True,
    )
    
    print("\n=== Ablation study complete ===")
    print(f"Results saved to: {args.save_dir}/ablation_results.json")


if __name__ == "__main__":
    main()
