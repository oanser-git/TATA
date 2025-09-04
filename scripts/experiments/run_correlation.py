#!/usr/bin/env python3
"""
Runner script for Section 4.3: Correlation Study.
Computes correlation between D/P/S scores and actual NIDS performance.

Usage:
    python scripts/experiments/run_correlation.py --dataset ids2017 --encoder-path data/artifacts/encoder.pt
"""

import argparse
from pathlib import Path

import numpy as np

from tata.datasets.loaders import load_dataset
from tata.experiments.correlation import run_correlation_study
from tata.embedding.encoder import Encoder


def main():
    parser = argparse.ArgumentParser(description="Run correlation study")
    parser.add_argument("--dataset", type=str, default="ids2017")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--encoder-path", type=str, default="data/artifacts/encoder.pt")
    parser.add_argument("--save-dir", type=str, default="data/artifacts/correlation")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load data
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

    # Load encoder
    encoder_path = Path(args.encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}. Run scripts/run_preliminary.py first.")

    print(f"Loading encoder from {encoder_path}...")
    encoder = Encoder.from_checkpoint(encoder_path, device=args.device)

    # Create encoder callable
    def encode_fn(X_features: np.ndarray) -> np.ndarray:
        return encoder.encode(X_features)

    print("\nRunning correlation study (Diversity, Proximity, Scarcity)...")
    results = run_correlation_study(
        X=X,
        y=y,
        encoder_callable=encode_fn,
        random_state=args.random_state,
        save_dir=args.save_dir,
        verbose=True,
    )

    print("\n=== Correlation study complete ===")
    print(f"Results saved to: {args.save_dir}/correlation_results.json")


if __name__ == "__main__":
    main()
