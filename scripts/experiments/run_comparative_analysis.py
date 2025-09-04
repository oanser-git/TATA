#!/usr/bin/env python3
"""
Runner for Section 4.5: Multi-Dataset Comparative Analysis.

Usage:
    python scripts/experiments/run_comparative_analysis.py --datasets ids2017 ids2018 mnist
"""

import argparse

from tata.experiments.comparative import run_comparative_analysis


def main():
    parser = argparse.ArgumentParser(description="Run multi-dataset comparative analysis")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="List of dataset names (default: all from Table 2)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--ae-epochs", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="data/artifacts/comparative")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Dataset Comparative Analysis (Section 4.5)")
    print("=" * 60)
    
    df = run_comparative_analysis(
        datasets=args.datasets,
        data_dir=args.data_dir,
        test_size=args.test_size,
        ae_epochs=args.ae_epochs,
        n_splits=args.n_splits,
        device=args.device,
        random_state=args.random_state,
        save_dir=args.save_dir,
        verbose=True,
    )
    
    print("\nComparative analysis complete.")
    print(f"Results saved to: {args.save_dir}/comparative_results.csv")


if __name__ == "__main__":
    main()
