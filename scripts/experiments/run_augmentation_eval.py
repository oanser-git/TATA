#!/usr/bin/env python3
"""
Runner script for Section 4.4: Augmentation Evaluation.
Evaluates trained RL agents with different reward configurations.

Usage:
    python scripts/experiments/run_augmentation_eval.py \
        --agent-dir d3rlpy_logs \
        --encoder-path data/artifacts/encoder.pt \
        --dataset ids2017
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tata.datasets.loaders import load_dataset
from tata.experiments.augmentation import evaluate_reward_configs
from tata.rl.environment import TataRLEnvironment
from tata.embedding.encoder import Encoder
from tata.embedding.clustering import run_clustering_pipeline
from tata.testbed.mock_testbed import MockTestbed
from tata.models.nids import RandomForestNIDS, SVMNIDS, DNNNIDS


def main():
    parser = argparse.ArgumentParser(description="Evaluate augmentation policies")
    parser.add_argument("--agent-dir", type=str, default="d3rlpy_logs",
                        help="Directory containing trained agent models")
    parser.add_argument("--encoder-path", type=str, default="data/artifacts/encoder.pt")
    parser.add_argument("--dataset", type=str, default="ids2017")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default="data/artifacts/augmentation")
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
        X = np.asarray(data["X"])
        y = np.asarray(data["y"])
    else:
        print(f"Loading dataset: {args.dataset}...")
        X, y = load_dataset(args.dataset, data_dir=args.data_dir)

    # Paper-standard 60/20/20 split
    _X_train, _X_temp, _y_train, _y_temp = train_test_split(
        X, y, test_size=0.4, random_state=args.random_state, stratify=y
    )
    _X_val, _X_test, _y_val, _y_test = train_test_split(
        _X_temp, _y_temp, test_size=0.5, random_state=args.random_state, stratify=_y_temp
    )
    X_train = np.asarray(_X_train)
    X_val = np.asarray(_X_val)
    X_test = np.asarray(_X_test)
    y_train = np.asarray(_y_train)
    y_val = np.asarray(_y_val)
    y_test = np.asarray(_y_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Load encoder
    encoder_path = Path(args.encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")

    print(f"Loading encoder from {encoder_path}...")
    encoder = Encoder.from_checkpoint(encoder_path, device=args.device)

    # Compute embeddings
    train_emb = encoder.encode(X_train_s)
    test_emb = encoder.encode(X_test_s)

    # Clustering
    best_k, kmeans, metadata = run_clustering_pipeline(
        train_emb, y_train, random_state=args.random_state
    )

    # Mock testbed
    testbed = MockTestbed(
        train_features=X_train,
        scaler=scaler,
        seed=args.random_state,
    )

    # Environment factory
    def env_factory():
        return TataRLEnvironment(
            encoder=encoder,
            cluster_metadata=metadata,
            initial_test_embeddings=test_emb,
            initial_test_labels=y_test,
            train_embeddings=train_emb,
            train_labels=y_train,
            testbed=testbed,
            max_steps_per_episode=5000,
            target_reward=0.9,
        )

    # Discover agent paths
    agent_dir = Path(args.agent_dir)
    agent_paths = {}
    for subdir in agent_dir.iterdir():
        if subdir.is_dir():
            model_file = subdir / "model.d3"
            if model_file.exists():
                agent_paths[subdir.name] = str(model_file)

    if not agent_paths:
        print(f"Warning: No agents found in {agent_dir}")
        return

    print(f"Found {len(agent_paths)} agents: {list(agent_paths.keys())}")

    # NIDS models
    print("Training baseline NIDS models...")
    nids_models = [RandomForestNIDS(), SVMNIDS(), DNNNIDS()]
    for m in nids_models:
        m.fit(X_train_s, y_train)

    print(f"\nRunning augmentation evaluation with {args.n_episodes} episodes...")
    results = evaluate_reward_configs(
        env_factory=env_factory,
        agent_paths=agent_paths,
        X_train=X_train_s,
        y_train=y_train,
        X_holdout=X_val_s,
        y_holdout=y_val,
        nids_models=nids_models,
        n_episodes=args.n_episodes,
        save_dir=args.save_dir,
        verbose=True,
    )

    print("\n=== Augmentation evaluation complete ===")
    print(f"Results saved to: {args.save_dir}/augmentation_results.json")


if __name__ == "__main__":
    main()
