#!/usr/bin/env python3
"""
Main reproduction orchestrator for TATA paper.

Runs the full paper reproduction pipeline end-to-end:
  1. Preliminary Phase: Train contrastive autoencoder (with optional HPO)
  2. Phase 1: Test set assessment (D/P/S metrics)
  3. Phase 2: Offline RL data collection
  4. Section 4.2: Ablation study (contrastive vs vanilla AE)
  5. Section 4.3: Correlation analysis (D/P/S vs NIDS F1)
  6. Appendix B: RL HPO (Split-Select-Retrain)
  7. Section 4.4: Augmentation evaluation
  8. Section 4.5: Multi-dataset comparative analysis

Usage:
    python scripts/reproduce_paper.py --dataset ids2018 --device cpu
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tata.datasets.loaders import load_dataset
from tata.embedding.clustering import run_clustering_pipeline
from tata.embedding.encoder import Encoder
from tata.experiments.ablation_ae import run_ablation
from tata.experiments.correlation import run_correlation_study
from tata.experiments.comparative import run_comparative_analysis
from tata.metrics.assessment import assess_test_set
from tata.models.autoencoders import ContrastiveAutoencoder
from tata.models.hpo import run_hpo
from tata.models.nids import RandomForestNIDS
from tata.models.training import train_autoencoder
from tata.rl.environment import TataRLEnvironment
from tata.rl.hpo import run_split_select_retrain
from tata.rl.offline_dataset import collect_transitions_with_random_agent, transitions_to_d3rlpy_dataset
from tata.testbed.mock_testbed import MockTestbed


def _stratified_split_60_20_20(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Paper-standard 60/20/20 stratified train/val/test split."""
    _X_train, _X_temp, _y_train, _y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_train = np.asarray(_X_train)
    X_temp = np.asarray(_X_temp)
    y_train = np.asarray(_y_train)
    y_temp = np.asarray(_y_temp)

    _X_val, _X_test, _y_val, _y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    X_val = np.asarray(_X_val)
    X_test = np.asarray(_X_test)
    y_val = np.asarray(_y_val)
    y_test = np.asarray(_y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def phase_preliminary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    use_hpo: bool = False,
    device: str = "cpu",
    random_state: int = 42,
    save_dir: str = "data/artifacts",
) -> Encoder:
    """Phase 0: Train contrastive autoencoder (with optional HPO)."""
    print("\n" + "=" * 60)
    print("PHASE 0: Preliminary — Contrastive Autoencoder")
    print("=" * 60)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    encoder_dims = [64, 32, 16]
    latent_dim = 3
    decoder_dims = [16, 32, 64]
    epochs = 250
    batch_size = 128
    learning_rate = 0.001
    lambda_contrastive = 0.1
    margin = 10.0

    if use_hpo:
        print("[Preliminary] Running Optuna HPO...")
        hpo_result = run_hpo(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            n_trials=20,
            study_name="tata_ae_hpo",
            device=device,
            output_dir=str(save_path / "hpo"),
        )
        best_params = hpo_result["best_params"]
        arch = best_params.get("encoder_arch", [64, 32, 16])
        latent_dim = best_params["latent_dim"]
        encoder_dims = [d for d in arch if d != latent_dim]
        decoder_dims = list(reversed(encoder_dims))
        epochs = best_params["epochs"]
        batch_size = best_params["batch_size"]
        learning_rate = best_params["learning_rate"]
        lambda_contrastive = best_params["lambda"]
        margin = best_params["margin"]
        print("[Preliminary] Retraining best model from HPO...")

    model = ContrastiveAutoencoder(
        input_dim=X_train.shape[1],
        encoder_dims=encoder_dims,
        latent_dim=latent_dim,
        decoder_dims=decoder_dims,
    )

    trained, history = train_autoencoder(
        model, X_train, y_train,
        epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate,
        lambda_contrastive=lambda_contrastive,
        margin=margin,
        device=device,
        verbose=True,
    )

    encoder = Encoder(trained, device=device)
    model_config = {
        "input_dim": X_train.shape[1],
        "encoder_dims": encoder_dims,
        "latent_dim": latent_dim,
        "decoder_dims": decoder_dims,
    }
    encoder.save(save_path / "encoder.pt", config=model_config)
    print(f"Encoder saved to {save_path / 'encoder.pt'}")

    return encoder


def phase_assessment(
    encoder: Encoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    save_dir: str = "data/artifacts",
) -> dict:
    """Phase 1: Assess test set quality with D/P/S metrics."""
    print("\n" + "=" * 60)
    print("PHASE 1: Assessment — Diversity / Proximity / Scarcity")
    print("=" * 60)

    train_emb = encoder.encode(X_train)
    test_emb = encoder.encode(X_test)

    best_k, kmeans, metadata = run_clustering_pipeline(
        train_emb, y_train, random_state=random_state
    )

    # Save clustering artifacts for downstream scripts
    save_path = Path(save_dir)
    with open(save_path / "kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    meta_serializable = {
        "n_clusters": metadata["n_clusters"],
        "best_k": metadata.get("best_k", best_k),
        "km_accuracy": metadata["km_accuracy"],
        "silhouette_score": metadata["silhouette_score"],
        "centroids": metadata["centroids"].tolist(),
        "cluster_majority_labels": metadata["cluster_majority_labels"],
    }
    with open(save_path / "cluster_metadata.json", "w") as f:
        json.dump(meta_serializable, f, indent=2)
    print(f"  Clustering artifacts saved to {save_path}")

    results = assess_test_set(
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        train_labels=y_train,
        test_labels=y_test,
        cluster_metadata=metadata,
    )

    print(f"  Diversity (D): {results['global']['diversity']:.4f}")
    print(f"  Proximity (P): {results['global']['proximity']:.4f}")
    print(f"  Scarcity (S):  {results['global']['scarcity']:.4f}")

    return results


def phase_collect_offline_data(
    encoder: Encoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    device: str = "cpu",
    random_state: int = 42,
    save_dir: str = "data/artifacts",
) -> str:
    """Phase 2a: Collect offline RL data with random agent."""
    print("\n" + "=" * 60)
    print("PHASE 2a: Collect Offline RL Data")
    print("=" * 60)

    from tata.embedding.clustering import run_clustering_pipeline

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_emb = encoder.encode(X_train)
    test_emb = encoder.encode(X_test)

    best_k, kmeans, metadata = run_clustering_pipeline(
        train_emb, y_train, random_state=random_state
    )

    testbed = MockTestbed(
        train_features=X_train,  # unscaled
        scaler=scaler,
        seed=random_state,
    )

    env = TataRLEnvironment(
        encoder=encoder,
        cluster_metadata=metadata,
        initial_test_embeddings=test_emb,
        initial_test_labels=y_test,
        train_embeddings=train_emb,
        train_labels=y_train,
        testbed=testbed,
    )

    transitions = collect_transitions_with_random_agent(env, n_episodes=500, seed=random_state)
    dataset = transitions_to_d3rlpy_dataset(transitions)

    dataset_path = str(save_path / "offline_dataset.h5")
    dataset.dump(dataset_path)
    print(f"Offline dataset saved to {dataset_path}")

    return dataset_path


def run_full_pipeline(args) -> None:
    """Run the complete TATA reproduction pipeline."""

    print("=" * 60)
    print("TATA Paper Reproduction Pipeline")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Random State: {args.random_state}")

    # Load dataset
    print("\n[Setup] Loading dataset...")
    X, y = load_dataset(args.dataset, data_dir=args.data_dir)
    print(f"  Loaded: X={X.shape}, classes={len(np.unique(y))}")

    # Paper-standard 60/20/20 split
    X_train, X_val, X_test, y_train, y_val, y_test = _stratified_split_60_20_20(
        X, y, random_state=args.random_state
    )
    print(f"  Split: Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Save splits and scaler for downstream scripts
    artifacts_path = Path(args.artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        artifacts_path / "splits.npz",
        X_train=X_train_s,
        y_train=y_train,
        X_val=X_val_s,
        y_val=y_val,
        X_test=X_test_s,
        y_test=y_test,
    )
    with open(artifacts_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Splits and scaler saved to {artifacts_path}")

    # Phase 0: Preliminary
    encoder = phase_preliminary(
        X_train_s, y_train,
        X_val=X_val_s,
        use_hpo=args.use_hpo,
        device=args.device,
        random_state=args.random_state,
        save_dir=args.artifacts_dir,
    )

    # Phase 1: Assessment (also saves clustering artifacts)
    phase_assessment(
        encoder=encoder,
        X_train=X_train_s,
        y_train=y_train,
        X_test=X_test_s,
        y_test=y_test,
        random_state=args.random_state,
        save_dir=args.artifacts_dir,
    )

    # Section 4.2: Ablation
    if args.run_ablation:
        print("\n" + "=" * 60)
        print("SECTION 4.2: Ablation Study")
        print("=" * 60)
        run_ablation(
            X=X, y=y,
            n_splits=args.ablation_splits,
            epochs=args.ablation_epochs,
            device=args.device,
            random_state=args.random_state,
            save_dir=f"{args.artifacts_dir}/ablation",
            verbose=True,
        )

    # Section 4.3: Correlation
    if args.run_correlation:
        print("\n" + "=" * 60)
        print("SECTION 4.3: Correlation Analysis")
        print("=" * 60)
        run_correlation_study(
            X=X, y=y,
            encoder_callable=lambda x: encoder.encode(x),
            random_state=args.random_state,
            save_dir=f"{args.artifacts_dir}/correlation",
            verbose=True,
        )

    # Phase 2: Offline RL + Appendix B HPO
    if args.run_rl:
        print("\n" + "=" * 60)
        print("PHASE 2 & APPENDIX B: RL + HPO")
        print("=" * 60)

        # Collect offline data
        offline_path = phase_collect_offline_data(
            encoder, X_train_s, y_train, X_test_s, y_test, scaler,
            device=args.device, random_state=args.random_state,
            save_dir=args.artifacts_dir,
        )

        # Load offline dataset
        import d3rlpy
        offline_dataset = d3rlpy.dataset.ReplayBuffer.load(offline_path)

        # Run Split-Select-Retrain HPO
        ssr_results = run_split_select_retrain(
            offline_dataset=offline_dataset,
            K=args.ssr_k,
            fqe_epochs=args.fqe_epochs,
            random_state=args.random_state,
            device=args.device,
            save_dir=f"{args.artifacts_dir}/rl_hpo",
            verbose=True,
        )

    # Section 4.4: Augmentation Evaluation
    if args.run_augmentation:
        print("\n" + "=" * 60)
        print("SECTION 4.4: Augmentation Evaluation")
        print("=" * 60)
        print("NOTE: Requires trained RL agents in d3rlpy_logs/")
        print("Run scripts/experiments/run_augmentation_eval.py separately.")

    # Section 4.5: Comparative Analysis
    if args.run_comparative:
        print("\n" + "=" * 60)
        print("SECTION 4.5: Multi-Dataset Comparative Analysis")
        print("=" * 60)
        run_comparative_analysis(
            datasets=args.comparative_datasets,
            data_dir=args.data_dir,
            ae_epochs=args.comparative_epochs,
            n_splits=args.comparative_splits,
            device=args.device,
            random_state=args.random_state,
            save_dir=f"{args.artifacts_dir}/comparative",
            verbose=True,
        )

    print("\n" + "=" * 60)
    print("REPRODUCTION PIPELINE COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TATA Paper Reproduction Orchestrator")

    # General
    parser.add_argument("--dataset", type=str, default="ids2018",
                        help="Primary dataset for reproduction (ids2017/ids2018)")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--artifacts-dir", type=str, default="data/artifacts")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-hpo", action="store_true",
                        help="Use Optuna HPO for autoencoder")

    # Phase toggles (BooleanOptionalAction allows --no-run-* to disable)
    parser.add_argument("--run-ablation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-correlation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-rl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-augmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-comparative", action=argparse.BooleanOptionalAction, default=False)

    # Hyperparameters
    parser.add_argument("--ablation-splits", type=int, default=10)
    parser.add_argument("--ablation-epochs", type=int, default=100)
    parser.add_argument("--correlation-variants", type=int, default=20)
    parser.add_argument("--ssr-k", type=int, default=20,
                        help="K folds for Split-Select-Retrain")
    parser.add_argument("--fqe-epochs", type=int, default=100)
    parser.add_argument("--comparative-datasets", nargs="+", default=None)
    parser.add_argument("--comparative-epochs", type=int, default=100)
    parser.add_argument("--comparative-splits", type=int, default=10)

    args = parser.parse_args()

    run_full_pipeline(args)


if __name__ == "__main__":
    main()
