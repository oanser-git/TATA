"""
Collect offline RL data using a random agent on the mock testbed.
This pre-collects transition tuples for offline RL training (CQL / TD3+BC).

Artifact requirements (in data/artifacts/):
  - splits.npz
  - scaler.pkl
  - encoder.pt          (with embedded config)
  - kmeans.pkl
  - cluster_metadata.json
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tata.datasets.loaders import load_dataset
from tata.datasets.preprocess import preprocess_pipeline
from tata.datasets.splits import stratified_split
from tata.embedding.clustering import run_clustering_pipeline
from tata.embedding.encoder import Encoder
from tata.models.autoencoder import ContrastiveAutoencoder
from tata.models.training import train_autoencoder
from tata.rl.environment import TataRLEnvironment
from tata.rl.offline_dataset import collect_transitions_with_random_agent, save_transitions
from tata.testbed.mock_testbed import MockTestbed


def _check_artifacts(artifacts_dir: Path):
    """Verify all required artifacts exist."""
    required = {
        "splits.npz": "Run scripts/run_preliminary.py first.",
        "scaler.pkl": "Run scripts/run_preliminary.py first.",
        "encoder.pt": "Run scripts/run_preliminary.py first.",
        "kmeans.pkl": "Run scripts/run_preliminary.py first.",
        "cluster_metadata.json": "Run scripts/run_preliminary.py first.",
    }
    missing = []
    for fname, hint in required.items():
        if not (artifacts_dir / fname).exists():
            missing.append(f"  - {fname}: {hint}")
    if missing:
        raise FileNotFoundError(
            f"Missing required artifacts in {artifacts_dir}:\n" + "\n".join(missing)
        )


def collect_offline_data(
    data_config_path: str = "configs/data.yaml",
    ae_config_path: str = "configs/autoencoder.yaml",
    rl_config_path: str = "configs/rl.yaml",
    use_pretrained: bool = True,
    seed: int = 42,
):
    """
    Main entry point: load artifacts, build env, collect random transitions.
    
    Args:
        use_pretrained: If True (default), load saved artifacts.
                        If False, recompute everything from scratch (slow).
    """
    # Load configs
    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)
    with open(ae_config_path) as f:
        ae_cfg = yaml.safe_load(f)
    with open(rl_config_path) as f:
        rl_cfg = yaml.safe_load(f)
    
    artifacts_dir = Path(data_cfg["paths"]["artifacts_dir"])
    
    if use_pretrained:
        # Validate and load existing artifacts
        _check_artifacts(artifacts_dir)
        print("[Collect] Loading pretrained artifacts...")
        
        # Load splits
        splits = np.load(artifacts_dir / "splits.npz")
        X_train = splits["X_train"]
        y_train_enc = splits["y_train"]
        X_test = splits["X_test"]
        y_test_enc = splits["y_test"]
        
        # Load scaler
        with open(artifacts_dir / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        # Load encoder (new self-contained format)
        encoder = Encoder.from_checkpoint(artifacts_dir / "encoder.pt")
        
        # Encode datasets
        train_embeddings = encoder.encode(X_train)
        test_embeddings = encoder.encode(X_test)
        
        # Load cluster metadata
        with open(artifacts_dir / "cluster_metadata.json") as f:
            meta_json = json.load(f)
        with open(artifacts_dir / "kmeans.pkl", "rb") as f:
            kmeans = pickle.load(f)
        
        cluster_metadata: dict[str, Any] = {
            "n_clusters": meta_json["n_clusters"],
            "centroids": np.array(meta_json["centroids"]),
            "cluster_majority_labels": meta_json["cluster_majority_labels"],
            "cluster_member_indices": {},
            "km_accuracy": meta_json["km_accuracy"],
            "silhouette_score": meta_json["silhouette_score"],
        }
        cluster_labels = kmeans.labels_
        n_clusters = int(cluster_metadata["n_clusters"])
        for c in range(n_clusters):
            cluster_metadata["cluster_member_indices"][c] = np.where(cluster_labels == c)[0].tolist()
    
    else:
        # Fallback: train from scratch using test dataset
        print("[Collect] No pretrained artifacts found. Training from scratch...")
        X, y = load_dataset("ids2017", data_dir="data")
        X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(
            X, y,
            train_ratio=data_cfg["data"]["split"]["train_ratio"],
            val_ratio=data_cfg["data"]["split"]["val_ratio"],
            test_ratio=data_cfg["data"]["split"]["test_ratio"],
            random_state=seed,
        )
        
        processed = preprocess_pipeline(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            scaler_type=data_cfg["data"]["preprocessing"]["scaler_type"],
            scaler_path=artifacts_dir / "scaler.pkl",
        )
        
        X_train_s = processed["X_train"].values
        y_train_enc = processed["y_train"]
        X_test_s = processed["X_test"].values
        y_test_enc = processed["y_test"]
        scaler = processed["scaler"]
        
        # Train encoder
        arch = ae_cfg["autoencoder"]["architecture"]["encoder_dims"]
        latent_dim = ae_cfg["autoencoder"]["architecture"]["latent_dim"]
        model = ContrastiveAutoencoder(
            input_dim=X_train_s.shape[1],
            encoder_dims=arch,
            latent_dim=latent_dim,
            decoder_dims=list(reversed(arch)),
        )
        trained_model, _ = train_autoencoder(
            model=model,
            X_train=X_train_s,
            y_train=y_train_enc,
            epochs=ae_cfg["autoencoder"]["training"]["epochs"],
            batch_size=ae_cfg["autoencoder"]["training"]["batch_size"],
            learning_rate=ae_cfg["autoencoder"]["training"]["learning_rate"],
            margin=ae_cfg["autoencoder"]["contrastive"]["margin"],
            lambda_contrastive=ae_cfg["autoencoder"]["contrastive"]["lambda"],
            verbose=False,
        )
        encoder = Encoder(trained_model)
        
        # Encode
        train_embeddings = encoder.encode(X_train_s)
        test_embeddings = encoder.encode(X_test_s)
        
        # Cluster
        best_k, kmeans, metadata = run_clustering_pipeline(
            embeddings=train_embeddings,
            true_labels=y_train_enc,
            k_range=ae_cfg["autoencoder"]["clustering"]["k_range"],
        )
        cluster_metadata = metadata
    
    # Create mock testbed
    print("[Collect] Creating mock testbed...")
    testbed = MockTestbed(
        train_features=np.asarray(X_train),  # unscaled for sampling base flows
        scaler=scaler,
        label=0,  # benign
        seed=seed,
    )
    
    # Create RL environment
    print("[Collect] Building RL environment...")
    env = TataRLEnvironment(
        encoder=encoder,
        cluster_metadata=cluster_metadata,
        initial_test_embeddings=test_embeddings,
        initial_test_labels=y_test_enc,
        train_embeddings=train_embeddings,
        train_labels=y_train_enc,
        testbed=testbed,
        reward_weights=rl_cfg["rl"]["environment"]["reward_weights"],
        max_steps_per_episode=rl_cfg["rl"]["environment"]["max_steps_per_episode"],
        target_reward=rl_cfg["rl"]["environment"]["target_reward"],
    )
    
    # Collect transitions with random agent
    print("[Collect] Starting random agent data collection...")
    n_episodes = rl_cfg["rl"]["offline_data_collection"]["n_episodes"]
    transitions = collect_transitions_with_random_agent(
        env=env,
        n_episodes=n_episodes,
        seed=rl_cfg["rl"]["offline_data_collection"]["random_seed"],
    )
    
    # Save
    output_path = rl_cfg["rl"]["offline_data_collection"]["output_path"]
    save_transitions(transitions, output_path)
    
    print("[Collect] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect offline RL data with random agent")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--ae-config", default="configs/autoencoder.yaml")
    parser.add_argument("--rl-config", default="configs/rl.yaml")
    parser.add_argument("--no-pretrained", action="store_true", help="Train encoder from scratch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    collect_offline_data(
        data_config_path=args.data_config,
        ae_config_path=args.ae_config,
        rl_config_path=args.rl_config,
        use_pretrained=not args.no_pretrained,
        seed=args.seed,
    )
