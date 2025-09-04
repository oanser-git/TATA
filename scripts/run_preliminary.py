"""
End-to-end script for the Preliminary Phase:
1. Train contrastive autoencoder (with optional Optuna HPO)
2. Run silhouette-guided k-means clustering
3. Save encoder, clusters, and metadata
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
from tata.models.hpo import run_hpo
from tata.models.training import train_autoencoder


def run_preliminary_phase(
    data_config_path: str = "configs/data.yaml",
    ae_config_path: str = "configs/autoencoder.yaml",
    dataset_name: str = "ids2017",
    data_dir: str = "data",
    use_hpo: bool = False,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run the full preliminary phase.
    
    Args:
        data_config_path: Path to data config YAML.
        ae_config_path: Path to autoencoder config YAML.
        use_hpo: Whether to run Optuna HPO or use fixed config.
        device: 'cpu' or 'cuda'.
        seed: Random seed for data splitting.
    
    Returns:
        Dict with paths to saved artifacts.
    """
    # Load configs
    with open(data_config_path) as f:
        data_cfg = yaml.safe_load(f)
    with open(ae_config_path) as f:
        ae_cfg = yaml.safe_load(f)
    
    # Step 1: Load data
    print("[Preliminary] Loading dataset...")
    X, y = load_dataset(dataset_name, data_dir=data_dir)
    print(f"  Loaded dataset: {dataset_name} | X={X.shape}, classes={len(np.unique(y))}")

    # Step 2: Split
    print("[Preliminary] Splitting dataset...")
    split_cfg = data_cfg["data"]["split"]
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(
        X, y,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        random_state=seed,
    )
    
    # Step 3: Preprocess
    print("[Preliminary] Preprocessing...")
    preproc_cfg = data_cfg["data"]["preprocessing"]
    artifacts_dir = Path(data_cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    processed = preprocess_pipeline(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        scaler_type=preproc_cfg["scaler_type"],
        scaler_path=artifacts_dir / "scaler.pkl",
    )
    
    X_train_s = processed["X_train"].values
    y_train_enc = processed["y_train"]
    X_val_s = processed["X_val"].values if processed["X_val"] is not None else None
    
    # Save label encoder
    with open(artifacts_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(processed["label_encoder"], f)
    
    # Save splits
    np.savez(
        artifacts_dir / "splits.npz",
        X_train=X_train_s,
        y_train=y_train_enc,
        X_val=X_val_s if X_val_s is not None else np.array([]),
        y_val=processed["y_val"] if processed["y_val"] is not None else np.array([]),
        X_test=processed["X_test"].values if processed["X_test"] is not None else np.array([]),
        y_test=processed["y_test"] if processed["y_test"] is not None else np.array([]),
    )
    
    # Step 4: Train autoencoder (HPO or fixed)
    ae_train_cfg = ae_cfg["autoencoder"]["training"]
    ae_contrastive_cfg = ae_cfg["autoencoder"]["contrastive"]
    
    # Initialize architecture variables
    arch = ae_cfg["autoencoder"]["architecture"]["encoder_dims"]
    latent_dim = ae_cfg["autoencoder"]["architecture"]["latent_dim"]
    decoder_dims = ae_cfg["autoencoder"]["architecture"].get("decoder_dims", list(reversed(arch)))
    best_params = None
    
    if use_hpo:
        print("[Preliminary] Running Optuna HPO...")
        hpo_cfg = ae_cfg["autoencoder"]["hpo"]
        hpo_result = run_hpo(
            X_train=X_train_s,
            y_train=y_train_enc,
            X_val=X_val_s,
            n_trials=hpo_cfg["n_trials"],
            study_name=hpo_cfg["study_name"],
            storage=hpo_cfg.get("storage"),
            config=hpo_cfg["search_space"],
            device=device,
            output_dir=str(artifacts_dir / "hpo"),
        )
        best_params = hpo_result["best_params"]
        
        # Reconstruct architecture from best params
        arch = best_params.get("encoder_arch", [64, 32, 16])
        latent_dim = best_params["latent_dim"]
    
    encoder_dims = [d for d in arch if d != latent_dim]
    decoder_dims = list(reversed(encoder_dims))
    
    model = ContrastiveAutoencoder(
        input_dim=X_train_s.shape[1],
        encoder_dims=encoder_dims,
        latent_dim=latent_dim,
        decoder_dims=decoder_dims,
    )
    
    if use_hpo:
        assert best_params is not None
        # Retrain best model on full data
        print("[Preliminary] Retraining best model from HPO...")
        trained_model, history = train_autoencoder(
            model=model,
            X_train=X_train_s,
            y_train=y_train_enc,
            X_val=X_val_s,
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            learning_rate=best_params["learning_rate"],
            margin=best_params["margin"],
            lambda_contrastive=best_params["lambda"],
            device=device,
            checkpoint_dir=str(artifacts_dir),
        )
    else:
        print("[Preliminary] Training with fixed config...")
        trained_model, history = train_autoencoder(
            model=model,
            X_train=X_train_s,
            y_train=y_train_enc,
            X_val=X_val_s,
            epochs=ae_train_cfg["epochs"],
            batch_size=ae_train_cfg["batch_size"],
            learning_rate=ae_train_cfg["learning_rate"],
            margin=ae_contrastive_cfg["margin"],
            lambda_contrastive=ae_contrastive_cfg["lambda"],
            device=device,
            checkpoint_dir=str(artifacts_dir),
        )
    
    # Save trained model WITH embedded config (self-contained)
    model_path = artifacts_dir / "encoder.pt"
    encoder = Encoder(trained_model, device=device)
    model_config = {
        "input_dim": X_train_s.shape[1],
        "encoder_dims": encoder_dims,
        "latent_dim": latent_dim,
        "decoder_dims": decoder_dims,
    }
    encoder.save(model_path, config=model_config)
    print(f"[Preliminary] Model saved to {model_path} (with embedded config)")
    
    # Also save model_config.json for backward compatibility
    with open(artifacts_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Step 5: Clustering
    print("[Preliminary] Running clustering...")
    cluster_cfg = ae_cfg["autoencoder"]["clustering"]
    train_embeddings = encoder.encode(X_train_s)
    
    best_k, kmeans, metadata = run_clustering_pipeline(
        embeddings=train_embeddings,
        true_labels=y_train_enc,
        k_range=cluster_cfg["k_range"],
        random_state=cluster_cfg["random_state"],
        n_init=cluster_cfg["n_init"],
    )
    
    # Save clustering artifacts
    with open(artifacts_dir / "kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    
    with open(artifacts_dir / "cluster_metadata.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        meta_serializable = {
            "n_clusters": metadata["n_clusters"],
            "best_k": metadata["best_k"],
            "km_accuracy": metadata["km_accuracy"],
            "silhouette_score": metadata["silhouette_score"],
            "centroids": metadata["centroids"].tolist(),
            "cluster_majority_labels": metadata["cluster_majority_labels"],
        }
        json.dump(meta_serializable, f, indent=2)
    
    print("[Preliminary] Preliminary phase complete!")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Best k: {best_k}")
    print(f"  KM Accuracy: {metadata['km_accuracy']:.4f}")
    print(f"  Silhouette: {metadata['silhouette_score']:.4f}")
    
    return {
        "artifacts_dir": str(artifacts_dir),
        "model_path": str(model_path),
        "latent_dim": latent_dim,
        "best_k": best_k,
        "km_accuracy": metadata["km_accuracy"],
        "silhouette": metadata["silhouette_score"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TATA Preliminary Phase")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--ae-config", default="configs/autoencoder.yaml")
    parser.add_argument("--dataset", default="ids2017",
                        help="Dataset name (e.g., ids2017, ids2018). Default: ids2017.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--hpo", action="store_true", help="Run Optuna HPO")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_preliminary_phase(
        data_config_path=args.data_config,
        ae_config_path=args.ae_config,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        use_hpo=args.hpo,
        device=args.device,
        seed=args.seed,
    )
