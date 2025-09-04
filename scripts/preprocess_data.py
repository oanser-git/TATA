"""
Preprocess data and save splits.
Standalone script for data preparation.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from tata.datasets.loaders import load_dataset
from tata.datasets.preprocess import preprocess_pipeline
from tata.datasets.splits import stratified_split


def main(
    config_path: str = "configs/data.yaml",
    dataset_name: str = "ids2017",
    data_dir: str = "data",
    seed: int = 42,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load data
    X, y = load_dataset(dataset_name, data_dir=data_dir)
    print(f"  Loaded dataset: {dataset_name} | X={X.shape}, classes={len(np.unique(y))}")

    # Split
    split_cfg = cfg["data"]["split"]
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(
        X, y,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        random_state=seed,
    )
    
    # Preprocess
    preproc_cfg = cfg["data"]["preprocessing"]
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    processed = preprocess_pipeline(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        scaler_type=preproc_cfg["scaler_type"],
        scaler_path=processed_dir / "scaler.pkl",
    )
    
    # Save
    import pickle
    with open(processed_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(processed["label_encoder"], f)
    
    np.savez(
        processed_dir / "splits.npz",
        X_train=processed["X_train"].values,
        y_train=processed["y_train"],
        X_val=processed["X_val"].values if processed["X_val"] is not None else np.array([]),
        y_val=processed["y_val"] if processed["y_val"] is not None else np.array([]),
        X_test=processed["X_test"].values if processed["X_test"] is not None else np.array([]),
        y_test=processed["y_test"] if processed["y_test"] is not None else np.array([]),
    )
    
    print(f"[Preprocess] Data saved to {processed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset and save splits")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--dataset", default="ids2017",
                        help="Dataset name (e.g., ids2017, ids2018). Default: ids2017.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        config_path=args.config,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        seed=args.seed,
    )
