"""
Stratified train/val/test splitting for NIDS datasets.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Create stratified train/val/test splits.
    
    Args:
        X: Feature DataFrame.
        y: Label Series.
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.
        random_state: Random seed.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    test_relative_ratio = test_ratio / val_test_ratio
    
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y,
        train_size=train_ratio,
        stratify=y,
        random_state=random_state,
    )
    
    # Second split: val vs test
    if val_ratio > 0 and test_ratio > 0:
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test,
            test_size=test_relative_ratio,
            stratify=y_val_test,
            random_state=random_state,
        )
    elif val_ratio > 0:
        X_val, y_val = X_val_test, y_val_test
        X_test, y_test = pd.DataFrame(), pd.Series(dtype=y.dtype)
    else:
        X_val, y_val = pd.DataFrame(), pd.Series(dtype=y.dtype)
        X_test, y_test = X_val_test, y_val_test
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_multiple_splits(
    X: pd.DataFrame,
    y: pd.Series,
    seeds: List[int],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    output_dir: Optional[str] = None,
) -> List[dict]:
    """
    Generate multiple stratified splits with different seeds.
    
    Args:
        X, y: Data.
        seeds: List of random seeds.
        train_ratio, val_ratio, test_ratio: Split ratios.
        output_dir: If provided, save each split as parquet.
    
    Returns:
        List of dicts, each containing the split data.
    """
    splits = []
    for seed in seeds:
        X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(
            X, y, train_ratio, val_ratio, test_ratio, random_state=seed
        )
        
        split_dict = {
            "seed": seed,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }
        
        if output_dir is not None:
            out = Path(output_dir) / f"seed_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            X_train.to_parquet(out / "X_train.parquet")
            y_train.to_frame("label").to_parquet(out / "y_train.parquet")
            if len(X_val) > 0:
                X_val.to_parquet(out / "X_val.parquet")
                y_val.to_frame("label").to_parquet(out / "y_val.parquet")
            if len(X_test) > 0:
                X_test.to_parquet(out / "X_test.parquet")
                y_test.to_frame("label").to_parquet(out / "y_test.parquet")
        
        splits.append(split_dict)
    
    return splits
