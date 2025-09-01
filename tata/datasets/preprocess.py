"""
Data preprocessing utilities for NIDS datasets.
Handles scaling, missing values, and feature selection.
"""

import pickle
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def clean_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    drop_inf: bool = True,
    drop_na: bool = True,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean a NIDS dataset by handling invalid values.
    
    Args:
        X: Feature DataFrame.
        y: Label Series.
        drop_inf: Remove rows with infinite values.
        drop_na: Remove rows with NaN values.
        drop_duplicates: Remove exact duplicate rows.
    
    Returns:
        Cleaned X, y.
    """
    # Combine for joint cleaning
    df = X.copy()
    df["__label__"] = y.values
    
    initial_shape = df.shape[0]
    
    if drop_inf:
        df = df.replace([np.inf, -np.inf], np.nan)
    
    if drop_na:
        df = df.dropna()
    
    if drop_duplicates:
        df = df.drop_duplicates()
    
    cleaned_shape = df.shape[0]
    if cleaned_shape < initial_shape:
        print(f"[clean_dataset] Dropped {initial_shape - cleaned_shape} rows ({initial_shape} -> {cleaned_shape})")
    
    y_clean = pd.Series(df["__label__"].values, name=y.name if y.name else "label")
    X_clean = df.drop(columns=["__label__"])
    
    return X_clean, y_clean


def encode_labels(
    y: pd.Series,
    label_encoder: Optional[LabelEncoder] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode string labels to integers.
    
    Args:
        y: Label Series.
        label_encoder: Existing encoder (for re-use).
        fit: Whether to fit a new encoder.
    
    Returns:
        Encoded labels, encoder.
    """
    if label_encoder is None and fit:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
    elif label_encoder is not None:
        le = label_encoder
        y_enc = le.transform(y)
    else:
        raise ValueError("Must provide label_encoder or set fit=True")
    
    return np.asarray(y_enc), le


def scale_features(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    scaler_type: str = "standard",
    scaler_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], object]:
    """
    Scale features using a scaler fitted ONLY on training data.
    
    Args:
        X_train: Training features.
        X_val: Validation features (optional).
        X_test: Test features (optional).
        scaler_type: 'standard' or 'minmax'.
        scaler_path: Path to save fitted scaler.
    
    Returns:
        Scaled X_train, X_val, X_test, scaler object.
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    
    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    
    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    if scaler_path is not None:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[scale_features] Scaler saved to {scaler_path}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocess_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    scaler_type: str = "standard",
    scaler_path: Optional[Union[str, Path]] = None,
    drop_inf: bool = True,
    drop_na: bool = True,
    drop_duplicates: bool = True,
) -> dict[str, Any]:
    """
    Full preprocessing pipeline: clean, encode labels, scale.
    
    Returns:
        dict with keys: X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, scaler.
    """
    # Clean each split independently (or could combine and re-split; 
    # cleaning per-split is safer to avoid data leakage)
    X_train, y_train = clean_dataset(X_train, y_train, drop_inf, drop_na, drop_duplicates)
    
    if X_val is not None and y_val is not None:
        X_val, y_val = clean_dataset(X_val, y_val, drop_inf, drop_na, drop_duplicates)
    
    if X_test is not None and y_test is not None:
        X_test, y_test = clean_dataset(X_test, y_test, drop_inf, drop_na, drop_duplicates)
    
    # Encode labels (fit on train only)
    y_train_enc, le = encode_labels(y_train, fit=True)
    
    y_val_enc = None
    if y_val is not None:
        y_val_enc, _ = encode_labels(y_val, label_encoder=le, fit=False)
    
    y_test_enc = None
    if y_test is not None:
        y_test_enc, _ = encode_labels(y_test, label_encoder=le, fit=False)
    
    # Scale features (fit on train only)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(
        X_train, X_val, X_test, scaler_type, scaler_path
    )
    
    return {
        "X_train": X_train_s,
        "y_train": y_train_enc,
        "X_val": X_val_s,
        "y_val": y_val_enc,
        "X_test": X_test_s,
        "y_test": y_test_enc,
        "label_encoder": le,
        "scaler": scaler,
        "feature_names": list(X_train.columns),
    }
