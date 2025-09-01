"""
Generic dataset loaders for TATA.

This module provides loaders for benchmark datasets used in the paper.
Datasets must be downloaded by the user and placed in the data/ directory.

The loaders are intentionally generic: they load CSV files, separate the label
column, drop non-numeric features, and return numpy arrays. Any dataset-specific
preprocessing (categorical encoding, specific column dropping, etc.) is the
responsibility of the user.

Supported datasets:
  - CIC-IDS2017 / IDS2018 (refined versions)
  - NSL-KDD, UNSW-NB15, Bot-IoT, ToN-IoT
  - CIC-DDoS2019, CTU-13
  - ISCX-IDS2012, ISCX-Tor, VPN-NonVPN, CIC-UNSW
  - MNIST, CIFAR-10 (cross-domain comparison)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder


def _clean_df(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generic cleaning: drop inf/nan, separate label, keep only numeric columns.

    Args:
        df: Raw DataFrame.
        label_col: Name of the label column.

    Returns:
        X: DataFrame of numeric features only.
        y: Series of labels.
    """
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. Available columns: {list(df.columns)}"
        )

    y = df[label_col].copy()
    X = df.drop(columns=[label_col])

    # Keep only numeric columns — user must encode categoricals themselves
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols]

    if X.shape[1] == 0:
        raise ValueError(
            "No numeric feature columns remain after dropping non-numeric columns. "
            "If your dataset has categorical features, encode them to numeric before "
            "passing to TATA, or pre-process the CSVs yourself."
        )

    return X, y


def load_csv_directory(
    data_dir: str,
    label_col: str = "label",
    file_pattern: str = "*.csv",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generic loader for a directory of CSV files.

    Args:
        data_dir: Directory containing CSV files.
        label_col: Name of the label column (default "label").
                   Change this if your dataset uses a different name
                   (e.g., "Label", "class", "attack_cat", etc.).
        file_pattern: Glob pattern for files (default "*.csv").

    Returns:
        X: DataFrame of numeric features.
        y: Series of labels.
    """
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")

    csv_files = sorted(path.glob(file_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in {path}")

    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    return _clean_df(df, label_col)


def load_csv_file(
    file_path: str,
    label_col: str = "label",
    header: str | int | None = "infer",
    names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generic loader for a single CSV file.

    Args:
        file_path: Path to CSV file.
        label_col: Name of the label column.
        header: pandas read_csv header argument (default "infer").
                Use None if the CSV has no header row.
        names: Optional list of column names to assign.

    Returns:
        X: DataFrame of numeric features.
        y: Series of labels.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    kwargs: dict = {"low_memory": False}
    if header is not None:
        kwargs["header"] = header
    if names is not None:
        kwargs["names"] = names

    df = pd.read_csv(path, **kwargs)
    return _clean_df(df, label_col)


# ---------------------------------------------------------------------------
# Convenience wrappers (thin, no hardcoded assumptions beyond defaults)
# ---------------------------------------------------------------------------


def load_ids2017(
    data_dir: str = "data/ids2017",
    label_col: str = "Label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load IDS2017 CSV files. Default label column: 'Label'."""
    return load_csv_directory(data_dir, label_col=label_col)


def load_ids2018(
    data_dir: str = "data/ids2018",
    label_col: str = "Label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load IDS2018 CSV files. Default label column: 'Label'."""
    return load_csv_directory(data_dir, label_col=label_col)


def load_nsl_kdd(
    data_dir: str = "data/nsl-kdd",
    train_file: str = "KDDTrain+.csv",
    label_col: str = "label",
    has_header: bool = False,
    n_features: int = 41,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load NSL-KDD training CSV.

    Defaults assume the standard NSL-KDD format (no header, 41 features
    + label + difficulty = 43 columns). Adjust parameters if your files differ.
    """
    path = Path(data_dir) / train_file
    if not path.exists():
        raise FileNotFoundError(f"NSL-KDD file not found: {path}")

    if has_header:
        df = pd.read_csv(path, low_memory=False)
    else:
        names = [f"f{i}" for i in range(n_features)] + [label_col, "difficulty"]
        df = pd.read_csv(path, names=names, low_memory=False)

    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])

    return _clean_df(df, label_col)


def load_unsw_nb15(
    data_dir: str = "data/unsw-nb15",
    train_file: str = "UNSW_NB15_training-set.csv",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load UNSW-NB15 training CSV. Default label column: 'label'."""
    return load_csv_file(Path(data_dir) / train_file, label_col=label_col)


def load_bot_iot(
    data_dir: str = "data/bot-iot",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load Bot-IoT CSV files. Default label column: 'label'."""
    return load_csv_directory(data_dir, label_col=label_col)


def load_ton_iot(
    data_dir: str = "data/ton-iot",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load ToN-IoT CSV files. Default label column: 'label'."""
    return load_csv_directory(data_dir, label_col=label_col)


def load_cic_ddos2019(
    data_dir: str = "data/cic-ddos2019",
    label_col: str = " Label",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load CIC-DDoS2019 CSV files.
    Default label column: ' Label' (note the leading space in the original
    CICFlowMeter export). Change this if your files use a different name.
    """
    return load_csv_directory(data_dir, label_col=label_col)


def load_ctu13(
    data_dir: str = "data/ctu-13",
    label_col: str = "Label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load CTU-13 CSV files. Default label column: 'Label'."""
    return load_csv_directory(data_dir, label_col=label_col)


def load_iscx_ids2012(
    data_dir: str = "data/iscx-ids2012",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load ISCX-IDS2012 CSV files. Default label column: 'label'."""
    return load_csv_directory(data_dir, label_col=label_col)


def load_iscx_tor(
    data_dir: str = "data/iscx-tor",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load ISCX-Tor CSV files. Default label column: 'label'."""
    return load_csv_directory(data_dir, label_col=label_col)


def load_vpn_nonvpn(
    data_dir: str = "data/vpn-nonvpn",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load VPN-NonVPN CSV files. Default label column: 'label'."""
    return load_csv_directory(data_dir, label_col=label_col)


def load_cic_unsw(
    data_dir: str = "data/cic-unsw",
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load CIC-UNSW CSV files. Default label column: 'label'."""
    return load_csv_directory(data_dir, label_col=label_col)


# ---------------------------------------------------------------------------
# Image datasets (cross-domain comparison)
# ---------------------------------------------------------------------------


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST via sklearn/openml."""
    try:
        mnist = fetch_openml("mnist_784", version=1, parser="auto")
        X = np.array(mnist.data, dtype=np.float32)
        y = np.array(mnist.target, dtype=int)
        return X, y
    except Exception as e:
        raise RuntimeError(f"Failed to load MNIST: {e}")


def load_cifar10() -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 via torchvision."""
    try:
        import torchvision

        dataset = torchvision.datasets.CIFAR10(
            root="data/cifar10", download=True, train=True
        )
        X = (
            np.array(dataset.data, dtype=np.float32).reshape(len(dataset.data), -1)
            / 255.0
        )
        y = np.array(dataset.targets, dtype=int)
        return X, y
    except Exception as e:
        raise RuntimeError(f"Failed to load CIFAR-10: {e}")


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def load_dataset(
    name: str,
    data_dir: str = "data",
    label_col: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unified dataset loader.

    Args:
        name: Dataset name (e.g., 'ids2017', 'nsl-kdd', 'mnist').
        data_dir: Root data directory.
        label_col: Optional override for the label column name.
                   If None, uses the loader's documented default.

    Returns:
        X: Feature matrix (n_samples, n_features).
        y: Integer labels (n_samples,).
    """
    name = name.lower().strip()

    loaders: dict = {
        "ids2017": load_ids2017,
        "ids2018": load_ids2018,
        "nsl-kdd": load_nsl_kdd,
        "unsw-nb15": load_unsw_nb15,
        "bot-iot": load_bot_iot,
        "ton-iot": load_ton_iot,
        "cic-ddos2019": load_cic_ddos2019,
        "ctu-13": load_ctu13,
        "iscx-ids2012": load_iscx_ids2012,
        "iscx-tor": load_iscx_tor,
        "vpn-nonvpn": load_vpn_nonvpn,
        "cic-unsw": load_cic_unsw,
        "mnist": load_mnist,
        "cifar10": load_cifar10,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Supported: {list(loaders.keys())}")

    if name in ("mnist", "cifar10"):
        X, y = loaders[name]()
    else:
        data_path = Path(data_dir) / name.replace("-", "_")
        kwargs: dict = {}
        if label_col is not None:
            kwargs["label_col"] = label_col
        X_df, y_series = loaders[name](str(data_path), **kwargs)
        X = X_df.values.astype(np.float32)
        le = LabelEncoder()
        y = le.fit_transform(y_series.astype(str))

    return X, y
