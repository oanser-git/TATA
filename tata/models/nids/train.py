"""
NIDS training with grid search from YAML configuration.
Supports RF, SVM, DNN with cross-validation.
"""

import itertools
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score

from tata.models.nids.random_forest import RandomForestNIDS
from tata.models.nids.svm import SVMNIDS
from tata.models.nids.dnn import DNNNIDS
from tata.models.nids.base import AbstractNIDS


def load_config(config_path: str = "configs/nids.yaml") -> dict[str, Any]:
    """Load NIDS configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def instantiate_model(model_type: str, params: dict, random_state: int = 42) -> AbstractNIDS:
    """
    Instantiate a NIDS model by type with given parameters.
    
    Args:
        model_type: 'rf', 'svm', or 'dnn'.
        params: Hyperparameter dict for the model.
        random_state: Random seed.
    
    Returns:
        Instantiated AbstractNIDS subclass.
    """
    if model_type == "rf":
        return RandomForestNIDS(config=params, random_state=random_state)
    elif model_type == "svm":
        return SVMNIDS(config=params, random_state=random_state)
    elif model_type == "dnn":
        return DNNNIDS(config=params, random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_with_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    param_grid: Dict[str, list],
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: int = 1,
) -> Tuple[AbstractNIDS, Dict[str, Any]]:
    """
    Train a NIDS model using grid search with cross-validation.
    
    For DNN, since sklearn GridSearchCV doesn't support our PyTorch wrapper
    out of the box, we do a manual grid search.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        model_type: 'rf', 'svm', or 'dnn'.
        param_grid: Parameter grid for grid search.
        cv_folds: Number of CV folds.
        random_state: Random seed.
        verbose: Verbosity level.
    
    Returns:
        Best model, results dict.
    """
    if model_type in ("rf", "svm"):
        return _sklearn_grid_search(
            X_train, y_train, model_type, param_grid, cv_folds, random_state, verbose
        )
    elif model_type == "dnn":
        return _dnn_grid_search(
            X_train, y_train, param_grid, cv_folds, random_state, verbose
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _sklearn_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    param_grid: Dict[str, list],
    cv_folds: int,
    random_state: int,
    verbose: int,
) -> Tuple[AbstractNIDS, Dict[str, Any]]:
    """Use sklearn GridSearchCV for RF and SVM."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    scorer = make_scorer(f1_score, average="macro")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    if model_type == "rf":
        base_estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    elif model_type == "svm":
        base_estimator = SVC(random_state=random_state, probability=True)
    else:
        raise ValueError(f"Unsupported sklearn model_type: {model_type}")
    
    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
        refit=True,
    )
    
    grid_search.fit(X_train, y_train)
    
    # Wrap the best estimator in our NIDS class
    best_params = grid_search.best_params_
    if model_type == "rf":
        model = RandomForestNIDS(config=best_params, random_state=random_state)
    else:
        model = SVMNIDS(config=best_params, random_state=random_state)
    
    # Fit on full training data with best params
    model.fit(X_train, y_train)
    
    results = {
        "best_params": best_params,
        "best_score": float(grid_search.best_score_),
        "cv_results": {
            k: [float(v) if isinstance(v, (int, float, np.floating)) else v
                for v in vals]
            for k, vals in grid_search.cv_results_.items()
            if k.startswith("mean_test") or k.startswith("std_test") or k.startswith("params")
        },
    }
    
    return model, results


def _dnn_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, list],
    cv_folds: int,
    random_state: int,
    verbose: int,
) -> Tuple[AbstractNIDS, Dict[str, Any]]:
    """Manual grid search for DNN (since sklearn doesn't natively support our PyTorch class)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Generate all combinations
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    if verbose > 0:
        print(f"DNN grid search: {len(combinations)} combinations, {cv_folds}-fold CV")
    
    best_score = -1.0
    best_params = None
    best_model = None
    all_results = []
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = DNNNIDS(config=params, random_state=random_state + fold)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average="macro")
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        all_results.append({"params": params, "mean_f1": float(mean_score), "fold_f1s": [float(s) for s in fold_scores]})
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            # Refit best model on full data later
    
    if verbose > 0:
        print(f"Best DNN params: {best_params}, F1: {best_score:.4f}")
    
    # Refit best model on full training data
    best_model = DNNNIDS(config=best_params, random_state=random_state)
    best_model.fit(X_train, y_train)
    
    results = {
        "best_params": best_params,
        "best_score": float(best_score),
        "all_results": all_results,
    }
    
    return best_model, results


def save_model(
    model: AbstractNIDS,
    save_dir: str,
    model_name: str,
    results: Optional[dict] = None,
) -> Path:
    """
    Save a trained NIDS model and optional results.
    
    For sklearn models (RF, SVM), saves with pickle.
    For DNN, saves PyTorch state dict.
    
    Args:
        model: Trained model.
        save_dir: Directory to save to.
        model_name: Name prefix for saved files.
        results: Optional results dict to save as JSON.
    
    Returns:
        Path to saved model file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_file = save_path / f"{model_name}.pkl"
    
    if isinstance(model, DNNNIDS):
        # Save PyTorch state dict separately
        torch_file = save_path / f"{model_name}.pt"
        if model.model is not None:
            torch.save(model.model.state_dict(), torch_file)
        # Save wrapper config with pickle
        pickle.dump(model, open(model_file, "wb"))
    else:
        pickle.dump(model, open(model_file, "wb"))
    
    if results is not None:
        results_file = save_path / f"{model_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
    
    return model_file


def load_model(model_file: str) -> AbstractNIDS:
    """Load a saved NIDS model."""
    model = pickle.load(open(model_file, "rb"))
    return model
