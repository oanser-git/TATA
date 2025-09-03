"""
RL hyperparameter tuning via Split-Select-Retrain (Appendix B).

Implements Nie et al. (2022) pipeline:
  1. Create K=20 independent 50/50 train/validation splits of offline data
  2. For each algorithm-hyperparameter config:
     a. Train candidate policy on each training split
     b. Evaluate on validation split using FQE
  3. Average K FQE estimates -> robust performance metric per config
  4. Select best config and retrain on full offline dataset
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from tata.rl.fqe import evaluate_policy_with_fqe


RL_CONFIGS = {
    "cql_default": {
        "algorithm": "CQL",
        "n_epochs": 100,
        "batch_size": 256,
        "learning_rate": 3e-4,
    },
    "cql_high_lr": {
        "algorithm": "CQL",
        "n_epochs": 100,
        "batch_size": 256,
        "learning_rate": 1e-3,
    },
    "cql_large_batch": {
        "algorithm": "CQL",
        "n_epochs": 100,
        "batch_size": 512,
        "learning_rate": 3e-4,
    },
    "td3bc_default": {
        "algorithm": "TD3PlusBC",
        "n_epochs": 100,
        "batch_size": 256,
        "learning_rate": 3e-4,
    },
    "td3bc_high_lr": {
        "algorithm": "TD3PlusBC",
        "n_epochs": 100,
        "batch_size": 256,
        "learning_rate": 1e-3,
    },
    "td3bc_large_batch": {
        "algorithm": "TD3PlusBC",
        "n_epochs": 100,
        "batch_size": 512,
        "learning_rate": 3e-4,
    },
}


def create_k_folds(
    dataset,
    K: int = 20,
    random_state: int = 42,
) -> list[tuple]:
    """
    Create K independent 50/50 train/validation splits.
    
    Args:
        dataset: d3rlpy dataset.
        K: Number of splits.
        random_state: Base random seed.
    
    Returns:
        List of (train_dataset, val_dataset) tuples.
    """
    observations = dataset.observations
    actions = dataset.actions
    rewards = dataset.rewards
    terminals = dataset.terminals
    timeouts = dataset.timeouts if hasattr(dataset, "timeouts") else np.zeros_like(terminals)
    
    n = len(observations)
    indices = np.arange(n)
    splits = []
    
    for k in range(K):
        rng = np.random.default_rng(random_state + k)
        perm = rng.permutation(n)
        mid = n // 2
        train_idx = perm[:mid]
        val_idx = perm[mid:]
        
        import d3rlpy
        train_ds = d3rlpy.dataset.MDPDataset(
            observations=observations[train_idx],
            actions=actions[train_idx],
            rewards=rewards[train_idx],
            terminals=terminals[train_idx],
            timeouts=timeouts[train_idx],
        )
        val_ds = d3rlpy.dataset.MDPDataset(
            observations=observations[val_idx],
            actions=actions[val_idx],
            rewards=rewards[val_idx],
            terminals=terminals[val_idx],
            timeouts=timeouts[val_idx],
        )
        splits.append((train_ds, val_ds))
    
    return splits


def train_candidate_policy(
    train_dataset,
    config: dict[str, Any],
    model_save_dir: str,
    seed: int = 42,
) -> str:
    """
    Train a single candidate policy.
    
    Args:
        train_dataset: d3rlpy dataset.
        config: Dict with algorithm, n_epochs, batch_size, learning_rate.
        model_save_dir: Where to save the model.
        seed: Random seed.
    
    Returns:
        Path to saved model file.
    """
    import d3rlpy
    import gymnasium as gym
    
    # Create algorithm instance
    algo_name = config["algorithm"]
    if algo_name == "CQL":
        algo = d3rlpy.algos.CQLConfig(
            actor_learning_rate=config["learning_rate"],
            critic_learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
        ).create()
    elif algo_name == "TD3PlusBC":
        algo = d3rlpy.algos.TD3PlusBCConfig(
            actor_learning_rate=config["learning_rate"],
            critic_learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
        ).create()
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Train
    save_path = Path(model_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_file = save_path / "model.d3"
    
    algo.fit(
        train_dataset,
        n_epochs=config["n_epochs"],
        save_interval=0,  # Don't auto-save during training
    )
    algo.save(str(model_file))
    
    return str(model_file)


def evaluate_config_with_fqe(
    candidate_path: str,
    val_dataset,
    fqe_epochs: int = 100,
    device: str = "cpu",
) -> float:
    """
    Evaluate a candidate policy on validation data using FQE.
    
    Args:
        candidate_path: Path to trained d3rlpy model.
        val_dataset: Validation dataset.
        fqe_epochs: FQE training epochs.
        device: Device for FQE.
    
    Returns:
        Estimated policy value (mean Q on initial states).
    """
    import d3rlpy
    
    # Load candidate policy
    algo = d3rlpy.load_learnable(candidate_path)
    
    # Evaluate with FQE
    value = evaluate_policy_with_fqe(
        dataset=val_dataset,
        policy=algo,
        gamma=0.99,
        epochs=fqe_epochs,
        device=device,
    )
    
    return value


def run_split_select_retrain(
    offline_dataset,
    configs: Optional[dict[str, dict[str, Any]]] = None,
    K: int = 20,
    fqe_epochs: int = 100,
    random_state: int = 42,
    device: str = "cpu",
    save_dir: str = "data/artifacts/rl_hpo",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run full Split-Select-Retrain pipeline (Appendix B).
    
    Args:
        offline_dataset: Full offline d3rlpy dataset.
        configs: Dict of config_name -> hyperparameter dict.
        K: Number of train/val splits.
        fqe_epochs: FQE training epochs per evaluation.
        random_state: Random seed.
        device: 'cpu' or 'cuda'.
        save_dir: Where to save results and candidate models.
        verbose: Print progress.
    
    Returns:
        Dict with best_config_name, best_score, all_scores, and final_model_path.
    """
    if configs is None:
        configs = RL_CONFIGS
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create K folds
    if verbose:
        print(f"[SSR] Creating {K} train/validation splits...")
    folds = create_k_folds(offline_dataset, K=K, random_state=random_state)
    
    # Step 2 & 3: For each config, train on each fold, evaluate with FQE
    config_scores = {}
    
    for config_name, config in configs.items():
        if verbose:
            print(f"\n[SSR] Evaluating config: {config_name}")
        
        fold_values = []
        
        for fold_idx, (train_ds, val_ds) in enumerate(folds):
            if verbose and (fold_idx + 1) % 5 == 0:
                print(f"  Fold {fold_idx + 1}/{K}...")
            
            # Train candidate
            candidate_dir = save_path / "candidates" / config_name / f"fold_{fold_idx}"
            candidate_path = train_candidate_policy(
                train_dataset=train_ds,
                config=config,
                model_save_dir=str(candidate_dir),
                seed=random_state + fold_idx,
            )
            
            # Evaluate with FQE
            value = evaluate_config_with_fqe(
                candidate_path=candidate_path,
                val_dataset=val_ds,
                fqe_epochs=fqe_epochs,
                device=device,
            )
            fold_values.append(value)
        
        mean_value = float(np.mean(fold_values))
        std_value = float(np.std(fold_values))
        config_scores[config_name] = {
            "mean_fqe_value": mean_value,
            "std_fqe_value": std_value,
            "fold_values": [float(v) for v in fold_values],
        }
        
        if verbose:
            print(f"  Mean FQE: {mean_value:.4f} +/- {std_value:.4f}")
    
    # Step 4: Select best config
    best_config_name = max(config_scores, key=lambda k: config_scores[k]["mean_fqe_value"])
    best_score = config_scores[best_config_name]["mean_fqe_value"]
    best_config = configs[best_config_name]
    
    if verbose:
        print(f"\n[SSR] Best config: {best_config_name} (FQE={best_score:.4f})")
    
    # Step 5: Retrain on full offline dataset with best config
    if verbose:
        print(f"[SSR] Retraining final model on full offline dataset...")
    
    final_model_dir = save_path / "final_model"
    final_model_path = train_candidate_policy(
        train_dataset=offline_dataset,
        config=best_config,
        model_save_dir=str(final_model_dir),
        seed=random_state,
    )
    
    # Save results
    results = {
        "best_config_name": best_config_name,
        "best_config": best_config,
        "best_score": best_score,
        "all_scores": config_scores,
        "K": K,
        "final_model_path": final_model_path,
    }
    
    with open(save_path / "ssr_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"[SSR] Results saved to {save_path / 'ssr_results.json'}")
    
    return results
