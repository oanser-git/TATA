"""
Augmentation experiment: partial rewards and NIDS performance impact.
Section 4.4 of the paper.

Key evaluations:
  1. Full reward (D+P+S) augmentation vs baselines
  2. Partial reward ablation: D, P, S, DP, PS, DS (6 cases for Fig 9)
  3. Transfer evaluation: train agent on dataset A, evaluate on dataset B

Reports macro-averaged precision, recall, and F1 for RF, SVM, DNN.
"""

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tata.rl.environment import TataRLEnvironment
from tata.rl.reward import compute_reward


REWARD_CONFIGS = {
    "full": {"diversity": 1.0, "proximity": 1.0, "scarcity": 1.0},
    "diversity_only": {"diversity": 1.0, "proximity": 0.0, "scarcity": 0.0},
    "proximity_only": {"diversity": 0.0, "proximity": 1.0, "scarcity": 0.0},
    "scarcity_only": {"diversity": 0.0, "proximity": 0.0, "scarcity": 1.0},
    "dp": {"diversity": 1.0, "proximity": 1.0, "scarcity": 0.0},
    "ps": {"diversity": 0.0, "proximity": 1.0, "scarcity": 1.0},
    "ds": {"diversity": 1.0, "proximity": 0.0, "scarcity": 1.0},
}


def rollout_agent(
    env: TataRLEnvironment,
    agent,
    n_episodes: int = 20,
    deterministic: bool = True,
) -> dict[str, Any]:
    """
    Rollout a trained agent for N episodes.

    Returns:
        Dict with episode statistics and the final augmented test set.
    """
    episode_rewards = []
    episode_lengths = []
    final_diversity = []
    final_proximity = []
    final_scarcity = []

    # Track the best test set (highest reward) across episodes
    best_test_emb = None
    best_test_labels = None
    best_reward = -float("inf")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action = agent.predict(np.expand_dims(obs, axis=0), deterministic=deterministic)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        final_diversity.append(info.get("diversity", 0.0))
        final_proximity.append(info.get("proximity", 0.0))
        final_scarcity.append(info.get("scarcity", 0.0))

        if info.get("reward", 0.0) > best_reward:
            best_reward = info["reward"]
            best_test_emb = env.test_embeddings.copy()
            best_test_labels = env.test_labels.copy()

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "final_diversity": final_diversity,
        "final_proximity": final_proximity,
        "final_scarcity": final_scarcity,
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_diversity": float(np.mean(final_diversity)),
        "mean_proximity": float(np.mean(final_proximity)),
        "mean_scarcity": float(np.mean(final_scarcity)),
        "best_test_embeddings": best_test_emb,
        "best_test_labels": best_test_labels,
    }


def evaluate_nids_on_augmented_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_aug_test: np.ndarray,
    y_aug_test: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    nids_models: list[Any],
) -> dict[str, dict[str, float]]:
    """
    Train each NIDS on (train + augmented_test) and evaluate on holdout.

    Returns:
        Dict mapping model_name -> {precision, recall, f1, accuracy}
    """
    results = {}

    # Combine train + augmented test
    X_combined = np.vstack([X_train, X_aug_test])
    y_combined = np.concatenate([y_train, y_aug_test])

    # Shuffle
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X_combined))
    X_combined = X_combined[perm]
    y_combined = y_combined[perm]

    for model in nids_models:
        model_name = type(model).__name__
        model_copy = type(model)(config=model.config, random_state=42)
        model_copy.fit(X_combined, y_combined)
        metrics = model_copy.evaluate(X_holdout, y_holdout, average="macro")
        results[model_name] = metrics

    return results


def evaluate_reward_configs(
    env_factory: Callable[[], TataRLEnvironment],
    agent_paths: dict[str, str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    nids_models: list[Any],
    n_episodes: int = 20,
    save_dir: str = "data/artifacts/augmentation",
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate multiple trained agents with different reward configs.
    Reports both test-set quality metrics and NIDS precision/recall/f1.

    Args:
        env_factory: Callable that returns a fresh TataRLEnvironment.
        agent_paths: Dict of config_name -> path to trained d3rlpy agent.
        X_train, y_train: Training data (scaled).
        X_holdout, y_holdout: Holdout evaluation data (scaled).
        nids_models: List of untrained NIDS model instances.
        n_episodes: Episodes per evaluation.
        save_dir: Where to save results.
        verbose: Print progress.

    Returns:
        Results dict per config.
    """
    import d3rlpy

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    results = {}

    for config_name, agent_path in agent_paths.items():
        if verbose:
            print(f"\n=== Evaluating: {config_name} ===")

        # Load agent
        agent = d3rlpy.load_learnable(agent_path)

        # Create fresh environment
        env = env_factory()

        # Rollout
        stats = rollout_agent(env, agent, n_episodes=n_episodes)

        if stats["best_test_embeddings"] is None:
            if verbose:
                print(f"  Warning: no valid episodes for {config_name}")
            continue

        # Evaluate NIDS on augmented test set
        nids_results = evaluate_nids_on_augmented_test(
            X_train=X_train,
            y_train=y_train,
            X_aug_test=stats["best_test_embeddings"],
            y_aug_test=stats["best_test_labels"],
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            nids_models=nids_models,
        )

        results[config_name] = {
            "metrics": {
                "mean_diversity": stats["mean_diversity"],
                "mean_proximity": stats["mean_proximity"],
                "mean_scarcity": stats["mean_scarcity"],
                "mean_reward": stats["mean_reward"],
            },
            "nids": nids_results,
        }

        if verbose:
            print(f"  D/P/S: {stats['mean_diversity']:.4f} / {stats['mean_proximity']:.4f} / {stats['mean_scarcity']:.4f}")
            for model_name, m in nids_results.items():
                print(f"  {model_name}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    # Save
    with open(save_path / "augmentation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_transfer_evaluation(
    train_dataset_name: str,
    eval_dataset_name: str,
    data_dir: str = "data",
    agent_path: str = "",
    n_episodes: int = 20,
    device: str = "cpu",
    random_state: int = 42,
    save_dir: str = "data/artifacts/transfer",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Transfer evaluation: train agent on one dataset, evaluate on another.
    Paper: train on IDS2018, evaluate on IDS2017.

    Requires pre-trained encoder and clusters for both datasets.
    """
    from tata.datasets.loaders import load_dataset
    from tata.embedding.clustering import run_clustering_pipeline
    from tata.embedding.encoder import Encoder
    from tata.models.autoencoders import ContrastiveAutoencoder
    from tata.models.nids import RandomForestNIDS, SVMNIDS, DNNNIDS
    from tata.models.training import train_autoencoder
    from tata.testbed.mock_testbed import MockTestbed

    import d3rlpy

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load evaluation dataset
    if verbose:
        print(f"Loading evaluation dataset: {eval_dataset_name}")
    X_eval, y_eval = load_dataset(eval_dataset_name, data_dir=data_dir)

    # 60/20/20 split for eval dataset
    _X_tr, _X_temp, _y_tr, _y_temp = train_test_split(
        X_eval, y_eval, test_size=0.4, random_state=random_state, stratify=y_eval
    )
    _X_val, _X_test, _y_val, _y_test = train_test_split(
        _X_temp, _y_temp, test_size=0.5, random_state=random_state, stratify=_y_temp
    )
    X_train_eval = np.asarray(_X_tr)
    X_val_eval = np.asarray(_X_val)
    X_test_eval = np.asarray(_X_test)
    y_train_eval = np.asarray(_y_tr)
    y_val_eval = np.asarray(_y_val)
    y_test_eval = np.asarray(_y_test)

    scaler = StandardScaler()
    X_train_eval_s = scaler.fit_transform(X_train_eval)
    X_test_eval_s = scaler.transform(X_test_eval)

    # Train encoder on eval dataset train set
    if verbose:
        print("Training encoder on evaluation dataset...")
    model = ContrastiveAutoencoder(
        input_dim=X_train_eval_s.shape[1],
        encoder_dims=[64, 32, 16],
        latent_dim=3,
        decoder_dims=[16, 32, 64],
    )
    trained, _ = train_autoencoder(
        model, X_train_eval_s, y_train_eval,
        epochs=250, batch_size=128,
        learning_rate=0.001,
        lambda_contrastive=0.1,
        margin=10.0,
        device=device,
        verbose=False,
    )
    encoder = Encoder(trained, device=device)

    train_emb = encoder.encode(X_train_eval_s)
    test_emb = encoder.encode(X_test_eval_s)

    best_k, kmeans, metadata = run_clustering_pipeline(
        train_emb, y_train_eval, random_state=random_state
    )

    # Build environment
    testbed = MockTestbed(
        train_features=X_train_eval,
        scaler=scaler,
        seed=random_state,
    )

    env = TataRLEnvironment(
        encoder=encoder,
        cluster_metadata=metadata,
        initial_test_embeddings=test_emb,
        initial_test_labels=y_test_eval,
        train_embeddings=train_emb,
        train_labels=y_train_eval,
        testbed=testbed,
    )

    # Load pre-trained agent
    if verbose:
        print(f"Loading pre-trained agent from {agent_path}")
    agent = d3rlpy.load_learnable(agent_path)

    # Rollout
    stats = rollout_agent(env, agent, n_episodes=n_episodes)

    # Evaluate NIDS
    nids_models = [RandomForestNIDS(), SVMNIDS(), DNNNIDS()]
    for m in nids_models:
        m.fit(X_train_eval_s, y_train_eval)

    nids_results = evaluate_nids_on_augmented_test(
        X_train=X_train_eval_s,
        y_train=y_train_eval,
        X_aug_test=stats["best_test_embeddings"] if stats["best_test_embeddings"] is not None else test_emb,
        y_aug_test=stats["best_test_labels"] if stats["best_test_labels"] is not None else y_test_eval,
        X_holdout=X_val_eval,  # use val as holdout
        y_holdout=y_val_eval,
        nids_models=nids_models,
    )

    results = {
        "train_dataset": train_dataset_name,
        "eval_dataset": eval_dataset_name,
        "metrics": {
            "mean_diversity": stats["mean_diversity"],
            "mean_proximity": stats["mean_proximity"],
            "mean_scarcity": stats["mean_scarcity"],
        },
        "nids": nids_results,
    }

    with open(save_path / "transfer_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nTransfer evaluation complete. Results saved to {save_path}")

    return results
