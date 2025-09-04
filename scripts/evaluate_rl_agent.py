"""
Evaluate trained RL agent on test set augmentation.
Measures improvement in D/P/S and impact on NIDS performance.

Artifact requirements (in data/artifacts/):
  - splits.npz
  - encoder.pt          (with embedded config)
  - kmeans.pkl
  - cluster_metadata.json
  - scaler.pkl          (for mock testbed)
  - label_encoder.pkl   (optional)
  Trained policy file (default: data/artifacts/rl_policy/cql_final.d3)
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tata.embedding.encoder import Encoder
from tata.metrics.assessment import assess_test_set
from tata.rl.environment import TataRLEnvironment
from tata.testbed.mock_testbed import MockTestbed


def _check_artifacts(artifacts_dir: Path, policy_path: Path):
    """Verify all required artifacts exist."""
    required = {
        "splits.npz": "Run scripts/run_preliminary.py first.",
        "encoder.pt": "Run scripts/run_preliminary.py first.",
        "kmeans.pkl": "Run scripts/run_preliminary.py first.",
        "cluster_metadata.json": "Run scripts/run_preliminary.py first.",
        "scaler.pkl": "Run scripts/run_preliminary.py first.",
    }
    missing = []
    for fname, hint in required.items():
        if not (artifacts_dir / fname).exists():
            missing.append(f"  - {fname}: {hint}")
    if missing:
        raise FileNotFoundError(
            f"Missing required artifacts in {artifacts_dir}:\n" + "\n".join(missing)
        )
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Trained policy not found: {policy_path}\n"
            "Run scripts/train_rl_agent.py first to train a policy."
        )


def evaluate_rl_agent(
    rl_config_path: str = "configs/rl.yaml",
    model_path: str = "data/artifacts/rl_policy/cql_final.d3",
    algorithm: str = "cql",
    artifacts_dir: str = "data/artifacts",
    seed: int = 42,
    n_episodes: int = 20,
):
    """
    Load trained policy and evaluate augmentation quality.
    
    Args:
        rl_config_path: Path to RL config.
        model_path: Path to trained policy file.
        algorithm: Algorithm name ('cql' or 'td3_plus_bc').
        artifacts_dir: Directory with all saved artifacts.
        seed: Random seed.
        n_episodes: Number of evaluation episodes.
    """
    import d3rlpy
    
    # Load configs
    with open(rl_config_path) as f:
        rl_cfg = yaml.safe_load(f)
    
    artifacts = Path(artifacts_dir)
    policy_path = Path(model_path)
    
    # Validate artifacts
    _check_artifacts(artifacts, policy_path)
    
    # Load preprocessed data splits (DO NOT regenerate)
    print("[Eval] Loading preprocessed data splits...")
    splits = np.load(artifacts / "splits.npz")
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    
    # Load scaler for mock testbed
    with open(artifacts / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Load encoder
    print("[Eval] Loading encoder...")
    encoder = Encoder.from_checkpoint(artifacts / "encoder.pt")
    
    # Encode datasets
    train_embeddings = encoder.encode(X_train)
    test_embeddings = encoder.encode(X_test)
    
    # Load clusters
    with open(artifacts / "cluster_metadata.json") as f:
        meta_json = json.load(f)
    with open(artifacts / "kmeans.pkl", "rb") as f:
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
    
    # Baseline assessment (original test set)
    print("[Eval] Baseline assessment (original test set)...")
    baseline_results = assess_test_set(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        train_labels=y_train,
        test_labels=y_test,
        cluster_metadata=cluster_metadata,
    )
    baseline = baseline_results["global"]
    print(f"  Baseline D: {baseline['diversity']:.4f} | P: {baseline['proximity']:.4f} | S: {baseline['scarcity']:.4f}")
    
    # Create mock testbed and env
    testbed = MockTestbed(
        train_features=X_train,  # unscaled for sampling
        scaler=scaler,
        label=0,
        seed=seed,
    )
    
    env = TataRLEnvironment(
        encoder=encoder,
        cluster_metadata=cluster_metadata,
        initial_test_embeddings=test_embeddings,
        initial_test_labels=y_test,
        train_embeddings=train_embeddings,
        train_labels=y_train,
        testbed=testbed,
        reward_weights=rl_cfg["rl"]["environment"]["reward_weights"],
        max_steps_per_episode=rl_cfg["rl"]["environment"]["max_steps_per_episode"],
        target_reward=rl_cfg["rl"]["environment"]["target_reward"],
    )
    
    # Load trained policy
    print(f"[Eval] Loading {algorithm} policy from {model_path}...")
    algo = d3rlpy.load_learnable(str(policy_path))
    algo.build_with_env(env)
    
    # Evaluate for n_episodes
    print(f"[Eval] Running {n_episodes} evaluation episodes...")
    diversity_improvements = []
    proximity_improvements = []
    scarcity_improvements = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        
        # Run one episode with trained policy
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = algo.predict(obs.reshape(1, -1))[0]
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Compare final metrics to baseline
        d_imp = info["diversity"] - baseline["diversity"]
        p_imp = info["proximity"] - baseline["proximity"]
        s_imp = info["scarcity"] - baseline["scarcity"]
        
        diversity_improvements.append(d_imp)
        proximity_improvements.append(p_imp)
        scarcity_improvements.append(s_imp)
        
        print(f"  Ep {ep+1}: D={info['diversity']:.4f}(+{d_imp:.4f}) "
              f"P={info['proximity']:.4f}(+{p_imp:.4f}) "
              f"S={info['scarcity']:.4f}(+{s_imp:.4f})")
    
    # Summary
    print("\n[Eval] Evaluation Summary")
    print(f"  Diversity: {np.mean(diversity_improvements):.4f} (+/- {np.std(diversity_improvements):.4f})")
    print(f"  Proximity: {np.mean(proximity_improvements):.4f} (+/- {np.std(proximity_improvements):.4f})")
    print(f"  Scarcity:  {np.mean(scarcity_improvements):.4f} (+/- {np.std(scarcity_improvements):.4f})")
    
    # Save results
    results = {
        "baseline": baseline,
        "diversity_mean_improvement": float(np.mean(diversity_improvements)),
        "proximity_mean_improvement": float(np.mean(proximity_improvements)),
        "scarcity_mean_improvement": float(np.mean(scarcity_improvements)),
    }
    
    eval_path = artifacts / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[Eval] Results saved to {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument("--rl-config", default="configs/rl.yaml")
    parser.add_argument("--model-path", default="data/artifacts/rl_policy/cql_final.d3")
    parser.add_argument("--algorithm", default="cql", choices=["cql", "td3_plus_bc"])
    parser.add_argument("--artifacts-dir", default="data/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-episodes", type=int, default=20)
    args = parser.parse_args()
    
    evaluate_rl_agent(
        rl_config_path=args.rl_config,
        model_path=args.model_path,
        algorithm=args.algorithm,
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
        n_episodes=args.n_episodes,
    )
