"""
Train offline RL agent (CQL or TD3+BC) using collected transitions.
Uses d3rlpy for offline deep reinforcement learning.
"""

import argparse
from pathlib import Path

import yaml

from tata.rl.offline_dataset import load_transitions, transitions_to_d3rlpy_dataset


def train_rl_agent(
    rl_config_path: str = "configs/rl.yaml",
    transition_path: str = "data/offline_transitions/mock_transitions.pkl",
):
    """
    Load offline transitions and train an RL agent with d3rlpy.
    """
    import d3rlpy
    
    # Load config
    with open(rl_config_path) as f:
        rl_cfg = yaml.safe_load(f)
    
    train_cfg = rl_cfg["rl"]["training"]
    
    # Load transitions
    print("[Train] Loading offline transitions...")
    if not Path(transition_path).exists():
        raise FileNotFoundError(
            f"Transition file not found: {transition_path}\n"
            "Run scripts/collect_offline_data.py first to collect transitions."
        )
    transitions = load_transitions(transition_path)
    
    # Convert to d3rlpy dataset
    print("[Train] Converting to d3rlpy dataset...")
    dataset = transitions_to_d3rlpy_dataset(transitions)
    print(f"  d3rlpy dataset size: {dataset.size()}")
    
    # Select algorithm and read hyperparameters from config
    algo_name = train_cfg["algorithm"].lower()
    actor_lr = train_cfg.get("actor_learning_rate", 3e-4)
    critic_lr = train_cfg.get("critic_learning_rate", 3e-4)
    batch_size = train_cfg.get("batch_size", 256)
    
    if algo_name == "cql":
        print(f"[Train] Using CQL (lr={actor_lr}, batch={batch_size})")
        algo = d3rlpy.algos.CQLConfig(
            actor_learning_rate=actor_lr,
            critic_learning_rate=critic_lr,
            batch_size=batch_size,
        ).create()
    elif algo_name == "td3_plus_bc":
        print(f"[Train] Using TD3+BC (lr={actor_lr}, batch={batch_size})")
        algo = d3rlpy.algos.TD3PlusBCConfig(
            actor_learning_rate=actor_lr,
            critic_learning_rate=critic_lr,
            batch_size=batch_size,
        ).create()
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Train
    print("[Train] Starting offline RL training...")
    model_save_path = Path(train_cfg["model_save_path"])
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    results = algo.fit(
        dataset,
        n_steps=train_cfg["n_steps"],
        n_steps_per_epoch=train_cfg["n_steps_per_epoch"],
        save_interval=train_cfg["save_interval"],
        experiment_name=train_cfg["experiment_name"],
        logging_steps=1000,
    )
    
    # Save final model (full learnable object for d3rlpy.load_learnable)
    final_name = train_cfg.get("final_model_name", f"{algo_name}_final.d3")
    final_path = model_save_path / final_name
    algo.save(str(final_path))
    print(f"[Train] Model saved to {final_path}")
    
    print("[Train] Training complete!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train offline RL agent")
    parser.add_argument("--rl-config", default="configs/rl.yaml")
    parser.add_argument("--transition-path", default="data/offline_transitions/mock_transitions.pkl")
    args = parser.parse_args()
    
    train_rl_agent(
        rl_config_path=args.rl_config,
        transition_path=args.transition_path,
    )
