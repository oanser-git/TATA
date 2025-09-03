"""
Offline RL dataset utilities for d3rlpy.
Handles collecting, saving, and loading transition tuples.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def collect_transitions_with_random_agent(
    env,
    n_episodes: int = 500,
    seed: int = 42,
) -> list[tuple[Any, ...]]:
    """
    Collect offline transition data using a random policy.
    
    Args:
        env: TataRLEnvironment instance.
        n_episodes: Number of episodes to run.
        seed: Random seed.
    
    Returns:
        List of (observation, action, reward, next_observation, terminal, timeout) tuples.
    """
    rng = np.random.default_rng(seed)
    transitions = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        terminated = False
        truncated = False
        step = 0
        
        while not terminated and not truncated:
            # Random action in [0, 1]^7
            action = rng.random(env.action_space.shape[0]).astype(np.float32)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            transitions.append((
                obs,
                action,
                reward,
                next_obs,
                float(terminated),
                float(truncated),
            ))
            
            obs = next_obs
            step += 1
        
        if (ep + 1) % 50 == 0:
            print(f"[Collect] Episode {ep + 1}/{n_episodes} | Steps: {step}")
    
    print(f"[Collect] Total transitions: {len(transitions)}")
    return transitions


def transitions_to_d3rlpy_dataset(
    transitions: list[tuple[Any, ...]],
):
    """
    Convert collected transitions to d3rlpy MDPDataset.
    
    Args:
        transitions: List of (obs, action, reward, next_obs, terminal, timeout) tuples.
    
    Returns:
        d3rlpy.dataset.MDPDataset
    """
    import d3rlpy
    
    observations = np.vstack([t[0] for t in transitions])
    actions = np.vstack([t[1] for t in transitions])
    rewards = np.array([t[2] for t in transitions], dtype=np.float32)
    terminals = np.array([t[4] for t in transitions], dtype=np.float32)
    timeouts = np.array([t[5] for t in transitions], dtype=np.float32)
    
    # d3rlpy handles next_observations internally when terminal flags are provided
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )
    
    return dataset


def save_transitions(
    transitions: list[tuple[Any, ...]],
    path: str,
):
    """Save transitions to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(transitions, f)
    print(f"[Dataset] Saved {len(transitions)} transitions to {path}")


def load_transitions(path: str) -> list[tuple[Any, ...]]:
    """Load transitions from disk."""
    with open(path, "rb") as f:
        transitions = pickle.load(f)
    print(f"[Dataset] Loaded {len(transitions)} transitions from {path}")
    return transitions
