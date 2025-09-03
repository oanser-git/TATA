"""
TATA RL Environment for test set augmentation.
Gymnasium-compatible POMDP that works with d3rlpy.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from tata.embedding.encoder import Encoder
from tata.metrics.assessment import assess_test_set
from tata.rl.reward import compute_reward
from tata.testbed.base import NetworkTestbed


class TataRLEnvironment(gym.Env[np.ndarray, np.ndarray]):
    """
    Partially Observable Markov Decision Process for test set augmentation.
    
    Observation: cluster centroids + summary statistics of Z_test.
    Action: continuous traffic-shaping configuration [0,1]^7.
    Reward: weighted combination of D, P, S metrics.
    
    Each step:
      1. Agent observes current latent space state
      2. Agent selects action (testbed config)
      3. Testbed generates one flow
      4. Flow is encoded and appended to Z_test
      5. D/P/S are recomputed -> reward
      6. New observation is returned
    
    Args:
        encoder: Trained encoder wrapper.
        cluster_metadata: From run_clustering_pipeline.
        initial_test_embeddings: Starting Z_test (n_test, latent_dim).
        initial_test_labels: Labels for initial Z_test.
        train_embeddings: Z_train (for proximity comparison).
        train_labels: Labels for Z_train.
        testbed: NetworkTestbed instance (mock or real).
        reward_weights: Weights for D/P/S in reward.
        max_steps_per_episode: Episode length limit.
        target_reward: Terminate episode if reward exceeds this.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        encoder: Encoder,
        cluster_metadata: dict[str, Any],
        initial_test_embeddings: np.ndarray,
        initial_test_labels: np.ndarray,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        testbed: NetworkTestbed,
        reward_weights: Optional[dict[str, float]] = None,
        max_steps_per_episode: int = 5000,
        target_reward: float = 0.9,  # Paper: episode terminates when reward > 0.9
    ):
        super().__init__()
        
        self.encoder = encoder
        self.cluster_metadata = cluster_metadata
        self.initial_test_embeddings = initial_test_embeddings.copy()
        self.initial_test_labels = initial_test_labels.copy()
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.testbed = testbed
        self.reward_weights = reward_weights or {"diversity": 1.0, "proximity": 1.0, "scarcity": 1.0}
        self.max_steps = max_steps_per_episode
        self.target_reward = target_reward
        
        # Mutable state (reset each episode)
        self.test_embeddings: np.ndarray = initial_test_embeddings.copy()
        self.test_labels: np.ndarray = initial_test_labels.copy()
        self.step_count = 0
        self.prev_reward = 0.0
        
        # Observation space
        n_clusters = cluster_metadata["n_clusters"]
        latent_dim = cluster_metadata["centroids"].shape[1]
        obs_dim = n_clusters * latent_dim + 5 * latent_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: 7 continuous traffic-shaping params
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Build observation from current latent space state."""
        centroids = self.cluster_metadata["centroids"].flatten()
        
        if len(self.test_embeddings) > 0:
            stats = np.concatenate([
                self.test_embeddings.mean(axis=0),
                self.test_embeddings.min(axis=0),
                self.test_embeddings.max(axis=0),
                self.test_embeddings.var(axis=0),
                self.test_embeddings.std(axis=0),
            ])
        else:
            latent_dim = self.cluster_metadata["centroids"].shape[1]
            stats = np.zeros(5 * latent_dim)
        
        obs = np.concatenate([centroids, stats]).astype(np.float32)
        return obs
    
    def _compute_metrics(self) -> tuple[float, float, float]:
        """Compute current D, P, S on the evolving test set."""
        results = assess_test_set(
            train_embeddings=self.train_embeddings,
            test_embeddings=self.test_embeddings,
            train_labels=self.train_labels,
            test_labels=self.test_labels,
            cluster_metadata=self.cluster_metadata,
        )
        g = results["global"]
        return g["diversity"], g["proximity"], g["scarcity"]
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to original test set."""
        super().reset(seed=seed)
        
        self.test_embeddings = self.initial_test_embeddings.copy()
        self.test_labels = self.initial_test_labels.copy()
        self.step_count = 0
        
        # Compute initial metrics
        d, p, s = self._compute_metrics()
        self.prev_reward = compute_reward(d, p, s, self.reward_weights)
        
        return self._get_observation(), {
            "diversity": d,
            "proximity": p,
            "scarcity": s,
            "reward": self.prev_reward,
        }
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one RL step.
        
        Args:
            action: Traffic-shaping config in [0,1]^7.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # 1. Apply action to testbed
        self.testbed.apply_config(action)
        
        # 2. Generate traffic (mock -> .npy path)
        pcap_path = self.testbed.generate_traffic()
        
        # 3. Collect flow features
        flow_features = self.testbed.collect_flows(pcap_path)  # (1, n_features)
        
        # 4. Encode to latent space
        new_embedding = self.encoder.encode(flow_features)  # (1, latent_dim)
        
        # 5. Append to evolving test set
        self.test_embeddings = np.vstack([self.test_embeddings, new_embedding])
        # Label of generated flow (from testbed, default = benign = 0)
        new_label = getattr(self.testbed, "label", 0)
        self.test_labels = np.append(self.test_labels, new_label)
        
        # 6. Compute updated metrics
        d, p, s = self._compute_metrics()
        reward = compute_reward(d, p, s, self.reward_weights)
        
        # 7. Check termination
        terminated = reward >= self.target_reward
        truncated = self.step_count >= self.max_steps
        
        info = {
            "diversity": d,
            "proximity": p,
            "scarcity": s,
            "reward": reward,
            "step": self.step_count,
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def render(self):
        pass
