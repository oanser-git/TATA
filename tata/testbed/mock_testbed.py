"""
Mock/Synthetic testbed for rapid RL pipeline validation.
No real network traffic; instead generates synthetic flow features
by perturbing samples from the training distribution based on the action.
"""

from typing import Optional

import numpy as np

from tata.testbed.base import NetworkTestbed


class MockTestbed(NetworkTestbed):
    """
    Mock testbed that simulates traffic generation without real VMs or tc.
    
    How it works:
      - Maintains statistics of the original training data
      - When an action is applied, samples a base flow and perturbs it
        proportionally to the action magnitudes
      - Higher actions = more aggressive perturbations = more "unusual" flows
      - Fast (~1ms per step) for rapid RL pipeline iteration
    
    Action mapping (continuous [0,1]):
      0: loss       -> perturb packet/byte count features
      1: jitter     -> perturb timing features
      2: delay      -> perturb timing features
      3: duplication -> perturb packet count features
      4: corruption -> perturb byte count features
      5: reordering -> perturb timing/order features
      6: correlation -> controls coherence of perturbation
    """
    
    def __init__(
        self,
        train_features: np.ndarray,
        scaler,  # fitted sklearn scaler
        label: int = 0,  # label for generated flows (e.g., 0 = Benign)
        seed: int = 42,
    ):
        """
        Args:
            train_features: Raw (unscaled) training features to sample from.
            scaler: Fitted StandardScaler/MinMaxScaler to transform generated flows.
            label: Integer label assigned to all generated flows.
            seed: Random seed.
        """
        self.train_features = train_features
        self.scaler = scaler
        self.label = label
        self.rng = np.random.default_rng(seed)
        
        # Pre-compute per-feature statistics for perturbation
        self.feature_means = train_features.mean(axis=0)
        self.feature_stds = train_features.std(axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0  # avoid division by zero
        
        # Track current config (for debugging)
        self.current_config = None
        self.n_generated = 0
    
    # Paper action ranges (Section 4.4)
    ACTION_RANGES = {
        "loss": (0.05, 0.10),       # 5% - 10%
        "jitter": (4.0, 10.0),      # 4ms - 10ms
        "delay": (10.0, 40.0),      # 10ms - 40ms
        "duplication": (0.001, 0.05), # 0.1% - 5%
        "corruption": (0.001, 0.10),  # 0.1% - 10%
        "reordering": (0.001, 0.50),  # 0.1% - 50%
        "correlation": (0.50, 1.00),  # 50% - 100%
    }

    def _map_action(self, raw_action: np.ndarray) -> dict[str, float]:
        """Map continuous [0,1] action to paper-specific parameter ranges."""
        action = np.clip(raw_action, 0.0, 1.0)
        keys = ["loss", "jitter", "delay", "duplication", "corruption", "reordering", "correlation"]
        mapped = {}
        for i, key in enumerate(keys):
            lo, hi = self.ACTION_RANGES[key]
            mapped[key] = float(lo + action[i] * (hi - lo))
        return mapped

    def apply_config(self, action: np.ndarray) -> None:
        """
        Store the action as the current configuration.
        Maps [0,1] continuous values to paper-specific tc parameter ranges.
        """
        self.current_config = self._map_action(action)
    
    def generate_traffic(self, scenario: str = "ssh") -> str:
        """
        Generate a synthetic flow feature vector.
        
        Args:
            scenario: Ignored in mock mode (for API compatibility).
        
        Returns:
            Path to a .npy file containing the generated flow (mock).
        """
        if self.current_config is None:
            raise RuntimeError("Must call apply_config() before generate_traffic()")
        
        # Sample a base flow from training data
        base_idx = self.rng.integers(0, len(self.train_features))
        base_flow = self.train_features[base_idx].copy().astype(float)
        
        # Extract action values
        cfg = self.current_config
        
        # Compute perturbation magnitude from action
        # More aggressive actions -> larger perturbations
        # We scale each action by the corresponding feature std
        perturbation = np.zeros_like(base_flow)
        
        # Feature groups (heuristic based on CICFlowMeter-like features)
        n_feat = len(base_flow)
        
        # Group 1: Packet count features (affected by loss, duplication)
        pkt_indices = list(range(0, min(20, n_feat)))
        # Group 2: Byte count features (affected by corruption)
        byte_indices = list(range(20, min(40, n_feat)))
        # Group 3: Timing features (affected by jitter, delay, reordering)
        time_indices = list(range(40, min(60, n_feat)))
        # Group 4: Statistical features (affected by correlation)
        stat_indices = list(range(60, n_feat))
        
        # Apply perturbations per group
        if len(pkt_indices) > 0:
            noise = self.rng.normal(0, cfg["loss"] + cfg["duplication"], len(pkt_indices))
            perturbation[pkt_indices] = noise * self.feature_stds[pkt_indices]
        
        if len(byte_indices) > 0:
            noise = self.rng.normal(0, cfg["corruption"], len(byte_indices))
            perturbation[byte_indices] = noise * self.feature_stds[byte_indices]
        
        if len(time_indices) > 0:
            noise = self.rng.normal(0, cfg["jitter"] + cfg["delay"] + cfg["reordering"], len(time_indices))
            perturbation[time_indices] = noise * self.feature_stds[time_indices]
        
        if len(stat_indices) > 0:
            # Correlation controls coherence: high correlation -> more structured noise
            coherence = cfg["correlation"]
            shared_noise = self.rng.normal(0, 1.0)
            noise = coherence * shared_noise + (1 - coherence) * self.rng.normal(0, 1.0, len(stat_indices))
            perturbation[stat_indices] = noise * self.feature_stds[stat_indices]
        
        # Apply perturbation to base flow
        perturbed_flow = base_flow + perturbation
        
        # Ensure non-negativity for count-like features
        perturbed_flow = np.clip(perturbed_flow, a_min=0.0, a_max=None)
        
        # Scale using the saved scaler (to match the latent space)
        scaled_flow = self.scaler.transform(perturbed_flow.reshape(1, -1))
        
        self.n_generated += 1
        
        # Save to temp file for API compatibility
        import tempfile
        tmp_path = tempfile.mktemp(suffix=".npy")
        np.save(tmp_path, scaled_flow)
        return tmp_path
    
    def collect_flows(self, pcap_path: str) -> np.ndarray:
        """
        In mock mode, the pcap_path is actually a .npy path from generate_traffic().
        
        Returns:
            Flow feature vector of shape (1, n_features).
        """
        flows = np.load(pcap_path)
        if flows.ndim == 1:
            flows = flows.reshape(1, -1)
        return flows
    
    def reset(self) -> None:
        """Reset configuration."""
        self.current_config = None
        self.n_generated = 0
    
    def close(self) -> None:
        """No-op for mock testbed."""
        pass
