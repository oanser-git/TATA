"""
Reward computation for RL agent.
Maps Diversity, Proximity, Scarcity to a scalar reward.
"""

from typing import Optional

import numpy as np


def compute_reward(
    diversity: float,
    proximity: float,
    scarcity: float,
    weights: Optional[dict[str, float]] = None,
) -> float:
    """
    Compute scalar reward from the three TATA metrics.
    
    R = w_D * D + w_P * P + w_S * S
    
    Higher metric values -> more challenging test set -> higher reward.
    
    Args:
        diversity: Global diversity score.
        proximity: Global proximity score.
        scarcity: Global scarcity score.
        weights: Dict with keys 'diversity', 'proximity', 'scarcity'.
    
    Returns:
        Scalar reward.
    """
    if weights is None:
        weights = {"diversity": 1.0, "proximity": 1.0, "scarcity": 1.0}
    
    reward = (
        weights.get("diversity", 1.0) * diversity +
        weights.get("proximity", 1.0) * proximity +
        weights.get("scarcity", 1.0) * scarcity
    )
    
    return float(reward)
