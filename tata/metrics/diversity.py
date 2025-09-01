"""
Diversity metric for TATA.
Uses the official vendi-score package.
https://github.com/vertaix/vendi-score
"""

import numpy as np
from vendi_score import vendi


def compute_diversity(
    test_embeddings: np.ndarray,
    min_max_normalize: bool = True,
    gamma: float = 1.0,
) -> float:
    """
    Compute diversity metric for a single cluster's test points.
    Uses the official vendi-score package.
    
    Args:
        test_embeddings: Array of shape (n_test, latent_dim).
        min_max_normalize: If True, normalize Vendi score to [0, 1].
        gamma: RBF kernel gamma parameter for Vendi score.
    
    Returns:
        Diversity score in [0, 1] if normalized, else [1, n_test].
    """
    n = test_embeddings.shape[0]
    if n <= 1:
        return 0.0 if min_max_normalize else 1.0
    
    # Use official vendi-score with RBF kernel
    v = vendi.score(
        test_embeddings,
        k=lambda x, y: np.exp(-gamma * np.sum((x - y) ** 2)),
        normalize=False,
    )
    
    if min_max_normalize:
        # Normalize: (V - 1) / (n - 1)
        return (v - 1.0) / (n - 1.0)
    return v
