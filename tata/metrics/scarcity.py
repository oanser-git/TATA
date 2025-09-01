"""
Scarcity metric for TATA.
Uses Gini coefficient.
"""

import numpy as np


def gini_coefficient(counts: np.ndarray, corrected: bool = True) -> float:
    """
    Compute Gini coefficient for a distribution.
    
    Gini = 0 means perfect equality (uniform distribution).
    Gini -> 1 means perfect inequality (all mass in one bin).
    
    Args:
        counts: Array of counts per category.
        corrected: Apply bias correction for small samples.
    
    Returns:
        Gini coefficient in [0, 1].
    """
    counts = np.asarray(counts, dtype=float)
    n = len(counts)
    
    if n == 0 or counts.sum() == 0:
        return 0.0
    
    # Sort counts
    sorted_counts = np.sort(counts)
    
    # Cumulative sums
    cumsum = np.cumsum(sorted_counts)
    
    # Gini formula
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    if corrected and n > 1:
        gini *= n / (n - 1)
    
    return float(np.clip(gini, 0.0, 1.0))


def compute_scarcity(
    negative_cluster_counts: np.ndarray,
    corrected: bool = True,
) -> float:
    """
    Compute scarcity metric for a single cluster.
    Scarcity = 1 - Gini(counts_per_negative_cluster).
    
    Higher scarcity means test points are more uniformly distributed
    across all possible negative clusters.
    
    Args:
        negative_cluster_counts: Array of counts mapping to each negative cluster.
        corrected: Apply bias correction for Gini.
    
    Returns:
        Scarcity score in [0, 1].
    """
    gini = gini_coefficient(negative_cluster_counts, corrected=corrected)
    return 1.0 - gini
