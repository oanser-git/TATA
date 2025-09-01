"""
Proximity metric for TATA.
Uses one-sided Kolmogorov-Smirnov statistic.
"""

import numpy as np


def one_sided_ks_statistic(
    d_test: np.ndarray,
    d_train: np.ndarray,
    alternative: str = "greater",
) -> float:
    """
    Compute one-sided Kolmogorov-Smirnov statistic.
    
    Tests whether test distances are significantly SMALLER than train distances
    (i.e., test points are closer to negative clusters than training points).
    
    H0: F_test(x) <= F_train(x) for all x
    H1: F_test(x) > F_train(x) for some x (test has more small distances)
    
    Args:
        d_test: Distances for test points to their negative clusters.
        d_train: Distances for train points to their negative clusters.
        alternative: 'greater' means we test if test CDF is above train CDF.
    
    Returns:
        KS statistic in [0, 1].
    """
    d_test = np.asarray(d_test, dtype=float)
    d_train = np.asarray(d_train, dtype=float)
    
    if len(d_test) == 0 or len(d_train) == 0:
        return 0.0
    
    # Combine and sort unique values for ECDF evaluation
    all_vals = np.sort(np.unique(np.concatenate([d_test, d_train])))
    
    if len(all_vals) == 0:
        return 0.0
    
    # ECDFs
    def ecdf(data, x):
        return np.searchsorted(np.sort(data), x, side="right") / len(data)
    
    F_test_vals = np.array([ecdf(d_test, v) for v in all_vals])
    F_train_vals = np.array([ecdf(d_train, v) for v in all_vals])
    
    if alternative == "greater":
        # Test CDF > Train CDF => more small values in test
        diff = F_test_vals - F_train_vals
    elif alternative == "less":
        diff = F_train_vals - F_test_vals
    else:
        diff = np.abs(F_test_vals - F_train_vals)
    
    ks_stat = float(np.max(diff)) if len(diff) > 0 else 0.0
    return float(np.clip(ks_stat, 0.0, 1.0))


def compute_proximity(
    d_test: np.ndarray,
    d_train: np.ndarray,
) -> float:
    """
    Compute proximity metric for a single cluster.
    
    Uses one-sided KS statistic where higher values indicate test points
    are significantly closer to negative clusters than training points.
    
    Args:
        d_test: Test distances to negative cluster centroids.
        d_train: Train distances to negative cluster centroids.
    
    Returns:
        Proximity score in [0, 1].
    """
    return one_sided_ks_statistic(d_test, d_train, alternative="greater")
