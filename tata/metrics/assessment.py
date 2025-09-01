"""
Phase 1 assessment orchestrator.
Computes diversity, proximity, and scarcity metrics for a test set
relative to a training set in the structured latent space.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from tata.metrics.diversity import compute_diversity
from tata.metrics.proximity import compute_proximity
from tata.metrics.scarcity import compute_scarcity


def assign_positive_negative_clusters(
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    cluster_centroids: np.ndarray,
    cluster_majority_labels: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each test point, find its positive cluster (same label, closest)
    and negative cluster (different label, closest).
    
    Args:
        test_embeddings: (n_test, latent_dim)
        test_labels: (n_test,) integer labels
        cluster_centroids: (n_clusters, latent_dim)
        cluster_majority_labels: (n_clusters,) majority label per cluster
    
    Returns:
        positive_clusters: (n_test,) cluster indices
        negative_clusters: (n_test,) cluster indices
    """
    n_test = test_embeddings.shape[0]
    n_clusters = cluster_centroids.shape[0]
    
    # Compute distances to all centroids
    # (n_test, n_clusters)
    distances = np.linalg.norm(
        test_embeddings[:, np.newaxis, :] - cluster_centroids[np.newaxis, :, :],
        axis=2,
    )
    
    positive_clusters = np.zeros(n_test, dtype=int)
    negative_clusters = np.zeros(n_test, dtype=int)
    
    for i in range(n_test):
        label = test_labels[i]
        dists = distances[i]
        
        # Positive: same label, minimum distance
        same_label_mask = np.array(cluster_majority_labels) == label
        if same_label_mask.any():
            pos_candidates = np.where(same_label_mask)[0]
            positive_clusters[i] = pos_candidates[np.argmin(dists[same_label_mask])]
        else:
            # Fallback: no cluster with same label, use closest overall
            positive_clusters[i] = np.argmin(dists)
        
        # Negative: different label, minimum distance
        diff_label_mask = ~same_label_mask
        if diff_label_mask.any():
            neg_candidates = np.where(diff_label_mask)[0]
            negative_clusters[i] = neg_candidates[np.argmin(dists[diff_label_mask])]
        else:
            # Fallback: no cluster with different label
            negative_clusters[i] = positive_clusters[i]
    
    return positive_clusters, negative_clusters


def compute_distances_to_negative_clusters(
    embeddings: np.ndarray,
    cluster_assignments: np.ndarray,
    negative_clusters: np.ndarray,
    cluster_centroids: np.ndarray,
) -> Dict[int, List[float]]:
    """
    Compute per-cluster distances to negative cluster centroids.
    
    Args:
        embeddings: (n_samples, latent_dim)
        cluster_assignments: (n_samples,) positive cluster index per sample
        negative_clusters: (n_samples,) negative cluster index per sample
        cluster_centroids: (n_clusters, latent_dim)
    
    Returns:
        Dict mapping cluster_id -> list of distances.
    """
    distances_per_cluster = {}
    for c in range(cluster_centroids.shape[0]):
        mask = cluster_assignments == c
        if not mask.any():
            distances_per_cluster[c] = []
            continue
        
        cluster_embs = embeddings[mask]
        cluster_negs = negative_clusters[mask]
        
        dists = np.linalg.norm(cluster_embs - cluster_centroids[cluster_negs], axis=1)
        distances_per_cluster[c] = dists.tolist()
    
    return distances_per_cluster


def assess_test_set(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    cluster_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full Phase 1 assessment: compute Diversity, Proximity, Scarcity.
    
    Args:
        train_embeddings: (n_train, latent_dim) from encoder.
        test_embeddings: (n_test, latent_dim) from encoder.
        train_labels: (n_train,) integer labels.
        test_labels: (n_test,) integer labels.
        cluster_metadata: Dict from build_cluster_metadata with keys:
            - n_clusters, centroids, cluster_majority_labels, cluster_member_indices
    
    Returns:
        Dict with global and per-cluster metrics.
    """
    n_clusters = cluster_metadata["n_clusters"]
    centroids = cluster_metadata["centroids"]
    cluster_majority_labels = cluster_metadata["cluster_majority_labels"]
    cluster_member_indices = cluster_metadata["cluster_member_indices"]
    
    # Step 1: Assign positive/negative clusters for test points
    test_pos_clusters, test_neg_clusters = assign_positive_negative_clusters(
        test_embeddings, test_labels, centroids, cluster_majority_labels
    )
    
    # Step 2: Assign positive/negative clusters for train points
    train_pos_clusters, train_neg_clusters = assign_positive_negative_clusters(
        train_embeddings, train_labels, centroids, cluster_majority_labels
    )
    
    # Step 3: Compute distances to negative clusters
    test_distances = compute_distances_to_negative_clusters(
        test_embeddings, test_pos_clusters, test_neg_clusters, centroids
    )
    train_distances = compute_distances_to_negative_clusters(
        train_embeddings, train_pos_clusters, train_neg_clusters, centroids
    )
    
    # Step 4: Per-cluster metrics
    per_cluster = {}
    diversities = []
    proximities = []
    scarcities = []
    
    for c in range(n_clusters):
        # Get test points whose positive cluster is c
        test_mask = test_pos_clusters == c
        test_points_c = test_embeddings[test_mask]
        
        # Diversity for cluster c
        if len(test_points_c) > 1:
            d = compute_diversity(test_points_c, min_max_normalize=True)
        elif len(test_points_c) == 1:
            d = 0.0  # Minimal diversity with 1 point
        else:
            d = 0.0  # No test points -> no diversity
        
        # Proximity for cluster c
        d_test_c = np.array(test_distances.get(c, []))
        d_train_c = np.array(train_distances.get(c, []))
        
        if len(d_test_c) > 0 and len(d_train_c) > 0:
            p = compute_proximity(d_test_c, d_train_c)
        else:
            p = 0.0
        
        # Scarcity for cluster c
        # Count how many test points map to each negative cluster
        neg_counts = {}
        all_possible_negs = set()
        for other_c in range(n_clusters):
            if cluster_majority_labels[other_c] != cluster_majority_labels[c]:
                all_possible_negs.add(other_c)
        
        test_neg_for_c = test_neg_clusters[test_mask]
        for neg_c in all_possible_negs:
            neg_counts[neg_c] = int((test_neg_for_c == neg_c).sum())
        
        counts_array = np.array(list(neg_counts.values()), dtype=float)
        if len(counts_array) > 0 and counts_array.sum() > 0:
            s = compute_scarcity(counts_array)
        else:
            s = 0.0
        
        per_cluster[c] = {
            "diversity": float(d),
            "proximity": float(p),
            "scarcity": float(s),
            "n_test_points": int(len(test_points_c)),
            "n_train_points": int(len(cluster_member_indices[c])),
            "majority_label": int(cluster_majority_labels[c]),
        }
        
        diversities.append(d)
        proximities.append(p)
        scarcities.append(s)
    
    # Step 5: Global aggregation
    # Diversity: average across clusters
    global_diversity = float(np.mean(diversities)) if len(diversities) > 0 else 0.0
    
    # Proximity: max across clusters (worst-case / most borderline)
    global_proximity = float(np.max(proximities)) if len(proximities) > 0 else 0.0
    
    # Scarcity: average across clusters
    global_scarcity = float(np.mean(scarcities)) if len(scarcities) > 0 else 0.0
    
    return {
        "global": {
            "diversity": global_diversity,
            "proximity": global_proximity,
            "scarcity": global_scarcity,
        },
        "per_cluster": per_cluster,
        "details": {
            "test_pos_clusters": test_pos_clusters.tolist(),
            "test_neg_clusters": test_neg_clusters.tolist(),
            "train_pos_clusters": train_pos_clusters.tolist(),
            "train_neg_clusters": train_neg_clusters.tolist(),
        },
    }
