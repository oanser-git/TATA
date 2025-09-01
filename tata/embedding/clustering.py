"""
Silhouette-guided k-means clustering for TATA.
Determines optimal k and assigns clusters to labels.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: Optional[List[int]] = None,
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple[int, float, KMeans]:
    """
    Find optimal number of clusters using silhouette score.
    
    Args:
        embeddings: Array of shape (n_samples, latent_dim).
        k_range: List of k values to try.
        random_state: Random seed.
        n_init: Number of initializations for k-means.
    
    Returns:
        best_k, best_silhouette, best_kmeans_model
    """
    if k_range is None:
        k_range = list(range(2, min(21, len(embeddings))))
    
    best_k = k_range[0]
    best_silhouette = -1.0
    best_model = None
    
    for k in k_range:
        if k >= len(embeddings):
            continue
        
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels_pred = kmeans.fit_predict(embeddings)
        
        score = silhouette_score(embeddings, labels_pred)
        
        if score > best_silhouette:
            best_silhouette = score
            best_k = k
            best_model = kmeans
    
    return best_k, best_silhouette, best_model


def evaluate_clustering(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    kmeans: KMeans,
) -> float:
    """
    Compute k-means accuracy: for each cluster, assign majority true label.
    
    Args:
        embeddings: Latent vectors.
        true_labels: Ground truth integer labels.
        kmeans: Fitted KMeans model.
    
    Returns:
        K-means accuracy (fraction of points in majority-label clusters).
    """
    cluster_labels = kmeans.labels_
    n_clusters = kmeans.n_clusters
    
    # Assign majority label to each cluster
    cluster_to_label = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        majority_label = int(np.bincount(true_labels[mask]).argmax())
        cluster_to_label[c] = majority_label
    
    # Predict labels based on cluster majority
    pred_labels = np.array([cluster_to_label[c] for c in cluster_labels])
    
    return float(accuracy_score(true_labels, pred_labels))


def build_cluster_metadata(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    kmeans: KMeans,
) -> Dict[str, Any]:
    """
    Build metadata for each cluster: centroid, majority label, member indices.
    
    Args:
        embeddings: Latent vectors (n_samples, latent_dim).
        true_labels: Ground truth integer labels.
        kmeans: Fitted KMeans model.
    
    Returns:
        Dict with keys:
            - n_clusters
            - centroids: np.ndarray (n_clusters, latent_dim)
            - cluster_majority_labels: List[int]
            - cluster_member_indices: Dict[int, List[int]]
            - silhouette_score: float
    """
    cluster_labels = kmeans.labels_
    n_clusters = kmeans.n_clusters
    
    cluster_majority_labels = []
    cluster_member_indices = {}
    
    for c in range(n_clusters):
        mask = cluster_labels == c
        indices = np.where(mask)[0].tolist()
        cluster_member_indices[c] = indices
        
        if len(indices) > 0:
            majority_label = int(np.bincount(true_labels[indices]).argmax())
        else:
            majority_label = -1
        cluster_majority_labels.append(majority_label)
    
    sil_score = silhouette_score(embeddings, cluster_labels) if n_clusters > 1 else 0.0
    
    return {
        "n_clusters": n_clusters,
        "centroids": kmeans.cluster_centers_,
        "cluster_majority_labels": cluster_majority_labels,
        "cluster_member_indices": cluster_member_indices,
        "silhouette_score": sil_score,
    }


def run_clustering_pipeline(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    k_range: Optional[List[int]] = None,
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple[int, KMeans, Dict[str, Any]]:
    """
    Full clustering pipeline: find optimal k, fit k-means, build metadata.
    
    Args:
        embeddings: Latent vectors from trained encoder.
        true_labels: Ground truth labels.
        k_range: Candidate k values.
        random_state: Random seed.
        n_init: k-means n_init parameter.
    
    Returns:
        best_k, fitted_kmeans, metadata_dict
    """
    best_k, best_sil, best_kmeans = find_optimal_k(
        embeddings, k_range=k_range, random_state=random_state, n_init=n_init
    )
    
    km_accuracy = evaluate_clustering(embeddings, true_labels, best_kmeans)
    metadata = build_cluster_metadata(embeddings, true_labels, best_kmeans)
    metadata["km_accuracy"] = km_accuracy
    metadata["best_k"] = best_k
    
    print(f"[Clustering] Optimal k={best_k}, Silhouette={best_sil:.4f}, KM-Accuracy={km_accuracy:.4f}")
    
    return best_k, best_kmeans, metadata
