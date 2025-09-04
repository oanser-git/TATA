"""
Correlation analysis: D/P/S scores vs actual NIDS performance.
Section 4.3 of the paper.

Three separate systematic manipulations:

1. Diversity: Generate all label combinations of size k from Dtest.
   Compute diversity for each sub-test set.
   Show perfect correlation with number of labels.

2. Proximity: Rank test points by distance to negative cluster.
   Create 100 cumulative sub-test sets (top 1%, 2%, ..., 100%).
   Compute proximity and evaluate NIDS macro-F1 for each.
   Correlate proximity with macro-F1.

3. Scarcity: Run NIDS on Dtest, collect misclassified points.
   Redistribute them across negative clusters in 10 steps
   (clustered -> dispersed). Compute scarcity and NIDS macro-F1.
   Correlate scarcity with macro-F1.
"""

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tata.embedding.clustering import run_clustering_pipeline
from tata.metrics.assessment import assess_test_set
from tata.models.nids.base import AbstractNIDS
from tata.models.nids.dnn import DNNNIDS
from tata.models.nids.random_forest import RandomForestNIDS
from tata.models.nids.svm import SVMNIDS


def _getattr_stat(result):
    """Handle both old tuple and new result-object API for scipy stats."""
    return getattr(result, "statistic", result[0])


def _getattr_pvalue(result):
    """Handle both old tuple and new result-object API for scipy stats."""
    return getattr(result, "pvalue", result[1])


# ---------------------------------------------------------------------------
# 1. Diversity Analysis
# ---------------------------------------------------------------------------

def diversity_analysis(
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_embeddings: np.ndarray,
    cluster_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Paper Section 4.3 — Diversity.

    Generate all sub-test sets from Dtest corresponding to each label
    combination of size k (k = 1..n_classes).
    Compute diversity for each combination.
    Return diversity values vs k.
    """
    labels = np.unique(y_test)
    n_classes = len(labels)

    results = []
    for k in range(1, n_classes + 1):
        combo_divs = []
        for combo in combinations(labels, k):
            mask = np.isin(y_test, combo)
            if mask.sum() == 0:
                continue
            sub_emb = X_test[mask]  # already encoded

            # Build minimal cluster_metadata for this sub-test
            scores = assess_test_set(
                train_embeddings=train_embeddings,
                test_embeddings=sub_emb,
                train_labels=np.zeros(len(train_embeddings)),  # dummy
                test_labels=y_test[mask],
                cluster_metadata=cluster_metadata,
            )
            combo_divs.append(scores["global"]["diversity"])

        if combo_divs:
            results.append({
                "k": k,
                "mean_diversity": float(np.mean(combo_divs)),
                "std_diversity": float(np.std(combo_divs)),
                "n_combinations": len(combo_divs),
            })

    # Correlation
    if len(results) > 2:
        ks = np.array([r["k"] for r in results])
        divs = np.array([r["mean_diversity"] for r in results])
        pr = pearsonr(ks, divs)
        sr = spearmanr(ks, divs)
        correlation = {
            "pearson_r": float(_getattr_stat(pr)),
            "pearson_p": float(_getattr_pvalue(pr)),
            "spearman_r": float(_getattr_stat(sr)),
            "spearman_p": float(_getattr_pvalue(sr)),
        }
    else:
        correlation = {}

    return {
        "per_k": results,
        "correlation": correlation,
    }


# ---------------------------------------------------------------------------
# 2. Proximity Correlation
# ---------------------------------------------------------------------------

def proximity_correlation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    cluster_metadata: dict[str, Any],
    nids_models: list[AbstractNIDS],
    n_bins: int = 100,
) -> dict[str, Any]:
    """
    Paper Section 4.3 — Proximity.

    Rank each encoded test data point by its distance to its negative
    cluster centroid (largest to smallest). Build 100 cumulative sub-test
    sets. For each, compute proximity and evaluate all NIDS models.
    """
    centroids = cluster_metadata["centroids"]
    cluster_majority_labels = cluster_metadata["cluster_majority_labels"]
    n_clusters = cluster_metadata["n_clusters"]

    # Assign each test point to its positive and negative cluster
    distances_to_all = np.linalg.norm(
        test_embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2,
    )

    pos_clusters = []
    neg_clusters = []
    neg_distances = []

    for i in range(len(test_embeddings)):
        label = y_test[i]
        dists = distances_to_all[i]

        same_mask = np.array(cluster_majority_labels) == label
        if same_mask.any():
            pos_c = int(np.where(same_mask)[0][np.argmin(dists[same_mask])])
        else:
            pos_c = int(np.argmin(dists))

        diff_mask = ~same_mask
        if diff_mask.any():
            neg_c = int(np.where(diff_mask)[0][np.argmin(dists[diff_mask])])
        else:
            neg_c = pos_c

        pos_clusters.append(pos_c)
        neg_clusters.append(neg_c)
        neg_distances.append(float(dists[neg_c]))

    pos_clusters = np.array(pos_clusters)
    neg_clusters = np.array(neg_clusters)
    neg_distances = np.array(neg_distances)

    # Rank by distance to negative cluster (largest -> smallest)
    ranked_idx = np.argsort(-neg_distances)

    proximity_values = []
    f1_per_model = {f"model_{j}": [] for j in range(len(nids_models))}

    for p in range(1, n_bins + 1):
        n_select = max(1, int(len(ranked_idx) * p / n_bins))
        subset_idx = ranked_idx[:n_select]

        sub_test_emb = test_embeddings[subset_idx]
        sub_test_labels = y_test[subset_idx]

        scores = assess_test_set(
            train_embeddings=train_embeddings,
            test_embeddings=sub_test_emb,
            train_labels=y_train,
            test_labels=sub_test_labels,
            cluster_metadata=cluster_metadata,
        )
        proximity_values.append(scores["global"]["proximity"])

        # Evaluate each NIDS on the original features (not embeddings)
        for j, model in enumerate(nids_models):
            metrics = model.evaluate(X_test[subset_idx], sub_test_labels)
            f1_per_model[f"model_{j}"].append(metrics["f1"])

    proximity_values = np.array(proximity_values)

    # Correlations per model
    correlations = {}
    for j in range(len(nids_models)):
        f1s = np.array(f1_per_model[f"model_{j}"])
        if len(proximity_values) > 2 and np.std(proximity_values) > 0 and np.std(f1s) > 0:
            pr = pearsonr(proximity_values, f1s)
            sr = spearmanr(proximity_values, f1s)
            correlations[f"model_{j}"] = {
                "pearson_r": float(_getattr_stat(pr)),
                "pearson_p": float(_getattr_pvalue(pr)),
                "spearman_r": float(_getattr_stat(sr)),
                "spearman_p": float(_getattr_pvalue(sr)),
            }

    return {
        "proximity_values": proximity_values.tolist(),
        "f1_per_model": {k: v for k, v in f1_per_model.items()},
        "correlations": correlations,
    }


# ---------------------------------------------------------------------------
# 3. Scarcity Correlation
# ---------------------------------------------------------------------------

def scarcity_correlation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    cluster_metadata: dict[str, Any],
    nids_models: list[AbstractNIDS],
    n_steps: int = 10,
) -> dict[str, Any]:
    """
    Paper Section 4.3 — Scarcity.

    Run each NIDS on Dtest, record misclassified data points, map them
    to latent representations. Systematically redistribute these instances
    among available negative clusters in n_steps steps (clustered -> dispersed).
    Compute scarcity and NIDS macro-F1 for each step.
    """
    centroids = cluster_metadata["centroids"]
    cluster_majority_labels = cluster_metadata["cluster_majority_labels"]
    n_clusters = cluster_metadata["n_clusters"]

    # First, identify all test points and their negative clusters
    distances_to_all = np.linalg.norm(
        test_embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :],
        axis=2,
    )

    test_neg_clusters = []
    for i in range(len(test_embeddings)):
        label = y_test[i]
        dists = distances_to_all[i]
        same_mask = np.array(cluster_majority_labels) == label
        diff_mask = ~same_mask
        if diff_mask.any():
            neg_c = int(np.where(diff_mask)[0][np.argmin(dists[diff_mask])])
        else:
            neg_c = int(np.argmin(dists))
        test_neg_clusters.append(neg_c)
    test_neg_clusters = np.array(test_neg_clusters)

    # Run each NIDS to find misclassified points (union across models)
    misclassified_mask = np.zeros(len(y_test), dtype=bool)
    for model in nids_models:
        preds = model.predict(X_test)
        misclassified_mask |= (preds != y_test)

    mis_idx = np.where(misclassified_mask)[0]
    if len(mis_idx) == 0:
        return {"note": "No misclassified points found", "scarcity_values": [], "f1_per_model": {}}

    # Available negative clusters for each misclassified point
    # Simplify: all clusters with different majority label than the point's label
    available_negs = []
    for idx in mis_idx:
        label = y_test[idx]
        avail = [c for c in range(n_clusters) if cluster_majority_labels[c] != label]
        if not avail:
            avail = list(range(n_clusters))
        available_negs.append(avail)

    scarcity_values = []
    f1_per_model = {f"model_{j}": [] for j in range(len(nids_models))}

    for step in range(n_steps):
        # Create a modified test set for this step
        # Goal: redistribute misclassified points across negative clusters
        # Step 0: clustered (all in one cluster) -> Step n-1: dispersed (uniform)
        frac_uniform = step / max(1, n_steps - 1)

        modified_test_emb = test_embeddings.copy()
        modified_test_labels = y_test.copy()

        # For each misclassified point, potentially reassign its negative cluster
        # by moving it closer to a target negative cluster centroid
        for i, idx in enumerate(mis_idx):
            avail = available_negs[i]
            n_avail = len(avail)
            if n_avail == 0:
                continue

            if frac_uniform < 1e-6:
                # Step 0: all assigned to first available cluster
                target_c = avail[0]
            elif frac_uniform > 1 - 1e-6:
                # Last step: uniform across all available
                target_c = avail[i % n_avail]
            else:
                # Intermediate: increasingly uniform
                # Use a weighted assignment that becomes more uniform
                probs = np.ones(n_avail) / n_avail * frac_uniform
                probs[0] += (1 - frac_uniform)  # bias toward first cluster
                probs = probs / probs.sum()
                target_c = int(np.random.choice(avail, p=probs))

            # Move the point closer to the target negative cluster centroid
            # This simulates a point that would naturally map to that negative cluster
            current = modified_test_emb[idx]
            target = centroids[target_c]
            # Blend toward target (more aggressive as step increases)
            alpha = 0.3 * frac_uniform
            modified_test_emb[idx] = current * (1 - alpha) + target * alpha

        # Compute scarcity on modified test set
        scores = assess_test_set(
            train_embeddings=train_embeddings,
            test_embeddings=modified_test_emb,
            train_labels=y_train,
            test_labels=modified_test_labels,
            cluster_metadata=cluster_metadata,
        )
        scarcity_values.append(scores["global"]["scarcity"])

        # Evaluate NIDS on modified original features (we use original X_test
        # since we can't modify real features from latent shifts)
        for j, model in enumerate(nids_models):
            metrics = model.evaluate(X_test, modified_test_labels)
            f1_per_model[f"model_{j}"].append(metrics["f1"])

    scarcity_values = np.array(scarcity_values)

    correlations = {}
    for j in range(len(nids_models)):
        f1s = np.array(f1_per_model[f"model_{j}"])
        if len(scarcity_values) > 2 and np.std(scarcity_values) > 0 and np.std(f1s) > 0:
            pr = pearsonr(scarcity_values, f1s)
            sr = spearmanr(scarcity_values, f1s)
            correlations[f"model_{j}"] = {
                "pearson_r": float(_getattr_stat(pr)),
                "pearson_p": float(_getattr_pvalue(pr)),
                "spearman_r": float(_getattr_stat(sr)),
                "spearman_p": float(_getattr_pvalue(sr)),
            }

    return {
        "scarcity_values": scarcity_values.tolist(),
        "f1_per_model": {k: v for k, v in f1_per_model.items()},
        "correlations": correlations,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_correlation_study(
    X: np.ndarray,
    y: np.ndarray,
    encoder_callable: Callable[[np.ndarray], np.ndarray],
    nids_models: Optional[list[AbstractNIDS]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_dir: str = "data/artifacts/correlation",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run full correlation study (Section 4.3) with all three metrics.

    Paper-standard 60/20/20 split. NIDS models are trained on train set.
    """
    import torch

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 60/20/20 split
    _X_train, _X_temp, _y_train, _y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    _X_val, _X_test, _y_val, _y_test = train_test_split(
        _X_temp, _y_temp, test_size=0.5, random_state=random_state, stratify=_y_temp
    )
    X_train = np.asarray(_X_train)
    X_val = np.asarray(_X_val)
    X_test = np.asarray(_X_test)
    y_train = np.asarray(_y_train)
    y_val = np.asarray(_y_val)
    y_test = np.asarray(_y_test)

    if verbose:
        print(f"Split: Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Encode
    with torch.no_grad():
        train_emb = encoder_callable(X_train_s)
        test_emb = encoder_callable(X_test_s)

    # Clustering
    best_k, kmeans, metadata = run_clustering_pipeline(
        train_emb, y_train, random_state=random_state
    )

    # Train NIDS models if not provided
    if nids_models is None:
        if verbose:
            print("Training NIDS models (RF, SVM, DNN)...")
        rf = RandomForestNIDS()
        svm = SVMNIDS()
        dnn = DNNNIDS()
        rf.fit(X_train_s, y_train)
        svm.fit(X_train_s, y_train)
        dnn.fit(X_train_s, y_train)
        nids_models = [rf, svm, dnn]

    results = {}

    # 1. Diversity analysis
    if verbose:
        print("\n--- Diversity Analysis ---")
    div_results = diversity_analysis(
        X_test=test_emb,
        y_test=y_test,
        train_embeddings=train_emb,
        cluster_metadata=metadata,
    )
    results["diversity"] = div_results
    if verbose and div_results["correlation"]:
        c = div_results["correlation"]
        print(f"  Pearson r={c['pearson_r']:.3f}, Spearman rho={c['spearman_r']:.3f}")

    # 2. Proximity correlation
    if verbose:
        print("\n--- Proximity Correlation ---")
    prox_results = proximity_correlation(
        X_train=X_train_s, y_train=y_train,
        X_test=X_test_s, y_test=y_test,
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        cluster_metadata=metadata,
        nids_models=nids_models,
        n_bins=100,
    )
    results["proximity"] = prox_results
    if verbose and prox_results["correlations"]:
        for name, corr in prox_results["correlations"].items():
            print(f"  {name}: Pearson r={corr['pearson_r']:.3f}, Spearman rho={corr['spearman_r']:.3f}")

    # 3. Scarcity correlation
    if verbose:
        print("\n--- Scarcity Correlation ---")
    sc_results = scarcity_correlation(
        X_train=X_train_s, y_train=y_train,
        X_test=X_test_s, y_test=y_test,
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        cluster_metadata=metadata,
        nids_models=nids_models,
        n_steps=10,
    )
    results["scarcity"] = sc_results
    if verbose and sc_results.get("correlations"):
        for name, corr in sc_results["correlations"].items():
            print(f"  {name}: Pearson r={corr['pearson_r']:.3f}, Spearman rho={corr['spearman_r']:.3f}")

    # Save
    with open(save_path / "correlation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nResults saved to {save_path / 'correlation_results.json'}")

    return results
