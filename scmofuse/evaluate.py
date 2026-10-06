"""
scMoFuse.evaluate
=================
Evaluation metrics for single-cell multi-omics integration.

- Clustering: ARI, NMI, FMI
- Integration:  graph connectivity, kBET / LISI (simplified)
"""

import numpy as np
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             fowlkes_mallows_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors


def clustering_metrics(labels_true, labels_pred):
    return {
        "ARI": adjusted_rand_score(labels_true, labels_pred),
        "NMI": normalized_mutual_info_score(labels_true, labels_pred),
        "FMI": fowlkes_mallows_score(labels_true, labels_pred),
    }


def silhouette(embeddings, labels):
    try:
        return silhouette_score(embeddings, labels)
    except Exception:
        return float("nan")


def graph_connectivity(embeddings, labels, k=15):
    """Fraction of k-NN neighbours that share the same cell type.

    Higher is better; measures how well the latent space separates cell types.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, idx = nn.kneighbors(embeddings)
    idx = idx[:, 1:]
    same = (labels[idx] == labels[:, None]).mean()
    return float(same)


def modality_mixing_score(embeddings, modality_labels, k=15):
    """Fraction of k-NN from a different modality (higher = better mixing)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, idx = nn.kneighbors(embeddings)
    idx = idx[:, 1:]
    diff = (modality_labels[idx] != modality_labels[:, None]).mean()
    return float(diff)


def evaluate_all(embeddings, cell_labels, modality_labels=None, k=15):
    metrics = {}
    # clustering via k-means on embeddings
    from sklearn.cluster import KMeans
    n_clusters = len(np.unique(cell_labels))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    pred = km.fit_predict(embeddings)
    metrics.update(clustering_metrics(cell_labels, pred))
    metrics["silhouette"] = silhouette(embeddings, cell_labels)
    metrics["graph_connectivity"] = graph_connectivity(embeddings, cell_labels, k)
    if modality_labels is not None:
        metrics["modality_mixing"] = modality_mixing_score(
            embeddings, modality_labels, k)
    return metrics
