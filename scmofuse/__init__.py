"""scMoFuse — Single-cell Multi-omics Fusion with Reliability Arbitration."""

from .model import scMoFuse
from .data import (
    MultiOmicsDataset, normalize_counts, binarize_atac,
    select_hvg, gene_activity_matrix, collate_multiomics,
)
from .evaluate import evaluate_all, clustering_metrics
from .utils import set_seed, make_mosaic_masks

__version__ = "1.0.0"
__all__ = [
    "scMoFuse",
    "MultiOmicsDataset", "normalize_counts", "binarize_atac",
    "select_hvg", "gene_activity_matrix", "collate_multiomics",
    "evaluate_all", "clustering_metrics",
    "set_seed", "make_mosaic_masks",
]
