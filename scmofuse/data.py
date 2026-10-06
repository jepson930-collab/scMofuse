"""
scMoFuse.data
=============
Data loading and preprocessing for single-cell multi-omics fusion.

Supports:
- scRNA-seq + scATAC-seq (10x Multiome)
- scRNA-seq + protein (CITE-seq)
- Paired / unpaired / mosaic scenarios
"""

import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset


def normalize_counts(counts, scale_factor=1e4):
    """Library-size normalisation + log1p (RNA / protein)."""
    if sp.issparse(counts):
        counts = counts.toarray()
    counts = counts.astype(np.float32)
    lib = counts.sum(1, keepdims=True)
    lib = np.where(lib == 0, 1, lib)
    norm = counts / lib * scale_factor
    return np.log1p(norm)


def binarize_atac(counts, threshold=0.5):
    """Binarise ATAC peak counts to 0/1."""
    if sp.issparse(counts):
        counts = counts.toarray()
    return (counts > threshold).astype(np.float32)


def select_hvg(adata_expr, n_top=2000):
    """Select highly variable genes by simple variance ranking."""
    if sp.issparse(adata_expr):
        expr = adata_expr.toarray()
    else:
        expr = np.asarray(adata_expr)
    mean = expr.mean(0)
    var = expr.var(0)
    dispersion = var / (mean + 1e-6)
    idx = np.argsort(dispersion)[::-1][:n_top]
    return np.sort(idx)


def gene_activity_matrix(atac_counts, peak_gene_map, n_genes):
    """Convert ATAC peak counts to gene activity scores.

    peak_gene_map: dict peak_index -> list of (gene_index, weight)
    Returns (n_cells, n_genes) dense matrix.
    """
    if sp.issparse(atac_counts):
        atac = atac_counts.toarray()
    else:
        atac = np.asarray(atac_counts)
    activity = np.zeros((atac.shape[0], n_genes), dtype=np.float32)
    for peak_idx, genes in peak_gene_map.items():
        for gene_idx, weight in genes:
            activity[:, gene_idx] += atac[:, peak_idx] * weight
    return activity


class MultiOmicsDataset(Dataset):
    """Dataset for paired/unpaired multi-omics data.

    Parameters
    ----------
    modalities : dict
        e.g. {"rna": rna_matrix, "atac": atac_matrix}
    labels : np.ndarray or None
        Cell-type labels (for evaluation only).
    masks : dict or None
        Boolean mask per modality indicating which cells are present
        (for mosaic integration). If None, all cells present in all modalities.
    """

    def __init__(self, modalities, labels=None, masks=None, metadata=None):
        self.modalities = {}
        for k, v in modalities.items():
            if sp.issparse(v):
                v = v.toarray()
            self.modalities[k] = v.astype(np.float32)
        self.n_cells = next(iter(self.modalities.values())).shape[0]
        self.labels = labels
        if masks is None:
            self.masks = {k: np.ones(self.n_cells, dtype=bool)
                          for k in self.modalities}
        else:
            self.masks = masks
        # optional per-cell integer/category metadata, e.g. donor, lane
        self.metadata = metadata

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        sample = {"idx": idx}
        for k, mat in self.modalities.items():
            sample[k] = mat[idx]
            sample[k + "_mask"] = self.masks[k][idx]
        if self.labels is not None:
            sample["label"] = self.labels[idx]
        if self.metadata is not None:
            for name, arr in self.metadata.items():
                sample[name] = np.asarray(arr)[idx]
        return sample


def collate_multiomics(batch):
    """Collate function for MultiOmicsDataset."""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], (int, np.integer, bool)):
            out[k] = torch.tensor(vals)
        else:
            out[k] = torch.tensor(np.stack(vals))
    return out


import torch  # noqa: E402  (needed for collate)
