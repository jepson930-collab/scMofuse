"""
scMoFuse.utils
==============
Miscellaneous utilities.
"""

import numpy as np
import torch
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(d, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


def make_mosaic_masks(n_cells, n_modalities, missing_frac=0.1, seed=42):
    """Create boolean masks for mosaic integration.

    Each cell has a `missing_frac` chance of missing each modality.
    Guarantees every cell has at least one modality.
    """
    rng = np.random.default_rng(seed)
    masks = np.ones((n_cells, n_modalities), dtype=bool)
    for i in range(n_cells):
        for j in range(n_modalities):
            if rng.random() < missing_frac:
                masks[i, j] = False
        # ensure at least one modality present
        if masks[i].sum() == 0:
            masks[i, rng.integers(n_modalities)] = True
    return masks
