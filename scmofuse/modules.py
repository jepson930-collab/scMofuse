"""
scMoFuse.modules
================
Core neural network modules for single-cell multi-omics fusion.

Incorporates design elements from 6 recent top-tier methods:
- scECDA  (Bioinformatics 2025): shared encoder + differential attention (DiffFormer)
- BiCLUM  (PLOS Comp Bio): bilateral cell/feature contrastive + bilinear decoder
- scMUSCLE (Brief Bioinform 2026): multi-subspace contrastive + adaptive graph conv
- scPairing (Cell Rep Methods 2025): CLIP-style hyperspherical VAE + adversarial disc
- ACE    (Genomics Proteomics Bioinf 2025): mosaic alignment + InfoNCE
- scMMAE (Brief Bioinform 2025): masked cross-attention multimodal autoencoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Modality-specific encoder
# ---------------------------------------------------------------------------
class ModalityEncoder(nn.Module):
    """Per-modality MLP encoder with batch-norm / dropout.

    Maps high-dimensional omics features (RNA/ATAC/protein) into a common
    latent dimension. Handles the dimensional-imbalance problem:
    RNA ~ tens of thousands of genes, ATAC ~ hundreds of thousands of peaks,
    protein ~ a few hundred. A fixed latent dim removes the imbalance.
    """

    def __init__(self, input_dim, latent_dim=128, hidden_dims=(512, 256),
                 dropout=0.2, norm="batch"):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder (reconstruction) with modality-specific likelihood
# ---------------------------------------------------------------------------
class ModalityDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims=(256, 512),
                 dropout=0.1, norm="batch"):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(h))
            elif norm == "layer":
                layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Differential attention (inspired by scECDA's DiffFormer)
# ---------------------------------------------------------------------------
class DifferentialAttention(nn.Module):
    """Differential attention that emphasises modality-specific signal.

    Given two modality embeddings Z1, Z2, computes the difference and uses it
    as a gating signal so that the fusion pays more attention to the modality
    carrying more information for each cell.
    """

    def __init__(self, latent_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.norm3 = nn.LayerNorm(latent_dim)

    def forward(self, z1, z2, return_gate=False):
        # z1, z2: (B, D)
        z1 = self.norm1(z1)
        z2 = self.norm2(z2)
        # cross-attention: z1 attends to z2
        z1_seq = z1.unsqueeze(1)          # (B, 1, D)
        z2_seq = z2.unsqueeze(1)
        attn_out, _ = self.attn(z1_seq, z2_seq, z2_seq)
        attn_out = attn_out.squeeze(1)     # (B, D)
        # differential gating
        diff = torch.cat([z1 - z2, z1 + z2], dim=1)
        gate = self.gate(diff)             # (B, D)
        fused = attn_out * gate + z1 * (1 - gate)
        fused = self.norm3(fused + self.ffn(fused))
        if return_gate:
            return fused, gate
        return fused


# ---------------------------------------------------------------------------
# Masked cross-attention fusion (inspired by scMMAE)
# ---------------------------------------------------------------------------
class MaskedCrossAttentionFusion(nn.Module):
    """Fuses multiple modality embeddings with random masking.

    During training, each modality embedding is randomly dropped with prob p.
    The remaining modalities are fused via cross-attention so the model learns
    to rely on any subset of modalities (handles mosaic / missing data).
    """

    def __init__(self, latent_dim, num_heads=4, num_modalities=2,
                 mask_prob=0.15, dropout=0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.mask_prob = mask_prob
        self.norm = nn.LayerNorm(latent_dim)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads,
                                                dropout=dropout, batch_first=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_modalities + 1, latent_dim) * 0.02)

    def forward(self, embeddings, training=True, present=None):
        """embeddings: list of (B, D) tensors, one per modality.

        present: optional (B, M) boolean/0-1 tensor forcing the
        corresponding modality token to be masked (external mosaic masks).
        """
        B = embeddings[0].shape[0]
        tokens = torch.stack(embeddings, dim=1)          # (B, M, D)
        # random masking during training
        if training and self.mask_prob > 0:
            mask = (torch.rand(B, self.num_modalities, 1,
                                device=tokens.device) > self.mask_prob).float()
            tokens = tokens * mask
        # externally supplied missing modalities
        if present is not None:
            tokens = tokens * present.float().unsqueeze(2)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        x = torch.cat([cls, tokens], dim=1)              # (B, M+1, D)
        x = x + self.pos_embed
        x = self.norm(x)
        cls_out, _ = self.cross_attn(x[:, :1], x, x)     # (B, 1, D)
        return cls_out.squeeze(1)                         # (B, D)


# ---------------------------------------------------------------------------
# Bilinear decoder (inspired by BiCLUM) for feature-level reconstruction
# ---------------------------------------------------------------------------
class BilinearDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(latent_dim))
        self.bias = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, cell_emb, feature_emb):
        # cell_emb: (B, D), feature_emb: (F, D) -> (B, F)
        z = F.softplus(self.scale) * (cell_emb @ feature_emb.t()) + self.bias
        return z


# ---------------------------------------------------------------------------
# Adversarial discriminator (inspired by scPairing)
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, latent_dim, n_classes=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Feature encoder (per-feature embedding, needed for feature-level contrast)
# ---------------------------------------------------------------------------
class FeatureEncoder(nn.Module):
    """Encode every feature of a modality into a latent vector.

    Unlike the cell-level ModalityEncoder (input = per-cell feature vector),
    this module produces one embedding *per feature*. To stay tied to the data
    while supporting variable batch sizes, each feature is summarised within
    the batch by three differentiable statistics, mean, standard deviation and
    detection (fraction of present cells with nonzero value), then mapped by a
    small MLP to latent_dim. Output shape (n_features, latent_dim).

    For unpaired / mosaic batches, statistics are computed only over cells in
    which the modality was actually observed.
    """

    def __init__(self, latent_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x, present=None):
        # x: (B, F); present: (B,) bool, cells where modality observed
        if present is not None:
            m = present.float().unsqueeze(1)          # (B, 1)
            n = m.sum().clamp(min=1.0)
            mean = (x * m).sum(0) / n
            var = (((x - mean.detach().unsqueeze(0)) ** 2) * m).sum(0) / n
            det = ((x > 0).float() * m).sum(0) / n
        else:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            det = (x > 0).float().mean(0)
        stats = torch.stack([mean, var.sqrt().clamp(min=0), det], dim=1)  # (F, 3)
        return self.net(stats)


# ---------------------------------------------------------------------------
# Prototype / centroid based clustering head (inspired by scECDA/scMUSCLE)
# ---------------------------------------------------------------------------
class ClusteringHead(nn.Module):
    def __init__(self, latent_dim, n_clusters, alpha=1.0):
        super().__init__()
        self.centroids = nn.Parameter(torch.randn(n_clusters, latent_dim))
        nn.init.xavier_normal_(self.centroids)
        self.alpha = alpha

    def forward(self, z):
        # Student-t soft assignment
        dist = torch.sum((z.unsqueeze(1) - self.centroids) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / q.sum(1)).t()
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()
