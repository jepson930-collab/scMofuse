"""
scMoFuse.losses
===============
Loss functions for multi-omics fusion.

- InfoNCE / CLIP-style contrastive alignment
- Cross-level cell- and feature-level contrastive alignment
- Reconstruction loss with modality-specific likelihoods
- Divergence-based clustering loss / KL on soft assignments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# InfoNCE contrastive loss (CLIP-style)
# ---------------------------------------------------------------------------
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07, learnable=True):
        super().__init__()
        if learnable:
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))
        else:
            self.register_buffer("logit_scale",
                                 torch.ones([]) * math.log(1 / temperature))

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logit_scale = self.logit_scale.exp()
        logits = z1 @ z2.t() * logit_scale
        labels = torch.arange(z1.shape[0], device=z1.device)
        loss = (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.t(), labels)) / 2
        return loss


# ---------------------------------------------------------------------------
# Cross-level contrastive loss (cell-level + feature-level)
# ---------------------------------------------------------------------------
class CrossLevelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature

    @staticmethod
    def _loss(z_i, z_j, tau):
        if z_i.shape[0] == 0:
            return torch.tensor(0.0, device=z_i.device)
        sim = (F.normalize(z_i, dim=1) @ F.normalize(z_j, dim=1).t()) / tau
        labels = torch.arange(z_i.shape[0], device=z_i.device)
        return (F.cross_entropy(sim, labels) +
                F.cross_entropy(sim.t(), labels)) / 2

    def forward(self, cell_z1, cell_z2, cell_mask,
                feat_z1=None, feat_z2=None):
        loss_cell = self._loss(cell_z1[cell_mask], cell_z2[cell_mask], self.tau)
        loss_feat = torch.tensor(0.0, device=cell_z1.device)
        if feat_z1 is not None and feat_z2 is not None:
            loss_feat = self._loss(feat_z1, feat_z2, self.tau)
        return loss_cell, loss_feat


# ---------------------------------------------------------------------------
# Feature-level contrastive loss with explicit anchor-positive index pairs
# ---------------------------------------------------------------------------
class PairedFeatureLoss(nn.Module):
    """InfoNCE over features with known correspondences.

    Feature sets of two modalities have different sizes, so the diagonal
    assumption of cell-level InfoNCE does not hold. Instead, explicit pairs
    (i, j) are supplied: feature i of modality a is the anchor and feature j
    of modality b is its positive; all other features of b are negatives.
    Pairs come from genomic priors, protein marker to gene mapping for
    RNA+protein, peak-to-gene links for RNA+ATAC, or unsupervised feature
    MNN matches. Symmetric (both anchor directions) cross entropy.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, emb_a, emb_b, pairs):
        # emb_a (Na, D), emb_b (Nb, D), pairs (K, 2) LongTensor
        if pairs is None or pairs.shape[0] == 0:
            return torch.tensor(0.0, device=emb_a.device)
        emb_a = F.normalize(emb_a, dim=1)
        emb_b = F.normalize(emb_b, dim=1)
        ia = pairs[:, 0]
        ib = pairs[:, 1]
        logits_ab = (emb_a @ emb_b.t()) / self.tau
        loss_a = F.cross_entropy(logits_ab[ia], ib)
        logits_ba = (emb_b @ emb_a.t()) / self.tau
        loss_b = F.cross_entropy(logits_ba[ib], ia)
        return (loss_a + loss_b) / 2


# ---------------------------------------------------------------------------
# Modality-specific reconstruction losses
# ---------------------------------------------------------------------------
class ReconstructionLoss(nn.Module):
    """Supports NB (RNA/protein), Bernoulli (ATAC), Gaussian (other)."""

    def __init__(self, modality_type="rna"):
        super().__init__()
        self.modality_type = modality_type

    def forward(self, recon, target, library_size=None, dispersion=None):
        if self.modality_type in ("rna", "protein"):
            return self._neg_binomial(recon, target, library_size, dispersion)
        elif self.modality_type == "atac":
            return self._bernoulli(recon, target)
        else:
            return F.mse_loss(recon, target)

    def _neg_binomial(self, recon, target, library_size, dispersion):
        if library_size is None:
            library_size = target.sum(1, keepdim=True)
        mu = F.softmax(recon, dim=1) * library_size
        if dispersion is None:
            dispersion = torch.ones_like(mu)
        # simplified NB negative log-likelihood
        eps = 1e-10
        theta = dispersion.exp().clamp(max=1e6)
        t1 = torch.lgamma(target + theta + eps) - torch.lgamma(theta + eps) - \
             torch.lgamma(target + 1.0)
        t2 = -(target + theta) * torch.log1p(mu / theta + eps)
        t3 = target * (torch.log(mu + eps) - torch.log(theta + eps))
        nll = -(t1 + t2 + t3).sum(1).mean()
        return nll

    def _bernoulli(self, recon, target):
        p = torch.sigmoid(recon)
        return F.binary_cross_entropy(p, target, reduction="none").sum(1).mean()


# ---------------------------------------------------------------------------
# Clustering losses
# ---------------------------------------------------------------------------
class KLDivLoss(nn.Module):
    """KL divergence between soft assignment q and target p."""

    def forward(self, q, p):
        return F.kl_div(q.log(), p, reduction="batchmean")


class DivergenceClusteringLoss(nn.Module):
    """Divergence-based clustering loss (Cauchy-Schwarz divergence)."""

    def __init__(self, num_cluster, rel_sigma=0.15, device="cpu"):
        super().__init__()
        self.num_cluster = num_cluster
        self.rel_sigma = rel_sigma
        self.device = device

    def forward(self, logist, hidden):
        hidden_kernel = self._kernel(hidden)
        l1 = self._d_cs(logist, hidden_kernel)
        l2 = 2 / (logist.size(0) * (logist.size(0) - 1)) * self._triu(logist @ logist.t())
        eye = torch.eye(self.num_cluster, device=self.device)
        m = torch.exp(-self._cdist(logist, eye))
        l3 = self._d_cs(m, hidden_kernel)
        return l1 + l2 + l3

    def _kernel(self, x):
        dist = self._cdist(x, x)
        dist = F.relu(dist)
        sigma2 = self.rel_sigma * torch.median(dist).detach()
        sigma2 = sigma2.clamp(min=1e-3)
        return torch.exp(-dist / (2 * sigma2))

    def _d_cs(self, A, K):
        nom = A.t() @ K @ A
        dnom = torch.diagonal(nom).unsqueeze(-1) @ torch.diagonal(nom).unsqueeze(0)
        nom = nom.clamp(min=1e-9)
        dnom = dnom.clamp(min=1e-9)
        n = self.num_cluster
        return 2 / (n * (n - 1)) * self._triu(nom / torch.sqrt(dnom))

    @staticmethod
    def _cdist(X, Y):
        return torch.cdist(X, Y) ** 2

    @staticmethod
    def _triu(X):
        return torch.triu(X, diagonal=1).sum()
