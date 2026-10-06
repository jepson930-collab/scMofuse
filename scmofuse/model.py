"""
scMoFuse.model
==============
Main model: scMoFuse — Single-cell Multi-omics Fusion with Adaptive
Cross-attention and Bilateral contrastive learning.

Design rationale (challenges -> modules):
  P1 dimension imbalance          -> per-modality encoders to a common dim
  P2 mosaic / missing modalities  -> masked cross-attention, mask-aware losses
  P3 conflicting modality signals -> per-cell differential attention gates
  P4 count/binary heterogeneity   -> modality-specific likelihoods when raw
                                     counts are available; Gaussian/MSE fallback
                                     for pre-normalised matrices
  P5 modality / batch bias        -> adversarial discriminators trained with a
                                     separate optimiser (min-max GAN)
  P6 cell- vs feature-level align -> bilateral contrast: cell-level InfoNCE plus
                                     feature-level InfoNCE over explicit
                                     correspondences (protein-gene / peak-gene),
                                     with per-feature embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    ModalityEncoder, ModalityDecoder, DifferentialAttention,
    MaskedCrossAttentionFusion, Discriminator, ClusteringHead, FeatureEncoder,
)
from .losses import (
    InfoNCE, BilateralContrastiveLoss, ReconstructionLoss, KLDivLoss, DDCLoss,
    PairedFeatureLoss,
)


class scMoFuse(nn.Module):
    def __init__(
        self,
        input_dims,                 # dict: modality -> feature dim
        latent_dim=128,
        n_clusters=10,
        hidden_dims=(512, 256),
        decoder_hidden=(256, 512),
        dropout=0.2,
        temperature=0.07,
        mask_prob=0.15,
        modality_types=None,        # dict: modality -> "rna"/"atac"/"protein"/"other"
        n_batches=1,
        use_adversarial=True,
        use_bilateral=True,
        use_clustering=True,
        use_feature_contrast=True,
        use_differential=True,
        use_masked_fusion=True,
        device="cpu",
    ):
        super().__init__()
        self.modalities = list(input_dims.keys())
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.n_batches = n_batches
        self.use_adversarial = use_adversarial
        self.use_bilateral = use_bilateral
        self.use_clustering = use_clustering
        self.use_feature_contrast = use_feature_contrast
        self.use_differential = use_differential
        self.use_masked_fusion = use_masked_fusion
        self.device = device

        if modality_types is None:
            modality_types = {m: "other" for m in self.modalities}
        self.modality_types = modality_types

        # --- cell-level encoders / decoders ---
        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(input_dims[m], latent_dim, hidden_dims, dropout)
            for m in self.modalities
        })
        self.decoders = nn.ModuleDict({
            m: ModalityDecoder(latent_dim, input_dims[m], decoder_hidden, dropout)
            for m in self.modalities
        })

        # --- feature-level encoders (one embedding per feature) ---
        if use_feature_contrast:
            self.feature_encoders = nn.ModuleDict({
                m: FeatureEncoder(latent_dim, dropout=dropout)
                for m in self.modalities
            })
            self.register_buffer(
                "feature_pairs",
                torch.zeros(0, 2, dtype=torch.long))

        # --- cross-attention fusion (handles mosaic via masking) ---
        self.diff_attn = DifferentialAttention(latent_dim, num_heads=4, dropout=dropout)
        self.fusion = MaskedCrossAttentionFusion(
            latent_dim, num_heads=4,
            num_modalities=len(self.modalities),
            mask_prob=mask_prob, dropout=dropout,
        )

        # --- clustering head ---
        if use_clustering:
            self.cluster_head = ClusteringHead(latent_dim, n_clusters)
            # gate for delayed activation (set False during warmup to avoid
            # drifting random centroids; Trainer may flip it mid-training)
            self.cluster_enabled = True

        # --- losses ---
        self.contrastive = InfoNCE(temperature=temperature)
        self.bilateral = BilateralContrastiveLoss(temperature=temperature)
        self.feature_loss = PairedFeatureLoss(temperature=0.1)
        self.recon_losses = nn.ModuleDict({
            m: ReconstructionLoss(modality_types[m]) for m in self.modalities
        })
        self.kl_loss = KLDivLoss()
        self.ddc_loss = DDCLoss(n_clusters, device=device)

        # --- adversarial discriminators ---
        if use_adversarial:
            self.modality_disc = Discriminator(latent_dim, n_classes=len(self.modalities))
            if n_batches > 1:
                self.batch_disc = Discriminator(latent_dim, n_classes=n_batches)

    # ------------------------------------------------------------------
    def set_feature_pairs(self, pairs):
        """Supply feature correspondences as an (K, 2) int array [idx_mod0, idx_mod1]."""
        if not self.use_feature_contrast:
            return
        self.feature_pairs = torch.as_tensor(pairs, dtype=torch.long)

    def encode(self, x_dict, masks=None):
        """Encode each modality (missing cells are zeroed)."""
        z = {}
        for m in self.modalities:
            x = x_dict[m]
            if masks is not None and m in masks:
                x = x * masks[m].float().unsqueeze(1)
            z[m] = self.encoders[m](x)
        return z

    def fuse(self, z_dict, training=True, return_gate=False, masks=None):
        """Fuse modality embeddings. Returns joint representation.

        masks: optional dict modality -> (B,) boolean presence flags for
        mosaic integration; cells with one modality are represented by the
        observed modality alone.
        """
        emb_list = [z_dict[m] for m in self.modalities]
        gate = None
        present = None
        if masks is not None:
            present = torch.stack(
                [masks[m].bool() for m in self.modalities], dim=1)

        if self.use_differential:
            if len(emb_list) == 2:
                e1, e2 = emb_list[0], emb_list[1]
                if masks is not None:
                    # replace a missing view by the observed view so the
                    # differential block degenerates to identity
                    p1 = masks[self.modalities[0]].bool().unsqueeze(1)
                    p2 = masks[self.modalities[1]].bool().unsqueeze(1)
                    e1u = torch.where(p1, e1, e2)
                    e2u = torch.where(p2, e2, e1u)
                else:
                    e1u, e2u = e1, e2
                if return_gate:
                    fused, gate = self.diff_attn(
                        e1u, e2u, return_gate=True)
                else:
                    fused = self.diff_attn(e1u, e2u)
            else:
                fused = emb_list[0]
                for i in range(1, len(emb_list)):
                    fused = self.diff_attn(fused, emb_list[i])
        else:
            # ablation: plain equal-weight averaging instead of differential
            fused = torch.stack(emb_list, dim=0).mean(dim=0)

        if self.use_masked_fusion:
            masked_joint = self.fusion(
                emb_list, training=training, present=present)
            joint_full = F.normalize(masked_joint + 0.5 * fused, dim=1)
        else:
            joint_full = F.normalize(fused, dim=1)

        if masks is None or len(self.modalities) != 2:
            joint = joint_full
        else:
            # mosaic: single-modality cells are represented by that modality
            p1 = masks[self.modalities[0]].bool()
            p2 = masks[self.modalities[1]].bool()
            only1, only2 = p1 & ~p2, p2 & ~p1
            joint = torch.zeros_like(joint_full)
            both = p1 & p2
            joint[both] = joint_full[both]
            joint[only1] = F.normalize(emb_list[0][only1], dim=1)
            joint[only2] = F.normalize(emb_list[1][only2], dim=1)

        if return_gate:
            return joint, gate
        return joint

    def encode_features(self, x_dict, masks=None):
        """Per-feature embeddings for every modality (feature-level contrast)."""
        feats = {}
        for m in self.modalities:
            present = masks[m] if (masks is not None and m in masks) else None
            feats[m] = self.feature_encoders[m](x_dict[m], present)
        return feats

    def _present(self, masks, m):
        if masks is not None and m in masks:
            return masks[m]
        return None

    def _modality_disc_logits(self, z_dict, masks):
        """Gather discriminator predictions over observed cells only."""
        logits, labels = [], []
        for mi, m in enumerate(self.modalities):
            present = self._present(masks, m)
            if present is not None:
                if present.sum() == 0:
                    continue
                logits.append(self.modality_disc(z_dict[m][present]))
                labels.append(torch.full(
                    (present.sum().item(),), mi, dtype=torch.long,
                    device=z_dict[m].device))
            else:
                logits.append(self.modality_disc(z_dict[m]))
                labels.append(torch.full(
                    (z_dict[m].shape[0],), mi, dtype=torch.long,
                    device=z_dict[m].device))
        return torch.cat(logits, dim=0), torch.cat(labels, dim=0)

    def forward(self, x_dict, masks=None, batch_indices=None):
        z_dict = self.encode(x_dict, masks)
        joint, gate = self.fuse(z_dict, training=self.training,
                                return_gate=True, masks=masks)

        # reconstruction
        recon = {m: self.decoders[m](joint) for m in self.modalities}

        # clustering
        q = self.cluster_head(joint) if self.use_clustering else None

        # modality embeddings for contrastive
        z_norm = {m: F.normalize(z_dict[m], dim=1) for m in self.modalities}

        # feature embeddings
        feat = self.encode_features(x_dict, masks) \
            if self.use_feature_contrast else None

        out = {
            "joint": joint,
            "z": z_dict,
            "z_norm": z_norm,
            "recon": recon,
            "q": q,
            "gate": gate,
            "feat": feat,
        }

        if self.use_adversarial and self.training:
            mod_logits, mod_labels = self._modality_disc_logits(z_dict, masks)
            out["modality_logits"] = mod_logits
            out["modality_labels"] = mod_labels
            if self.n_batches > 1 and batch_indices is not None:
                out["batch_logits"] = self.batch_disc(joint)
        return out

    # ------------------------------------------------------------------
    def discriminator_loss(self, x_dict, masks=None, batch_indices=None):
        """Discriminator-side objective (minimised by the disc optimiser).

        Runs the encoders in eval-style fashion but keeps gradients to the
        encoders detached; only discriminator parameters receive gradients.
        """
        z_dict = self.encode(x_dict, masks)
        with torch.no_grad():
            pass  # encoders must run, but we detach their outputs below
        z_detached = {m: z_dict[m].detach() for m in self.modalities}
        logits, labels = self._modality_disc_logits(z_detached, masks)
        loss = F.cross_entropy(logits, labels)
        if self.n_batches > 1 and batch_indices is not None:
            joint = self.fuse(z_detached, training=False)
            loss = loss + F.cross_entropy(
                self.batch_disc(joint), batch_indices)
        return loss

    # ------------------------------------------------------------------
    def compute_loss(self, out, x_dict, masks=None, batch_indices=None,
                     weights=None):
        """Compute the full task (encoder-side) loss."""
        if weights is None:
            weights = {}
        w = {
            "recon": weights.get("recon", 1.0),
            "contrast": weights.get("contrast", 1.0),
            "bilateral": weights.get("bilateral", 1.0),
            "feature": weights.get("feature", 1.0),
            "kl": weights.get("kl", 0.1),
            "ddc": weights.get("ddc", 0.1),
            "adv": weights.get("adv", 1.0),
        }

        joint = out["joint"]
        recon = out["recon"]
        z_norm = out["z_norm"]
        q = out["q"]
        losses = {}

        # 1. reconstruction (modality-specific likelihood, mask-aware)
        recon_loss = 0.0
        for m in self.modalities:
            if masks is not None and m in masks:
                mask = masks[m].float()
                r = self.recon_losses[m](recon[m], x_dict[m])
                recon_loss = recon_loss + (r * mask).mean()
            else:
                recon_loss = recon_loss + self.recon_losses[m](recon[m], x_dict[m])
        losses["recon"] = recon_loss / len(self.modalities)

        # 2. pairwise InfoNCE contrastive between modalities (co-present
        # cells only when mosaic masks are given)
        contrast_loss = 0.0
        mods = self.modalities
        n_pairs = 0
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                if masks is not None:
                    paired = masks[mods[i]].bool() & masks[mods[j]].bool()
                    if paired.sum() <= 1:
                        continue
                    contrast_loss = contrast_loss + self.contrastive(
                        z_norm[mods[i]][paired], z_norm[mods[j]][paired])
                else:
                    contrast_loss = contrast_loss + self.contrastive(
                        z_norm[mods[i]], z_norm[mods[j]])
                n_pairs += 1
        losses["contrast"] = contrast_loss / max(n_pairs, 1)

        # 3. bilateral cell-level contrastive (only for co-present cells)
        if self.use_bilateral and masks is not None:
            bilateral_loss = 0.0
            for i in range(len(mods)):
                for j in range(i + 1, len(mods)):
                    paired = masks[mods[i]] & masks[mods[j]]
                    if paired.sum() > 1:
                        cell_loss, _ = self.bilateral(
                            z_norm[mods[i]], z_norm[mods[j]], paired)
                        bilateral_loss = bilateral_loss + cell_loss
            losses["bilateral"] = bilateral_loss / max(n_pairs, 1)
        else:
            losses["bilateral"] = torch.tensor(0.0, device=self.device)

        # 3b. feature-level contrastive over explicit correspondences
        if self.use_feature_contrast and out.get("feat") is not None \
                and self.feature_pairs.shape[0] > 0:
            feat_loss = 0.0
            for i in range(len(mods)):
                for j in range(i + 1, len(mods)):
                    feat_loss = feat_loss + self.feature_loss(
                        out["feat"][mods[i]], out["feat"][mods[j]],
                        self.feature_pairs)
            losses["feature"] = feat_loss / max(n_pairs, 1)
        else:
            losses["feature"] = torch.tensor(0.0, device=self.device)

        # 4. clustering losses
        if self.use_clustering and q is not None \
                and getattr(self, "cluster_enabled", True):
            p = self.cluster_head.target_distribution(q)
            losses["kl"] = self.kl_loss(q, p)
            losses["ddc"] = self.ddc_loss(q, joint)
        else:
            losses["kl"] = torch.tensor(0.0, device=self.device)
            losses["ddc"] = torch.tensor(0.0, device=self.device)

        # 5. encoder-side adversarial objective.
        #    The discriminator is frozen (its own optimiser handles its
        #    parameters), so the encoder maximises classification error via
        #    the negated cross entropy. Genuine min-max with two optimisers.
        if self.use_adversarial and self.training:
            adv = torch.tensor(0.0, device=self.device)
            if "modality_logits" in out and "modality_labels" in out:
                disc_loss = F.cross_entropy(out["modality_logits"],
                                            out["modality_labels"])
                adv = adv - disc_loss          # encoder maximises confusion
            if self.n_batches > 1 and batch_indices is not None and \
                    "batch_logits" in out:
                adv = adv - F.cross_entropy(
                    out["batch_logits"], batch_indices)
            losses["adv"] = adv
        else:
            losses["adv"] = torch.tensor(0.0, device=self.device)

        total = (w["recon"] * losses["recon"]
                 + w["contrast"] * losses["contrast"]
                 + w["bilateral"] * losses["bilateral"]
                 + w["feature"] * losses["feature"]
                 + w["kl"] * losses["kl"]
                 + w["ddc"] * losses["ddc"]
                 + w["adv"] * losses["adv"])
        return total, losses

    def get_embedding(self, x_dict, masks=None):
        self.eval()
        with torch.no_grad():
            z_dict = self.encode(x_dict, masks)
            joint = self.fuse(z_dict, training=False, masks=masks)
        return joint.cpu().numpy()

    def predict_cluster(self, x_dict, masks=None):
        self.eval()
        with torch.no_grad():
            z_dict = self.encode(x_dict, masks)
            joint = self.fuse(z_dict, training=False)
            q = self.cluster_head(joint)
        return q.argmax(1).cpu().numpy()
