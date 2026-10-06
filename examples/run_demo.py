"""
scMoFuse quick-start demo (synthetic data, CPU-only)
====================================================
A self-contained example that lets a reviewer exercise scMoFuse without
downloading any single-cell dataset.

It synthesises a small paired RNA + ATAC cohort with a known cell-type
structure, trains the model with the same two-optimiser min-max recipe used
in the manuscript, and reports clustering metrics on the learned joint
embedding.

Usage
-----
    python examples/run_demo.py                     # ~1 min on CPU
    python examples/run_demo.py --epochs 30 --n-cells 600
    python examples/run_demo.py --mosaic            # 10% of modality entries missing

Dependencies: torch, numpy, scipy, scikit-learn (see requirements.txt).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

# make the demo runnable straight from a clone, without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scmofuse import (  # noqa: E402
    scMoFuse, MultiOmicsDataset, collate_multiomics,
    normalize_counts, binarize_atac, set_seed, make_mosaic_masks,
    evaluate_all, clustering_metrics,
)


def make_synthetic(n_cells, n_genes, n_peaks, n_types, seed):
    """Cluster-structured RNA counts and ATAC peak accessibility."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_types, size=n_cells)

    # per-cluster transcriptional / accessibility programme
    gene_rate = rng.gamma(shape=2.0, scale=1.0, size=(n_types, n_genes)) + 0.2
    peak_prob = rng.beta(a=1.5, b=6.0, size=(n_types, n_peaks))

    depth = rng.gamma(shape=3.0, scale=1.0, size=(n_cells, 1))
    rna = rng.poisson(gene_rate[labels] * depth).astype(np.float32)
    atac = (rng.random((n_cells, n_peaks)) < peak_prob[labels]).astype(np.float32)
    return rna, atac, labels


def main():
    ap = argparse.ArgumentParser(
        description="scMoFuse synthetic CPU demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--n-cells", type=int, default=400)
    ap.add_argument("--n-genes", type=int, default=200)
    ap.add_argument("--n-peaks", type=int, default=300)
    ap.add_argument("--n-types", type=int, default=4)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5,
                    help="epochs before the clustering head is activated")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mosaic", action="store_true",
                    help="randomly drop modalities to emulate mosaic integration")
    args = ap.parse_args()

    if args.n_cells < args.batch_size:
        ap.error("--n-cells must be >= --batch-size")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 1. synthetic paired multi-omics data
    # ------------------------------------------------------------------
    rna, atac, labels = make_synthetic(
        args.n_cells, args.n_genes, args.n_peaks, args.n_types, args.seed)
    rna = normalize_counts(rna)          # library-size + log1p
    atac = binarize_atac(atac)           # peaks -> 0/1

    masks = None
    if args.mosaic:
        m = make_mosaic_masks(args.n_cells, 2, missing_frac=0.1, seed=args.seed)
        masks = {"rna": m[:, 0], "atac": m[:, 1]}

    dataset = MultiOmicsDataset({"rna": rna, "atac": atac},
                                labels=labels, masks=masks)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, collate_fn=collate_multiomics)

    # ------------------------------------------------------------------
    # 2. model + the two-optimiser min-max recipe
    # ------------------------------------------------------------------
    model = scMoFuse(
        input_dims={"rna": args.n_genes, "atac": args.n_peaks},
        latent_dim=args.latent,
        n_clusters=args.n_types,
        hidden_dims=(128, 64),
        decoder_hidden=(64, 128),
        modality_types={"rna": "rna", "atac": "atac"},
        device=device,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt_disc = torch.optim.Adam(model.modality_disc.parameters(), lr=args.lr)

    # delay the clustering objective until the latent space is stable
    model.cluster_enabled = False

    print(f"device={device}  cells={args.n_cells}  "
          f"modalities=RNA+ATAC  mosaic={args.mosaic}")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        if epoch == args.warmup:
            model.cluster_enabled = True
        running = 0.0
        for batch in loader:
            x = {m: batch[m].to(device) for m in ("rna", "atac")}
            bmask = {m: batch[m + "_mask"].to(device) for m in ("rna", "atac")}

            out = model(x, masks=bmask)
            loss, _ = model.compute_loss(out, x, masks=bmask)
            opt.zero_grad()
            loss.backward()
            opt.step()

            dloss = model.discriminator_loss(x, masks=bmask)
            opt_disc.zero_grad()
            dloss.backward()
            opt_disc.step()

            running += float(loss)
        print(f"epoch {epoch + 1:02d}/{args.epochs}  loss={running / len(loader):.4f}")

    # ------------------------------------------------------------------
    # 3. evaluate the joint embedding
    # ------------------------------------------------------------------
    model.eval()
    x_full = {"rna": torch.tensor(rna, device=device),
              "atac": torch.tensor(atac, device=device)}
    f_mask = None
    if masks is not None:
        f_mask = {m: torch.tensor(masks[m], device=device) for m in ("rna", "atac")}

    emb = model.get_embedding(x_full, f_mask)
    metrics = evaluate_all(emb, labels)
    pred = model.predict_cluster(x_full, f_mask)
    q_metrics = clustering_metrics(labels, pred)

    print(f"\nfinished in {time.time() - t0:.1f}s  (embedding {emb.shape})")
    print("k-means on joint embedding:")
    for k in ("ARI", "NMI", "FMI", "silhouette", "graph_connectivity"):
        print(f"    {k:<20s} {metrics[k]:.4f}")
    print("model cluster head (argmax of soft assignment):")
    for k in ("ARI", "NMI", "FMI"):
        print(f"    {k:<20s} {q_metrics[k]:.4f}")


if __name__ == "__main__":
    main()