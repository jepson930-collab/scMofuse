# scMoFuse

**Single-cell multi-omics fusion with adaptive cross-attention, differential
gating and bilateral contrastive learning.**

scMoFuse is an unsupervised deep-learning framework that integrates paired or
*partially observed* (mosaic) single-cell modalities — e.g. scRNA-seq +
scATAC-seq (10x Multiome) or scRNA-seq + protein (CITE-seq) — into a single
joint cell embedding for clustering, visualisation and cell-type discovery.

This repository contains the model implementation and a runnable CPU demo.
No dataset download is required to verify that the code works.

- Repository: <https://github.com/jepson930-collab/scMofuse>
- Python: >= 3.8, CPU is sufficient to run the demo

---

## Design at a glance

Each design element maps to a concrete problem in multi-omics integration:

| Problem | Module in `scmofuse/` | Option |
| --- | --- | --- |
| Modality dimension imbalance (RNA ≫ protein) | per-modality MLP encoders to a shared latent dim | `ModalityEncoder` |
| Mosaic / missing modalities | masked cross-attention fusion, mask-aware losses | `use_masked_fusion` |
| Conflicting modality signals | per-cell differential attention gate | `use_differential` |
| Count vs. binary heterogeneity | modality-specific likelihoods (NB / Bernoulli / Gaussian) | `ReconstructionLoss` |
| Modality / batch bias | adversarial discriminators (separate optimiser) | `use_adversarial` |
| Cell- and feature-level alignment | bilateral cell-level InfoNCE + feature-level InfoNCE over explicit correspondences | `use_bilateral`, `use_feature_contrast` |
| Cell-type structure | Student-t clustering head + KL / DDC objectives | `use_clustering` |

All switches above can be turned off individually, which makes the model a
direct drop-in for ablation studies.

---

## Installation

```bash
git clone https://github.com/jepson930-collab/scMofuse.git
cd scMofuse

# core runtime (torch / numpy / scipy / scikit-learn)
pip install -r requirements.txt

# optional: install as a package (needed only for `import scmofuse` outside the repo)
pip install -e .
```

`torch`, `numpy`, `scipy` and `scikit-learn` are the only hard requirements.
`anndata`, `scanpy`, `matplotlib` and `tqdm` are optional (real-data I/O and
plotting) and are listed as commented lines in `requirements.txt`.

---

## Quick start (no data download)

```bash
python examples/run_demo.py
python examples/run_demo.py --epochs 60 --n-cells 600    # a bit longer / larger
python examples/run_demo.py --mosaic                     # 10% of modality entries missing
```

The demo synthesises a small paired RNA + ATAC cohort with a known cell-type
structure, trains scMoFuse with the two-optimiser min-max recipe, and evaluates
the joint embedding. Default run (~3 s on CPU):

```
device=cpu  cells=400  modalities=RNA+ATAC  mosaic=False
epoch 01/20  loss=341.4933
...
epoch 20/20  loss=278.3517

finished in 2.8s  (embedding (400, 32))
k-means on joint embedding:
    ARI                  1.0000
    NMI                  1.0000
    FMI                  1.0000
    silhouette           0.7625
    graph_connectivity   1.0000
model cluster head (argmax of soft assignment):
    ARI                  0.5589
    NMI                  0.6006
    FMI                  0.6759
```

The synthetic cohort is deliberately easy, so downstream k-means on the joint
embedding recovers the ground-truth types exactly. This is the protocol used in
the manuscript for reporting ARI/NMI/FMI. The built-in Student-t cluster head is
a secondary readout that converges more slowly and tightens with longer training
(`--epochs 60` raises its ARI to ≈0.66 here).

> On Windows, `scikit-learn` may print a harmless `joblib` warning about
> physical core detection; it does not affect the results.

---

## Using your own data

Data is passed as a dictionary of modality name → array of shape
`(n_cells, n_features)`, with all modalities row-aligned to the same cells.

```python
import torch
from torch.utils.data import DataLoader
from scmofuse import (scMoFuse, MultiOmicsDataset, collate_multiomics,
                      normalize_counts, binarize_atac, set_seed)

set_seed(0)

# 1. preprocess each modality
rna  = normalize_counts(rna_counts)   # library-size + log1p, sparse or dense
atac = binarize_atac(atac_counts)     # peaks -> 0/1

# 2. wrap and batch
dataset = MultiOmicsDataset({"rna": rna, "atac": atac}, labels=cell_labels)
loader  = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True,
                     collate_fn=collate_multiomics)

# 3. build the model
model = scMoFuse(
    input_dims={"rna": rna.shape[1], "atac": atac.shape[1]},
    latent_dim=128, n_clusters=n_clusters,
    modality_types={"rna": "rna", "atac": "atac"},
    device="cpu",
)

opt      = torch.optim.Adam(model.parameters(), lr=1e-3)
opt_disc = torch.optim.Adam(model.modality_disc.parameters(), lr=1e-3)  # adversarial

# 4. train (encoder step + discriminator step)
for epoch in range(200):
    model.train()
    for batch in loader:
        x     = {m: batch[m] for m in ("rna", "atac")}
        masks = {m: batch[m + "_mask"] for m in ("rna", "atac")}
        loss, _ = model.compute_loss(model(x, masks=masks), x, masks=masks)
        opt.zero_grad(); loss.backward(); opt.step()

        d_loss = model.discriminator_loss(x, masks=masks)
        opt_disc.zero_grad(); d_loss.backward(); opt_disc.step()

# 5. use the joint embedding
embedding = model.get_embedding({"rna": torch.tensor(rna),
                                 "atac": torch.tensor(atac)})
```

To read `.h5ad` files directly, install the optional dependencies and convert
them to arrays, e.g. `adata.X` / `adata.obsm` → the dictionaries above.

### Mosaic / missing modalities

scMoFuse natively supports cells for which only a subset of modalities was
measured. Pass a per-modality boolean presence array:

```python
from scmofuse import make_mosaic_masks

m = make_mosaic_masks(n_cells, n_modalities=2, missing_frac=0.1, seed=0)
masks = {"rna": m[:, 0], "atac": m[:, 1]}
dataset = MultiOmicsDataset({"rna": rna, "atac": atac}, masks=masks)
```

Missing entries are zeroed before encoding, excluded from the reconstruction and
contrastive terms, and handled by the masked cross-attention fusion. Cells with a
single observed modality are represented by that modality alone.

### Feature-level correspondences (optional)

Cell-level contrast uses the diagonal (co-measured cells). To additionally align
features, supply explicit correspondences (protein marker → gene, or peak → gene):

```python
model.set_feature_pairs([(gene_idx, protein_idx), ...])
```

---

## Public API

| Symbol | Purpose |
| --- | --- |
| `scMoFuse` | the model: `encode`, `fuse`, `forward`, `compute_loss`, `discriminator_loss`, `get_embedding`, `predict_cluster`, `set_feature_pairs` |
| `MultiOmicsDataset`, `collate_multiomics` | dataset and batch collation for row-aligned modalities |
| `normalize_counts`, `binarize_atac`, `select_hvg`, `gene_activity_matrix` | preprocessing helpers |
| `make_mosaic_masks` | generate mosaic missing-modality masks |
| `set_seed`, `to_device` | reproducibility / device helpers |
| `evaluate_all`, `clustering_metrics` | ARI / NMI / FMI, silhouette, graph connectivity, modality mixing |

`evaluate_all(embedding, cell_labels, modality_labels=None)` runs k-means on the
joint embedding and returns all metrics as a dict.

---

## Repository layout

```
scMoFuse/
├── scmofuse/
│   ├── __init__.py     # public API
│   ├── model.py        # scMoFuse model (encoder/fusion/losses wiring)
│   ├── modules.py      # encoders, decoders, differential & masked cross-attention,
│   │                   # discriminators, feature encoder, clustering head
│   ├── losses.py       # InfoNCE, bilateral / feature contrast, NB-Bernoulli recon,
│   │                   # KL and DDC clustering objectives
│   ├── data.py         # dataset, collation, preprocessing helpers
│   ├── evaluate.py     # clustering & integration metrics
│   └── utils.py        # seeding, mosaic masks
├── examples/
│   └── run_demo.py     # self-contained synthetic CPU demo
├── requirements.txt
└── setup.py
```

---

## Reproducing the manuscript results

The manuscript trains scMoFuse with the API shown above; the exact recipe is:

1. `normalize_counts` for RNA / protein, `binarize_atac` (or
   `gene_activity_matrix`) for ATAC, and `select_hvg` for feature selection.
2. Optional `set_feature_pairs` for feature-level contrast (marker→gene or
   peak→gene links).
3. Two Adam optimisers: `compute_loss` for the encoder side, and
   `discriminator_loss` for the adversarial discriminators. The clustering
   objective is activated after a short warm-up (`model.cluster_enabled`).
4. `evaluate_all` for clustering (ARI/NMI/FMI), silhouette, graph connectivity
   and modality mixing; `predict_cluster` or k-means for cell-type assignment.

Datasets are not redistributed here for size and licensing reasons; they are
listed in the Data Availability statement of the manuscript. Download them,
convert to the modality dictionaries described above, and run the recipe.

---

## Environment

Verified with Python 3.8.5, `torch==2.4.1` (CPU), `numpy==1.24.4`,
`scikit-learn==1.3.2`. Any Python >= 3.8 with recent versions of the four core
packages should work.

---

## Citation

If you find scMoFuse useful, please cite the accompanying manuscript (the
reference will be updated upon publication) and this repository.