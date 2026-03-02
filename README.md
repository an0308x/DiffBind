# DiffBind

# DiffBindFR: SE(3) Equivariant Network for Flexible Protein-Ligand Docking

A PyTorch reproduction of **DiffBindFR** from:

> Zhu et al., *"DiffBindFR: An SE(3) Equivariant Network for Flexible Protein-Ligand Docking"*, Chemical Science 2024.

DiffBindFR jointly samples ligand poses **and** binding-pocket side-chain torsion angles using an SE(3)-equivariant diffusion generative model, enabling flexible receptor docking without exhaustive conformational search.

---

## Architecture Overview

```
Input: Apo/Holo protein pocket + ligand graph
        │
        ├─ Protein encoder  (GVP – Geometric Vector Perceptron)
        ├─ Ligand encoder   (Graph Transformer / EGNN)
        │
        └─ SE(3)-Equivariant Diffusion Model
               │
               ├─ Ligand translation  (R³ diffusion)
               ├─ Ligand rotation     (SO(3) diffusion)
               ├─ Ligand torsion      (T^d diffusion)
               └─ Side-chain χ₁ angles (T^k diffusion)
                        │
                        ▼
               MDN Confidence Model → Top-1 pose selection
```

---

## Repository Layout

```
DiffBindFR/
├── README.md
├── environment.yml           # Conda environment
├── configs/
│   └── default.yaml          # Hyperparameters
├── models/
│   ├── __init__.py
│   ├── gvp_encoder.py        # GVP protein backbone encoder
│   ├── ligand_encoder.py     # Graph Transformer ligand encoder
│   ├── diffusion.py          # SE(3) diffusion model (core)
│   ├── score_network.py      # Score / denoising network
│   └── mdn_confidence.py     # MDN confidence scoring model
├── data/
│   ├── __init__.py
│   ├── dataset.py            # PDBbind / CD cross-dock dataset
│   ├── featurize.py          # Protein & ligand featurization
│   └── transforms.py         # Data augmentation & noise schedules
├── utils/
│   ├── __init__.py
│   ├── so3.py                # SO(3) / IGSO3 utilities
│   ├── torus.py              # Torus diffusion for torsion angles
│   ├── geometry.py           # RMSD, chi-angle helpers
│   └── sampling.py           # ODE / SDE solvers
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # L-RMSD, sc-RMSD, C-Dist, PB-success
│   └── posebusters.py        # PoseBusters wrapper
├── scripts/
│   ├── train.py              # Training entry point
│   ├── inference.py          # Docking inference
│   └── build_cd_testset.py   # CD cross-dock test set construction
└── tests/
    └── test_models.py        # Unit tests
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate diffbindfr

# Optional: install PoseBusters for validity checks
pip install posebusters
```

---

## Quick Start

### Training

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir /path/to/PDBbind2020 \
    --split time_split \
    --gpus 4
```

### Inference (flexible docking)

```bash
python scripts/inference.py \
    --protein receptor.pdb \
    --ligand ligand.sdf \
    --n_poses 40 \
    --solver SDE \
    --denoising_steps 20 \
    --confidence MDN \
    --output docked_poses.sdf
```

### Evaluation

```bash
python evaluation/metrics.py \
    --pred docked_poses.sdf \
    --ref crystal.sdf \
    --protein receptor.pdb
```

---

## Key Hyperparameters (`configs/default.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_poses` | 40 | Poses sampled per complex |
| `denoising_steps` | 20 | SDE/ODE steps |
| `solver` | SDE | Diffusion solver type |
| `pocket_radius` | 10.0 Å | Binding pocket definition |
| `mdn_gaussians` | 10 | MDN mixture components |
| `gvp_layers` | 3 | GVP encoder depth |
| `gt_layers` | 6 | Graph Transformer depth |
| `hidden_dim` | 256 | Node embedding dimension |

---

## Results (reproduced targets)

### PDBbind Time-Split Test Set

| Method | L-RMSD top1 | sc-RMSD top1 |
|--------|------------|--------------|
| AutoDock Vina | ~37% | – |
| DiffDock | ~38% | – |
| KarmaDock | ~43% | – |
| **DiffBindFR-MDN** | **~51%** | **~72%** |

*(Success rate = fraction with RMSD < 2 Å)*

### CD Cross-Dock Test Set (CASF2016 Apo-Holo)

| Method | SR (RMSD<2Å) | PB-SR |
|--------|-------------|-------|
| Glide | 13.2% | 13.2% |
| KarmaDock | 10.9% | 10.9% |
| **DiffBindFR-MDN** | **–** | **56.1%** |

---

## Citation

```bibtex
@article{zhu2024diffbindfr,
  title   = {DiffBindFR: An SE(3) Equivariant Network for Flexible Protein-Ligand Docking},
  author  = {Zhu, Jintao and Gu, Zhonghui and Pei, Jianfeng and Lai, Luhua},
  journal = {Chemical Science},
  year    = {2024},
  doi     = {10.1039/D3SC06803J}
}
```

---

## Acknowledgements

This reproduction builds on:
- [GVP-GNN](https://github.com/drorlab/gvp-pytorch) (Jing et al., ICLR 2021)
- [DiffDock](https://github.com/gcorso/DiffDock) (Corso et al., ICLR 2023)
- [KarmaDock](https://github.com/schrojunzhang/KarmaDock) (Zhang et al., Nature Comp. Sci. 2023)
