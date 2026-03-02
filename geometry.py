"""
Geometry utility functions for DiffBindFR.

Covers chi angle computation, coordinate transforms, and clash detection.
"""

import math
import torch
from torch import Tensor
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Dihedral angle computation
# ──────────────────────────────────────────────────────────────────────────────

def dihedral_angle(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """
    Compute the dihedral angle (in radians) defined by four points.

    p0, p1, p2, p3: [..., 3]
    Returns [...] tensor of angles in [-π, π].
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = b1 / b1.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Component of b0 and b2 perpendicular to b1
    v = b0 - (b0 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm
    w = b2 - (b2 * b1_norm).sum(dim=-1, keepdim=True) * b1_norm

    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1_norm, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def chi1_from_residue_coords(
    n_pos: Tensor,   # [3]
    ca_pos: Tensor,  # [3]
    cb_pos: Tensor,  # [3]
    cg_pos: Tensor,  # [3]  – first side-chain heavy atom
) -> Tensor:
    """Compute χ₁ torsion angle for a residue."""
    return dihedral_angle(
        n_pos.unsqueeze(0),
        ca_pos.unsqueeze(0),
        cb_pos.unsqueeze(0),
        cg_pos.unsqueeze(0),
    ).squeeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate transforms
# ──────────────────────────────────────────────────────────────────────────────

def apply_torsion(
    coords: Tensor,      # [N, 3] all atom positions
    torsion_idx: tuple,  # (i, j, k, l) atom indices defining the bond j-k
    delta: Tensor,       # scalar – angle to rotate by
) -> Tensor:
    """
    Apply a torsion rotation of `delta` radians about bond (j→k),
    moving all atoms on the `l` side of the bond.

    Returns updated coords [N, 3].
    """
    i, j, k, l = torsion_idx
    # Rotation axis: normalised j→k bond vector
    axis = coords[k] - coords[j]
    axis = axis / axis.norm().clamp(min=1e-8)

    # Rodrigues rotation of delta around axis, applied to atoms on l-side
    # (BFS from l not crossing j-k)
    cos_d = torch.cos(delta)
    sin_d = torch.sin(delta)

    # For simplicity, rotate atom l (extend to whole subtree in practice)
    v = coords[l] - coords[k]
    v_rot = (cos_d * v
             + sin_d * torch.cross(axis, v)
             + (1 - cos_d) * (axis * v).sum() * axis)

    new_coords = coords.clone()
    new_coords[l] = coords[k] + v_rot
    return new_coords


def center_and_scale(coords: Tensor, ref_coords: Tensor | None = None) -> tuple:
    """
    Centre coords by subtracting their mean (or ref_coords mean).

    Returns (centred_coords, centroid).
    """
    centroid = coords.mean(dim=0) if ref_coords is None else ref_coords.mean(dim=0)
    return coords - centroid, centroid


# ──────────────────────────────────────────────────────────────────────────────
# Clash detection
# ──────────────────────────────────────────────────────────────────────────────

def has_clash(
    lig_pos: Tensor,    # [Nl, 3]
    pro_pos: Tensor,    # [Np, 3]
    clash_dist: float = 1.5,
) -> bool:
    """
    Return True if any ligand-protein atom pair is closer than `clash_dist` Å.
    Used to filter out severely clashing cross-dock poses (SI §1).
    """
    dist = torch.cdist(lig_pos, pro_pos)   # [Nl, Np]
    return (dist < clash_dist).any().item()


def count_clashes(lig_pos: Tensor, pro_pos: Tensor, clash_dist: float = 1.5) -> int:
    """Count number of clashing ligand-protein atom pairs."""
    dist = torch.cdist(lig_pos, pro_pos)
    return (dist < clash_dist).sum().item()


# ──────────────────────────────────────────────────────────────────────────────
# Pocket backbone RMSD
# ──────────────────────────────────────────────────────────────────────────────

def pocket_backbone_rmsd(
    pred_ca: Tensor,  # [N, 3] predicted Cα positions
    ref_ca: Tensor,   # [N, 3] reference Cα positions
) -> float:
    """
    Pocket backbone Cα RMSD (Fig. S1 metric in the paper).
    Residues are assumed to be pre-matched.
    """
    return ((pred_ca - ref_ca) ** 2).sum(dim=-1).mean().sqrt().item()
