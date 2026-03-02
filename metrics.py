"""
Evaluation metrics for flexible protein-ligand docking.

Implements:
    L-RMSD   – Ligand Heavy-Atom RMSD (after symmetry-corrected alignment)
    sc-RMSD  – Side-chain heavy-atom RMSD for pocket residues
    C-Dist   – Ligand centroid distance
    |Δχ₁|   – Proportion of pocket residues with |Δχ₁| < 15°
    PB-success – PoseBusters validity + L-RMSD < 2 Å
"""

import math
import numpy as np
import torch
from torch import Tensor
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Core RMSD
# ──────────────────────────────────────────────────────────────────────────────

def rmsd(pred: Tensor, ref: Tensor) -> Tensor:
    """
    Heavy-atom RMSD between predicted and reference coordinates.

    pred, ref: [N, 3]
    Returns scalar tensor.
    """
    assert pred.shape == ref.shape, f"Shape mismatch: {pred.shape} vs {ref.shape}"
    return ((pred - ref) ** 2).sum(dim=-1).mean().sqrt()


def kabsch_rmsd(pred: Tensor, ref: Tensor) -> Tensor:
    """
    Minimum RMSD after optimal (Kabsch) rotation alignment.
    Aligns `pred` onto `ref`.

    pred, ref: [N, 3]
    """
    # Centre
    pred_c = pred - pred.mean(dim=0)
    ref_c  = ref  - ref.mean(dim=0)

    # Covariance matrix
    H = pred_c.T @ ref_c   # [3, 3]
    U, S, Vt = torch.linalg.svd(H)

    # Correct reflection
    d = torch.det(Vt.T @ U.T)
    D = torch.diag(torch.tensor([1., 1., d], device=pred.device))
    R = Vt.T @ D @ U.T     # optimal rotation

    pred_aligned = pred_c @ R.T
    return rmsd(pred_aligned, ref_c)


def symmetry_corrected_rmsd(
    pred: Tensor, ref: Tensor, mol=None
) -> Tensor:
    """
    RMSD accounting for molecular symmetry (graph automorphisms).

    If `mol` (RDKit Mol) is provided, use canonical atom ordering
    via the automorphism group to find the minimum RMSD over equivalent
    atom mappings. Falls back to plain RMSD otherwise.
    """
    if mol is None:
        return rmsd(pred, ref)

    try:
        from rdkit.Chem import rdmolops
        matches = mol.GetSubstructMatches(mol, uniquify=False)
        min_rmsd = float("inf")
        for match in matches:
            perm = list(match)
            r = rmsd(pred[perm], ref)
            if r.item() < min_rmsd:
                min_rmsd = r.item()
        return torch.tensor(min_rmsd)
    except Exception:
        return rmsd(pred, ref)


# ──────────────────────────────────────────────────────────────────────────────
# Ligand RMSD (L-RMSD)
# ──────────────────────────────────────────────────────────────────────────────

def ligand_rmsd(
    pred_pos: Tensor,
    ref_pos: Tensor,
    mol=None,
    use_symmetry: bool = True,
) -> float:
    """
    L-RMSD: ligand heavy-atom RMSD.

    pred_pos, ref_pos: [N, 3] heavy-atom coordinates
    """
    if use_symmetry and mol is not None:
        val = symmetry_corrected_rmsd(pred_pos, ref_pos, mol)
    else:
        val = rmsd(pred_pos, ref_pos)
    return val.item()


# ──────────────────────────────────────────────────────────────────────────────
# Side-chain RMSD (sc-RMSD)
# ──────────────────────────────────────────────────────────────────────────────

def sidechain_rmsd(pred_residues: list, ref_residues: list) -> float:
    """
    sc-RMSD: RMSD of side-chain heavy atoms (all atoms except N, CA, C, O)
    for a set of matched pocket residues.

    pred_residues, ref_residues: list of dicts with key 'sc_coords' [K_i, 3].
    """
    pred_coords, ref_coords = [], []
    for p, r in zip(pred_residues, ref_residues):
        if "sc_coords" in p and "sc_coords" in r and p["sc_coords"] is not None:
            pc = p["sc_coords"]
            rc = r["sc_coords"]
            n = min(pc.shape[0], rc.shape[0])
            pred_coords.append(pc[:n])
            ref_coords.append(rc[:n])
    if not pred_coords:
        return float("nan")
    pred_all = torch.cat(pred_coords, dim=0)
    ref_all  = torch.cat(ref_coords,  dim=0)
    return rmsd(pred_all, ref_all).item()


# ──────────────────────────────────────────────────────────────────────────────
# Centroid distance (C-Dist)
# ──────────────────────────────────────────────────────────────────────────────

def centroid_distance(pred_pos: Tensor, ref_pos: Tensor) -> float:
    """Euclidean distance between predicted and reference ligand centroids."""
    return (pred_pos.mean(dim=0) - ref_pos.mean(dim=0)).norm().item()


# ──────────────────────────────────────────────────────────────────────────────
# Chi-1 angle accuracy  |Δχ₁|
# ──────────────────────────────────────────────────────────────────────────────

def delta_chi1(
    pred_chi1: Tensor,  # [K] predicted χ₁ in radians
    ref_chi1: Tensor,   # [K] reference χ₁ in radians
) -> Tensor:
    """
    Absolute difference in χ₁ angles, wrapped to [0°, 180°].
    Returns [K] tensor in degrees.
    """
    diff = (pred_chi1 - ref_chi1 + math.pi) % (2 * math.pi) - math.pi
    return diff.abs() * (180 / math.pi)


def chi1_accuracy(
    pred_chi1: Tensor,
    ref_chi1: Tensor,
    threshold: float = 15.0,
) -> float:
    """
    |Δχ₁| < threshold:  proportion of pocket residues satisfying this criterion.
    Default threshold = 15° following DiffBindFR paper convention.
    """
    dc = delta_chi1(pred_chi1, ref_chi1)
    return (dc < threshold).float().mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Success rate helpers
# ──────────────────────────────────────────────────────────────────────────────

def success_rate(
    rmsd_list: list,
    threshold: float = 2.0,
) -> float:
    """Fraction of poses with L-RMSD below `threshold` Å."""
    n_success = sum(r < threshold for r in rmsd_list)
    return n_success / len(rmsd_list) if rmsd_list else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_docking(
    pred_pos: Tensor,
    ref_pos: Tensor,
    pred_chi1: Optional[Tensor] = None,
    ref_chi1: Optional[Tensor] = None,
    pred_residues: Optional[list] = None,
    ref_residues: Optional[list] = None,
    mol=None,
) -> dict:
    """
    Compute all DiffBindFR evaluation metrics for a single complex.

    Returns a dict with:
        l_rmsd, c_dist, sc_rmsd, chi1_acc, success_2A
    """
    l_rmsd = ligand_rmsd(pred_pos, ref_pos, mol)
    c_dist = centroid_distance(pred_pos, ref_pos)

    results = dict(
        l_rmsd=l_rmsd,
        c_dist=c_dist,
        success_2A=float(l_rmsd < 2.0),
    )

    if pred_chi1 is not None and ref_chi1 is not None:
        results["chi1_acc_15deg"] = chi1_accuracy(pred_chi1, ref_chi1, 15.0)
        results["delta_chi1_mean"] = delta_chi1(pred_chi1, ref_chi1).mean().item()

    if pred_residues is not None and ref_residues is not None:
        results["sc_rmsd"] = sidechain_rmsd(pred_residues, ref_residues)

    return results


def aggregate_metrics(per_complex_results: list) -> dict:
    """
    Aggregate per-complex metric dicts into mean/median statistics
    (following Table S6/S7/S8 in the supplementary).
    """
    import statistics
    keys = per_complex_results[0].keys() if per_complex_results else []
    agg = {}
    for k in keys:
        vals = [r[k] for r in per_complex_results if not math.isnan(r.get(k, float("nan")))]
        if vals:
            agg[f"{k}_mean"]   = sum(vals) / len(vals)
            agg[f"{k}_median"] = statistics.median(vals)
            agg[f"{k}_std"]    = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return agg
