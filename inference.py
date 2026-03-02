"""
DiffBindFR inference script: dock a ligand into a protein pocket.

Usage:
    python scripts/inference.py \\
        --protein receptor.pdb \\
        --ligand  ligand.sdf \\
        --checkpoint checkpoints/best.pt \\
        --n_poses 40 \\
        --solver SDE \\
        --denoising_steps 20 \\
        --confidence MDN \\
        --output docked_poses.sdf
"""

import argparse
import os
import sys
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.diffusion import DiffBindFR
from models.mdn_confidence import MDNConfidenceModel
from data.dataset import parse_pocket_from_pdb, featurise_pocket
from evaluation.metrics import evaluate_docking


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("DiffBindFR inference")
    p.add_argument("--protein",          required=True)
    p.add_argument("--ligand",           required=True)
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--config",           default="configs/default.yaml")
    p.add_argument("--n_poses",          type=int, default=40)
    p.add_argument("--solver",           default="SDE", choices=["SDE", "ODE"])
    p.add_argument("--denoising_steps",  type=int, default=20)
    p.add_argument("--confidence",       default="MDN",  choices=["MDN", "Smina"])
    p.add_argument("--pocket_radius",    type=float, default=10.0)
    p.add_argument("--output",           default="docked_poses.sdf")
    p.add_argument("--ref_ligand",       default=None,
                   help="Reference ligand SDF for RMSD calculation")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Single-complex inference
# ──────────────────────────────────────────────────────────────────────────────

def dock(
    model: DiffBindFR,
    mdn_model: MDNConfidenceModel,
    batch,
    n_poses: int,
    solver: str,
    n_steps: int,
    device: torch.device,
) -> list:
    """
    Generate `n_poses` docked structures and rank them.

    Returns list of (score, lig_pos) tuples sorted by descending confidence.
    """
    import types
    poses = []

    for _ in range(n_poses):
        result = model.sample(batch, n_steps=n_steps, solver=solver)
        poses.append(result["lig_pos"].cpu())

    # Score with MDN confidence model
    scored = []
    for lig_pos in poses:
        score_batch = _make_score_batch(batch, lig_pos.to(device))
        with torch.no_grad():
            score = mdn_model.score(score_batch).item()
        scored.append((score, lig_pos))

    scored.sort(key=lambda x: x[0], reverse=True)  # highest score = best
    return scored


def _make_score_batch(template_batch, lig_pos):
    """Clone batch and replace ligand positions."""
    import copy, types
    new = types.SimpleNamespace(**vars(template_batch))
    new.lig_pos = lig_pos
    return new


# ──────────────────────────────────────────────────────────────────────────────
# Build batch from protein PDB + ligand SDF
# ──────────────────────────────────────────────────────────────────────────────

def build_inference_batch(protein_path: str, ligand_path: str, pocket_radius: float, device):
    import types
    from rdkit import Chem
    from models.ligand_encoder import mol_to_graph
    from data.dataset import get_torsion_bonds

    mol = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=True)[0]
    if mol is None:
        raise ValueError(f"Could not parse ligand from {ligand_path}")

    try:
        conf = mol.GetConformer()
        lig_pos_init = torch.tensor(conf.GetPositions(), dtype=torch.float)
    except Exception:
        raise ValueError("Ligand SDF has no 3D conformer. Generate one first.")

    ligand_center = lig_pos_init.mean(dim=0).numpy()
    residues = parse_pocket_from_pdb(protein_path, pocket_radius, ligand_center)
    if len(residues) == 0:
        raise ValueError("No pocket residues found. Check radius and protein file.")

    pocket_feats = featurise_pocket(residues)
    lig_graph = mol_to_graph(mol)

    rot_bonds = get_torsion_bonds(mol)
    if rot_bonds:
        tor_i = torch.tensor([b[1] for b in rot_bonds], dtype=torch.long)
        tor_j = torch.tensor([b[2] for b in rot_bonds], dtype=torch.long)
        tor_edge_index = torch.stack([tor_i, tor_j])
        lig_torsions   = torch.zeros(len(rot_bonds))
    else:
        tor_edge_index = torch.zeros(2, 0, dtype=torch.long)
        lig_torsions   = torch.zeros(0)

    batch = types.SimpleNamespace(
        # Protein
        node_s    = pocket_feats["node_s"].to(device),
        node_v    = pocket_feats["node_v"].to(device),
        edge_s    = pocket_feats["edge_s"].to(device),
        edge_v    = pocket_feats["edge_v"].to(device),
        edge_index= pocket_feats["edge_index"].to(device),
        pro_pos   = pocket_feats["ca_pos"].to(device),
        pro_batch = torch.zeros(pocket_feats["n_residues"], dtype=torch.long, device=device),
        # Ligand
        lig_atom_feats = lig_graph["atom_feats"].to(device),
        lig_bond_feats = lig_graph["bond_feats"].to(device),
        lig_edge_index = lig_graph["edge_index"].to(device),
        lig_pos_crystal= lig_pos_init.to(device),
        lig_pos        = lig_pos_init.clone().to(device),
        lig_batch      = torch.zeros(mol.GetNumAtoms(), dtype=torch.long, device=device),
        # Torsions
        tor_edge_index = tor_edge_index.to(device),
        lig_torsions   = lig_torsions.to(device),
        sc_torsions    = torch.zeros(0, device=device),
        sc_residue_idx = torch.zeros(0, dtype=torch.long, device=device),
        tor_batch      = torch.zeros(len(lig_torsions), dtype=torch.long, device=device),
        sc_batch       = torch.zeros(0, dtype=torch.long, device=device),
        # Diffusion placeholders (will be overwritten during sampling)
        sigma_tr  = torch.ones(1, device=device),
        sigma_rot = torch.ones(1, device=device),
        sigma_tor = torch.ones(1, device=device),
        sigma_sc  = torch.ones(1, device=device),
    )
    return batch, mol


# ──────────────────────────────────────────────────────────────────────────────
# Write output SDF
# ──────────────────────────────────────────────────────────────────────────────

def write_sdf(mol, scored_poses: list, output_path: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    writer = Chem.SDWriter(output_path)
    for rank, (score, lig_pos) in enumerate(scored_poses):
        try:
            conf = mol.GetConformer()
        except Exception:
            AllChem.EmbedMolecule(mol)
            conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, lig_pos[i].tolist())
        mol.SetProp("DiffBindFR_score", f"{score:.4f}")
        mol.SetProp("DiffBindFR_rank", str(rank + 1))
        writer.write(mol)
    writer.close()
    print(f"Written {len(scored_poses)} poses to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load models
    model     = DiffBindFR(cfg).to(device)
    mdn_model = MDNConfidenceModel(hidden_dim=cfg["model"]["hidden_dim"]).to(device)

    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    mdn_model.load_state_dict(ck["mdn"])
    model.eval(); mdn_model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Build batch
    batch, mol = build_inference_batch(
        args.protein, args.ligand, args.pocket_radius, device
    )

    # Sample poses
    print(f"Sampling {args.n_poses} poses with {args.solver} solver "
          f"({args.denoising_steps} steps)...")
    scored_poses = dock(
        model, mdn_model, batch,
        n_poses=args.n_poses,
        solver=args.solver,
        n_steps=args.denoising_steps,
        device=device,
    )

    # Write output
    write_sdf(mol, scored_poses, args.output)

    # Optionally evaluate against reference
    if args.ref_ligand:
        from rdkit import Chem
        ref_mol = Chem.SDMolSupplier(args.ref_ligand, removeHs=False)[0]
        if ref_mol is not None:
            try:
                ref_pos = torch.tensor(
                    ref_mol.GetConformer().GetPositions(), dtype=torch.float
                )
                top1_pos = scored_poses[0][1]
                metrics = evaluate_docking(top1_pos, ref_pos, mol=mol)
                print(f"\nTop-1 evaluation:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.3f}")
            except Exception as e:
                print(f"Could not evaluate: {e}")


if __name__ == "__main__":
    main()
