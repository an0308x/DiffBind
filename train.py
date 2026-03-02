"""
DiffBindFR training script.

Usage:
    python scripts/train.py --config configs/default.yaml \\
        --data_dir /path/to/PDBbind2020 --gpus 4
"""

import argparse
import os
import math
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.diffusion import DiffBindFR
from models.mdn_confidence import MDNConfidenceModel
from data.dataset import PDBBindDataset


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("DiffBindFR trainer")
    p.add_argument("--config",   default="configs/default.yaml")
    p.add_argument("--data_dir", required=True, help="Path to PDBbind2020 root")
    p.add_argument("--split",    default="time_split", choices=["time_split", "mlsf_split"])
    p.add_argument("--gpus",     type=int, default=1)
    p.add_argument("--output",   default="checkpoints/")
    p.add_argument("--resume",   default=None, help="Checkpoint to resume from")
    p.add_argument("--wandb",    action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Collate function
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Simple collate that filters None items and stacks into a namespace."""
    import types
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    result = types.SimpleNamespace()
    B = len(batch)

    # Per-complex scalars
    result.pdb_ids = [b["pdb_id"] for b in batch]

    # Sample diffusion times and compute sigmas
    tr_cfg  = {"sigma_min": 0.1,    "sigma_max": 19.0}
    rot_cfg = {"sigma_min": 0.03,   "sigma_max": 1.55}
    tor_cfg = {"sigma_min": 0.0314, "sigma_max": 3.14}

    t = torch.rand(B)
    result.t_tr  = t.clone()
    result.t_rot = t.clone()
    result.t_tor = t.clone()
    result.t_sc  = t.clone()
    result.sigma_tr  = (tr_cfg["sigma_min"]  ** (1-t) * tr_cfg["sigma_max"]  ** t)
    result.sigma_rot = (rot_cfg["sigma_min"] ** (1-t) * rot_cfg["sigma_max"] ** t)
    result.sigma_tor = (tor_cfg["sigma_min"] ** (1-t) * tor_cfg["sigma_max"] ** t)
    result.sigma_sc  = (tor_cfg["sigma_min"] ** (1-t) * tor_cfg["sigma_max"] ** t)

    # Cat ligand features with batch indices
    lig_atom_feats, lig_bond_feats = [], []
    lig_edge_index, lig_pos_crystal = [], []
    lig_batch = []
    for i, b in enumerate(batch):
        n = b["lig_atom_feats"].shape[0]
        lig_atom_feats.append(b["lig_atom_feats"])
        lig_bond_feats.append(b["lig_bond_feats"])
        offset = sum(batch[j]["lig_atom_feats"].shape[0] for j in range(i))
        lig_edge_index.append(b["lig_edge_index"] + offset)
        lig_pos_crystal.append(b["lig_pos_crystal"])
        lig_batch.append(torch.full((n,), i, dtype=torch.long))

    result.lig_atom_feats  = torch.cat(lig_atom_feats)
    result.lig_bond_feats  = torch.cat(lig_bond_feats) if any(b.shape[0] > 0 for b in lig_bond_feats) else torch.zeros(0, lig_bond_feats[0].shape[-1])
    result.lig_edge_index  = torch.cat(lig_edge_index, dim=1) if lig_edge_index else torch.zeros(2, 0, dtype=torch.long)
    result.lig_pos_crystal = torch.cat(lig_pos_crystal)
    result.lig_pos         = result.lig_pos_crystal.clone()
    result.lig_batch       = torch.cat(lig_batch)

    # Cat protein features
    pro_s, pro_v = [], []
    pro_edge_s, pro_edge_v, pro_edge_index = [], [], []
    pro_pos, pro_batch = [], []
    for i, b in enumerate(batch):
        np_ = b["prot_n_residues"]
        pro_s.append(b["prot_node_s"])
        pro_v.append(b["prot_node_v"])
        pro_pos.append(b["prot_ca_pos"])
        offset = sum(batch[j]["prot_n_residues"] for j in range(i))
        pro_edge_s.append(b["prot_edge_s"])
        pro_edge_v.append(b["prot_edge_v"])
        pro_edge_index.append(b["prot_edge_index"] + offset)
        pro_batch.append(torch.full((np_,), i, dtype=torch.long))

    result.node_s     = torch.cat(pro_s)
    result.node_v     = torch.cat(pro_v)
    result.edge_s     = torch.cat(pro_edge_s)
    result.edge_v     = torch.cat(pro_edge_v)
    result.edge_index = torch.cat(pro_edge_index, dim=1)
    result.pro_pos    = torch.cat(pro_pos)
    result.pro_batch  = torch.cat(pro_batch)

    # Torsions
    tor_edges = [b["tor_edge_index"] for b in batch]
    result.tor_edge_index = torch.cat(tor_edges, dim=1) if any(e.shape[1] > 0 for e in tor_edges) else torch.zeros(2, 0, dtype=torch.long)
    result.lig_torsions   = torch.cat([b["lig_torsions"] for b in batch])
    result.sc_torsions    = torch.cat([b["sc_torsions"]  for b in batch])
    result.sc_residue_idx = torch.cat([b["sc_residue_idx"] for b in batch])

    # Batch indices for torsions
    tor_batch = []
    for i, b in enumerate(batch):
        tor_batch.append(torch.full((b["lig_torsions"].shape[0],), i, dtype=torch.long))
    result.tor_batch = torch.cat(tor_batch)
    result.sc_batch  = torch.zeros(0, dtype=torch.long)  # placeholder

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # Datasets
    train_ds = PDBBindDataset(
        args.data_dir,
        split="train",
        pocket_radius=cfg["data"]["pocket_radius"],
        max_residues=cfg["data"]["max_pocket_residues"],
        max_atoms=cfg["data"]["max_ligand_atoms"],
        augment=True,
    )
    val_ds = PDBBindDataset(
        args.data_dir,
        split="val",
        pocket_radius=cfg["data"]["pocket_radius"],
        max_residues=cfg["data"]["max_pocket_residues"],
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
    )

    # Models
    model     = DiffBindFR(cfg).to(device)
    mdn_model = MDNConfidenceModel(hidden_dim=cfg["model"]["hidden_dim"]).to(device)

    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        mdn_model.load_state_dict(ck["mdn"])
        print(f"Resumed from {args.resume}")

    # Optimisers
    opt_score = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    opt_mdn = torch.optim.AdamW(
        mdn_model.parameters(),
        lr=cfg["training"]["lr"],
    )

    scaler = GradScaler()

    # Logging
    if args.wandb:
        import wandb
        wandb.init(project="DiffBindFR", config=cfg)

    step = 0
    best_val = float("inf")

    for epoch in range(cfg["training"]["max_epochs"]):
        model.train()
        mdn_model.train()

        for batch in train_loader:
            if batch is None:
                continue
            batch_gpu = _to_device(batch, device)

            # Score network loss
            opt_score.zero_grad()
            with autocast():
                losses = model.training_loss(batch_gpu)
                loss = losses["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt_score)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            scaler.step(opt_score)
            scaler.update()

            # MDN loss (use crystal poses)
            opt_mdn.zero_grad()
            with autocast():
                batch_gpu.lig_pos = batch_gpu.lig_pos_crystal
                mdn_out  = mdn_model(batch_gpu)
                mdn_loss = mdn_model.compute_loss(batch_gpu, mdn_out)
            scaler.scale(mdn_loss).backward()
            scaler.step(opt_mdn)
            scaler.update()

            step += 1

            if step % cfg["training"]["log_interval"] == 0:
                log_str = (
                    f"Epoch {epoch} | Step {step} | "
                    f"Loss {loss.item():.4f} | "
                    f"TR {losses['loss_tr'].item():.4f} | "
                    f"ROT {losses['loss_rot'].item():.4f} | "
                    f"TOR {losses['loss_tor'].item():.4f} | "
                    f"SC {losses['loss_sc'].item():.4f} | "
                    f"MDN {mdn_loss.item():.4f}"
                )
                print(log_str)
                if args.wandb:
                    wandb.log({
                        "loss": loss.item(),
                        "loss_tr": losses["loss_tr"].item(),
                        "loss_rot": losses["loss_rot"].item(),
                        "loss_mdn": mdn_loss.item(),
                        "step": step,
                    })

            if step % cfg["training"]["val_interval"] == 0:
                val_loss = _validate(model, mdn_model, val_loader, device, cfg)
                print(f"  Val loss: {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "model": model.state_dict(),
                        "mdn": mdn_model.state_dict(),
                        "step": step,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    }, os.path.join(args.output, "best.pt"))
                    print(f"  ✓ Saved best checkpoint (val={val_loss:.4f})")

        # Epoch checkpoint
        torch.save({
            "model": model.state_dict(),
            "mdn": mdn_model.state_dict(),
            "step": step,
            "epoch": epoch,
        }, os.path.join(args.output, f"epoch_{epoch:04d}.pt"))


@torch.no_grad()
def _validate(model, mdn_model, loader, device, cfg):
    model.eval(); mdn_model.eval()
    total, n = 0., 0
    for batch in loader:
        if batch is None:
            continue
        batch_gpu = _to_device(batch, device)
        losses = model.training_loss(batch_gpu)
        total += losses["loss"].item()
        n += 1
    model.train(); mdn_model.train()
    return total / max(n, 1)


def _to_device(batch, device):
    """Move all tensor attributes of a SimpleNamespace to device."""
    import types
    for k, v in vars(batch).items():
        if isinstance(v, torch.Tensor):
            setattr(batch, k, v.to(device))
    return batch


if __name__ == "__main__":
    args = parse_args()
    train(args)
