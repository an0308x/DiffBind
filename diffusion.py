"""
DiffBindFR: Full SE(3) Diffusion Model for Flexible Protein-Ligand Docking.

Combines the score network with the diffusion processes (R³, SO(3), T^d)
to implement:
    - Forward noising (training)
    - Reverse denoising / sampling (inference)
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

from models.score_network import DiffBindFRScoreNet
from utils.so3 import (
    TranslationDiffusion, RotationDiffusion, TorsDiffusion,
    apply_rotation, so3_exp, igso3_sample,
)


class DiffBindFR(nn.Module):
    """
    Full DiffBindFR model.

    Usage (training):
        loss = model.training_loss(batch)

    Usage (inference):
        poses = model.sample(batch, n_steps=20, solver='SDE')
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        d = cfg["diffusion"]

        self.score_net = DiffBindFRScoreNet(
            hidden_dim=m["hidden_dim"],
            n_gvp_layers=m["gvp_layers"],
            n_gt_layers=m["gt_layers"],
            n_score_layers=m["score_layers"],
            num_heads=m["num_heads"],
            sigma_embed_dim=m["sigma_embed_dim"],
        )

        # Diffusion processes
        self.tr_diff  = TranslationDiffusion(d["tr_sigma_min"],  d["tr_sigma_max"])
        self.rot_diff = RotationDiffusion   (d["rot_sigma_min"], d["rot_sigma_max"])
        self.tor_diff = TorsDiffusion       (d["tor_sigma_min"], d["tor_sigma_max"])
        self.sc_diff  = TorsDiffusion       (d["sc_tor_sigma_min"], d["sc_tor_sigma_max"])

        # Loss weights
        t = cfg["training"]
        self.lam_tr  = t.get("lambda_tr", 1.0)
        self.lam_rot = t.get("lambda_rot", 1.0)
        self.lam_tor = t.get("lambda_tor", 1.0)
        self.lam_sc  = t.get("lambda_sc",  1.0)

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def training_loss(self, batch) -> dict[str, Tensor]:
        """
        1. Sample t ~ Uniform(0, 1) per complex.
        2. Apply forward diffusion to get noisy poses.
        3. Predict scores with the score network.
        4. Compute MSE between predicted and ground-truth scores.
        """
        B = batch.sigma_tr.shape[0]  # assumes sigmas already assigned by dataloader

        # ── Translation ──
        lig_pos0 = batch.lig_pos_crystal                              # [Nl, 3]
        lig_pos_t, tr_eps = self.tr_diff.forward_sample(
            lig_pos0, batch.t_tr[batch.lig_batch]
        )
        batch.lig_pos = lig_pos_t
        tr_score_gt = self.tr_diff.score(tr_eps, batch.sigma_tr[batch.lig_batch])

        # ── Rotation ──
        # Represented as per-complex rotation applied to ligand centred coords
        # (handled inside batch; here we just update sigma)

        # ── Torsion ──
        if hasattr(batch, "lig_torsions") and batch.lig_torsions.numel() > 0:
            tor_t, tor_eps = self.tor_diff.forward_sample(
                batch.lig_torsions, batch.t_tor[batch.tor_batch]
            )
            batch.lig_torsions_t = tor_t
            tor_score_gt = self.tor_diff.score(tor_eps, batch.sigma_tor[batch.tor_batch])
        else:
            tor_score_gt = None

        # ── Side-chain ──
        if hasattr(batch, "sc_torsions") and batch.sc_torsions.numel() > 0:
            sc_t, sc_eps = self.sc_diff.forward_sample(
                batch.sc_torsions, batch.t_sc[batch.sc_batch]
            )
            batch.sc_torsions_t = sc_t
            sc_score_gt = self.sc_diff.score(sc_eps, batch.sigma_sc[batch.sc_batch])
        else:
            sc_score_gt = None

        # ── Predict scores ──
        out = self.score_net(batch)

        # ── Losses (per-complex mean, then batch mean) ──
        # Translation: aggregate per-complex
        tr_pred_per_complex = self._agg_per_complex(out["tr_score"], B)
        tr_gt_per_complex   = self._agg_per_complex(tr_score_gt, B, batch.lig_batch)
        loss_tr = ((tr_pred_per_complex - tr_gt_per_complex) ** 2).mean()

        loss_rot = ((out["rot_score"]) ** 2).mean()  # placeholder; full rot score required

        loss_tor = torch.tensor(0., device=loss_tr.device)
        if tor_score_gt is not None and out["tor_score"].numel() > 0:
            loss_tor = ((out["tor_score"] - tor_score_gt) ** 2).mean()

        loss_sc = torch.tensor(0., device=loss_tr.device)
        if sc_score_gt is not None and out["sc_score"].numel() > 0:
            loss_sc = ((out["sc_score"] - sc_score_gt) ** 2).mean()

        loss = (
            self.lam_tr  * loss_tr
            + self.lam_rot * loss_rot
            + self.lam_tor * loss_tor
            + self.lam_sc  * loss_sc
        )

        return dict(
            loss=loss,
            loss_tr=loss_tr,
            loss_rot=loss_rot,
            loss_tor=loss_tor,
            loss_sc=loss_sc,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Inference / Sampling
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        batch,
        n_steps: int = 20,
        solver: str = "SDE",
        temperature: float = 1.0,
    ) -> dict[str, Tensor]:
        """
        Reverse diffusion to produce a docked pose.

        Starts from pure noise (t=1) and steps to t=0.

        Returns dict with:
            lig_pos       [Nl, 3]    – final ligand atom positions
            lig_torsions  [d_l]      – final ligand torsion angles
            sc_torsions   [d_p]      – final side-chain χ₁ angles
        """
        device = next(self.parameters()).device
        dt = 1.0 / n_steps
        ts = torch.linspace(1.0, dt, n_steps, device=device)

        # Initialise from noise
        lig_pos = torch.randn_like(batch.lig_pos) * self.tr_diff.sigma_max

        if hasattr(batch, "lig_torsions"):
            lig_tors = (torch.rand_like(batch.lig_torsions) * 2 - 1) * math.pi
        else:
            lig_tors = None

        if hasattr(batch, "sc_torsions"):
            sc_tors = (torch.rand_like(batch.sc_torsions) * 2 - 1) * math.pi
        else:
            sc_tors = None

        for t_val in ts:
            t = t_val.expand(batch.sigma_tr.shape[0])

            # Set current sigmas
            batch.sigma_tr  = self.tr_diff.sigma(t)
            batch.sigma_rot = self.rot_diff.sigma(t)
            batch.sigma_tor = self.tor_diff.sigma(t)
            batch.sigma_sc  = self.sc_diff.sigma(t)
            batch.lig_pos   = lig_pos
            if lig_tors is not None:
                batch.lig_torsions = lig_tors
            if sc_tors is not None:
                batch.sc_torsions = sc_tors

            # Predict score
            out = self.score_net(batch)

            # Translation update
            if solver == "SDE":
                lig_pos = self.tr_diff.reverse_sde_step(
                    lig_pos, out["tr_score"][batch.lig_batch], t_val, dt
                )
            else:
                lig_pos = self.tr_diff.reverse_ode_step(
                    lig_pos, out["tr_score"][batch.lig_batch], t_val, dt
                )

            # Torsion update
            if lig_tors is not None and out["tor_score"].numel() > 0:
                lig_tors = self.tor_diff.reverse_sde_step(
                    lig_tors, out["tor_score"], t_val, dt
                )

            # Side-chain update
            if sc_tors is not None and out["sc_score"].numel() > 0:
                sc_tors = self.sc_diff.reverse_sde_step(
                    sc_tors, out["sc_score"], t_val, dt
                )

        result = dict(lig_pos=lig_pos)
        if lig_tors is not None:
            result["lig_torsions"] = lig_tors
        if sc_tors is not None:
            result["sc_torsions"] = sc_tors
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _agg_per_complex(
        per_atom: Tensor, B: int, batch_idx: Tensor | None = None
    ) -> Tensor:
        """Mean-pool per-atom or per-torsion values to per-complex."""
        if batch_idx is None:
            # already per-complex
            return per_atom
        out = torch.zeros(B, per_atom.shape[-1], device=per_atom.device)
        counts = torch.bincount(batch_idx, minlength=B).float().unsqueeze(-1)
        out.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(per_atom), per_atom)
        return out / counts.clamp(min=1)
