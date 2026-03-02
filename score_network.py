"""
DiffBindFR Score Network.

Given a noisy protein-ligand complex (at diffusion time t), predicts the
denoising score for:
    1. Ligand translation   (R³)
    2. Ligand rotation      (SO(3))
    3. Ligand torsion       (T^d)
    4. Pocket side-chain χ₁ (T^k)

Architecture:
    - GVP encodes the pocket residues → (node_s^p, node_v^p)
    - Graph Transformer encodes the ligand atoms → node_s^l
    - Cross-attention + equivariant message passing between protein & ligand
    - Separate MLP heads for each score component

The network is SE(3)-equivariant by construction:
    - Scalar outputs are invariant
    - Vector outputs transform correctly under rotations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from models.gvp_encoder import GVPProteinEncoder
from models.ligand_encoder import LigandEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Sigma / time embedding
# ──────────────────────────────────────────────────────────────────────────────

class SigmaEmbedding(nn.Module):
    """
    Embeds the diffusion noise level σ via sinusoidal Fourier features
    followed by a small MLP, following the DiffDock convention.
    """

    def __init__(self, embed_dim: int = 64, n_freqs: int = 32):
        super().__init__()
        self.n_freqs = n_freqs
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freqs, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, sigma: Tensor) -> Tensor:
        """sigma: [B] → [B, embed_dim]"""
        log_sigma = torch.log(sigma.clamp(min=1e-8))
        freqs = torch.arange(self.n_freqs, device=sigma.device, dtype=sigma.dtype)
        freqs = 2 ** freqs
        x = log_sigma.unsqueeze(-1) * freqs.unsqueeze(0)  # [B, n_freqs]
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.mlp(x)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-attention between ligand atoms and pocket residues
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Ligand → Protein and Protein → Ligand cross-attention.
    Processes pairwise interactions weighted by distance.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        d = hidden_dim

        self.q_lig = nn.Linear(d, d, bias=False)
        self.k_pro = nn.Linear(d, d, bias=False)
        self.v_pro = nn.Linear(d, d, bias=False)

        self.q_pro = nn.Linear(d, d, bias=False)
        self.k_lig = nn.Linear(d, d, bias=False)
        self.v_lig = nn.Linear(d, d, bias=False)

        self.out_lig = nn.Linear(d, d)
        self.out_pro = nn.Linear(d, d)

        self.norm_lig = nn.LayerNorm(d)
        self.norm_pro = nn.LayerNorm(d)

    def forward(
        self,
        lig: Tensor,        # [Nl, D]
        pro: Tensor,        # [Np, D]
        lig_pos: Tensor,    # [Nl, 3]
        pro_pos: Tensor,    # [Np, 3]
        lig_batch: Tensor,  # [Nl]   batch index
        pro_batch: Tensor,  # [Np]   batch index
    ) -> tuple[Tensor, Tensor]:
        """
        For simplicity, process each complex in batch independently.
        Returns updated (lig, pro).
        """
        B = lig_batch.max().item() + 1
        lig_out = lig.clone()
        pro_out = pro.clone()

        for b in range(B):
            lm = lig_batch == b
            pm = pro_batch == b
            l = lig[lm]    # [nl, D]
            p = pro[pm]    # [np, D]
            lp = lig_pos[lm]  # [nl, 3]
            pp = pro_pos[pm]  # [np, 3]

            # Distance bias: exp(-d/10)
            dist = torch.cdist(lp, pp)  # [nl, np]
            dist_bias = torch.exp(-dist / 10.0)

            # Ligand attends to protein
            Q = self.q_lig(l)   # [nl, D]
            K = self.k_pro(p)   # [np, D]
            V = self.v_pro(p)
            scale = math.sqrt(self.hidden_dim)
            attn = (Q @ K.T) / scale + dist_bias  # [nl, np]
            attn = F.softmax(attn, dim=-1)
            l_ctx = attn @ V  # [nl, D]
            lig_out[lm] = self.norm_lig(l + self.out_lig(l_ctx))

            # Protein attends to ligand
            Q2 = self.q_pro(p)
            K2 = self.k_lig(l)
            V2 = self.v_lig(l)
            attn2 = (Q2 @ K2.T) / scale + dist_bias.T
            attn2 = F.softmax(attn2, dim=-1)
            p_ctx = attn2 @ V2
            pro_out[pm] = self.norm_pro(p + self.out_pro(p_ctx))

        return lig_out, pro_out


# ──────────────────────────────────────────────────────────────────────────────
# Score heads
# ──────────────────────────────────────────────────────────────────────────────

def mlp_head(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


# ──────────────────────────────────────────────────────────────────────────────
# DiffBindFR Score Network
# ──────────────────────────────────────────────────────────────────────────────

class DiffBindFRScoreNet(nn.Module):
    """
    Full SE(3)-equivariant score network for flexible docking.

    Inputs
    ------
    Noisy protein-ligand complex at time t:
        - Pocket graph   (residue coords + features)
        - Ligand graph   (atom coords + features + torsion graph)
        - Diffusion time t (or per-component σ values)
        - Current ligand centroid, rotation, torsions, sc-torsions

    Outputs
    -------
        tr_score  : [B, 3]   – translation score
        rot_score : [B, 3]   – rotation score (axis-angle)
        tor_score : [B, d_l] – ligand torsion score
        sc_score  : [B, d_p] – side-chain χ₁ score
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_gvp_layers: int = 3,
        n_gt_layers: int = 6,
        n_score_layers: int = 6,
        num_heads: int = 8,
        sigma_embed_dim: int = 64,
        drop_rate: float = 0.1,
        gvp_node_v_dim: int = 16,
    ):
        super().__init__()
        D = hidden_dim

        # Encoders
        self.prot_enc = GVPProteinEncoder(
            hidden_dim=D,
            node_v_dim=gvp_node_v_dim,
            n_layers=n_gvp_layers,
            drop_rate=drop_rate,
        )
        self.lig_enc = LigandEncoder(
            hidden_dim=D,
            num_heads=num_heads,
            n_layers=n_gt_layers,
            drop_rate=drop_rate,
        )

        # Sigma embeddings (one per diffusion component)
        self.sigma_emb = SigmaEmbedding(sigma_embed_dim)

        # Cross-attention
        self.cross_attn = CrossAttention(D, num_heads=4)

        # Fusion: protein centroid + ligand centroid → combined
        self.fuse_lig = nn.Linear(D + sigma_embed_dim, D)
        self.fuse_pro = nn.Linear(D + gvp_node_v_dim * 3 + sigma_embed_dim, D)

        # Score prediction heads
        self.tr_head  = mlp_head(D, 3, D)     # translation score
        self.rot_head = mlp_head(D, 3, D)     # rotation score (axis-angle)
        self.tor_head = mlp_head(D, 1, D)     # per-torsion score
        self.sc_head  = mlp_head(D, 1, D)     # per-side-chain score

    def forward(self, batch) -> dict[str, Tensor]:
        """
        batch must have attributes:
            prot_*          – protein graph features
            lig_atom_feats  – [Nl, F_a]
            lig_bond_feats  – [El, F_b]
            lig_edge_index  – [2, El]
            lig_pos         – [Nl, 3]  current noisy ligand coords
            pro_pos         – [Np, 3]  pocket Cα positions
            sigma_tr/rot/tor/sc – [B] noise levels
            lig_batch       – [Nl] batch indices
            pro_batch       – [Np] batch indices
            tor_edge_index  – [2, E_t]  torsion bond pairs
            sc_residue_idx  – [K]       which residues have sc torsions
        """
        # Encode pocket
        pro_s, pro_v = self.prot_enc(batch)     # [Np, D], [Np, nv, 3]

        # Encode ligand
        lig_s = self.lig_enc(
            batch.lig_atom_feats,
            batch.lig_bond_feats,
            batch.lig_edge_index,
        )  # [Nl, D]

        # Sigma embeddings
        sigma_tr  = self.sigma_emb(batch.sigma_tr)   # [B, se]
        sigma_rot = self.sigma_emb(batch.sigma_rot)
        sigma_tor = self.sigma_emb(batch.sigma_tor)
        sigma_sc  = self.sigma_emb(batch.sigma_sc)

        # Expand sigma to per-atom / per-residue via batch index
        lig_sigma = sigma_tr[batch.lig_batch]   # [Nl, se]
        pro_sigma = sigma_sc[batch.pro_batch]   # [Np, se]

        # Fuse sigma into node features
        lig_s = self.fuse_lig(torch.cat([lig_s, lig_sigma], dim=-1))

        pro_v_flat = pro_v.view(pro_v.shape[0], -1)  # [Np, nv*3]
        pro_s = self.fuse_pro(torch.cat([pro_s, pro_v_flat, pro_sigma], dim=-1))

        # Cross-attention between protein and ligand
        lig_s, pro_s = self.cross_attn(
            lig_s, pro_s,
            batch.lig_pos, batch.pro_pos,
            batch.lig_batch, batch.pro_batch,
        )

        # ── Translation score: mean-pool ligand atoms per complex ──
        B = batch.sigma_tr.shape[0]
        lig_mean = torch.zeros(B, lig_s.shape[-1], device=lig_s.device)
        lig_mean.scatter_add_(
            0,
            batch.lig_batch.unsqueeze(-1).expand_as(lig_s),
            lig_s,
        )
        counts = torch.bincount(batch.lig_batch, minlength=B).float().unsqueeze(-1)
        lig_mean = lig_mean / counts.clamp(min=1)

        tr_score  = self.tr_head(lig_mean)   # [B, 3]
        rot_score = self.rot_head(lig_mean)  # [B, 3]

        # ── Torsion score: per torsion-bond atom pair ──
        if hasattr(batch, "tor_edge_index") and batch.tor_edge_index.shape[1] > 0:
            ti, tj = batch.tor_edge_index
            tor_feat = (lig_s[ti] + lig_s[tj]) / 2.0  # [E_t, D]
            tor_score = self.tor_head(tor_feat).squeeze(-1)  # [E_t]
        else:
            tor_score = torch.zeros(0, device=lig_s.device)

        # ── Side-chain score: per residue ──
        if hasattr(batch, "sc_residue_idx") and len(batch.sc_residue_idx) > 0:
            sc_feat = pro_s[batch.sc_residue_idx]  # [K, D]
            sc_score = self.sc_head(sc_feat).squeeze(-1)  # [K]
        else:
            sc_score = torch.zeros(0, device=pro_s.device)

        return dict(
            tr_score=tr_score,
            rot_score=rot_score,
            tor_score=tor_score,
            sc_score=sc_score,
        )
