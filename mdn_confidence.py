"""
MDN (Mixture Density Network) Confidence Model.

Given a docked complex, estimates a confidence score by modelling the
distribution of pairwise distances between pocket residues and ligand atoms
as a Gaussian mixture.

Used to rank the 40 generated poses and select the top-1.

Reference: DiffBindFR SI §5, and Méndez-Lucio et al., Nature Machine
Intelligence (2021).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

from models.gvp_encoder import GVPProteinEncoder
from models.ligand_encoder import LigandEncoder


class MDNConfidenceModel(nn.Module):
    """
    Architecture
    ─────────────
    Protein pocket   → GVP encoder    → v^p  [Np, D]
    Ligand graph     → Graph Transformer → v^l  [Nl, D]
                          ↓ outer product (pairwise)
                       [Np × Nl, 2D]
                          ↓ Feed-forward
                       (π, μ, σ)  for each pair  – K Gaussians
    Loss: negative log-likelihood of the true minimum pairwise distance
          + auxiliary cross-entropy on atom & bond type prediction

    Final confidence score U(x) = -∑_r ∑_s log P(d_rs | v^p_r, v^l_s)
    (lower U = higher confidence → higher score = −U)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_gvp_layers: int = 3,
        n_gt_layers: int = 3,
        K: int = 10,               # number of Gaussian mixture components
        drop_rate: float = 0.1,
        gvp_node_v_dim: int = 8,
        n_atom_types: int = 14,    # for auxiliary atom-type prediction
        n_bond_types: int = 4,     # for auxiliary bond-type prediction
    ):
        super().__init__()
        D = hidden_dim
        self.K = K
        self.D = D

        # Encoders (smaller than the score network)
        self.prot_enc = GVPProteinEncoder(
            hidden_dim=D,
            node_v_dim=gvp_node_v_dim,
            n_layers=n_gvp_layers,
            drop_rate=drop_rate,
        )
        self.lig_enc = LigandEncoder(
            hidden_dim=D,
            num_heads=4,
            n_layers=n_gt_layers,
            drop_rate=drop_rate,
        )

        # Feed-forward for MDN parameters from pairwise embeddings
        self.pair_ff = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, 3 * K),   # π (unnormalized), μ, log σ
        )

        # Auxiliary heads
        self.atom_head = nn.Linear(D, n_atom_types)
        self.bond_head = nn.Linear(D, n_bond_types)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, batch) -> dict[str, Tensor]:
        """
        Returns MDN parameters for each residue-atom pair.

        batch attributes:
            prot_*          – protein pocket graph
            lig_atom_feats, lig_bond_feats, lig_edge_index
            pro_batch, lig_batch  – [Np], [Nl]
        """
        pro_s, _ = self.prot_enc(batch)       # [Np, D]
        lig_s = self.lig_enc(
            batch.lig_atom_feats,
            batch.lig_bond_feats,
            batch.lig_edge_index,
        )                                       # [Nl, D]

        # Build all-pairs (residue, atom) within each complex
        B = batch.pro_batch.max().item() + 1
        mdn_params_list = []
        pair_info = []

        for b in range(B):
            pm = batch.pro_batch == b
            lm = batch.lig_batch == b
            ps = pro_s[pm]   # [np, D]
            ls = lig_s[lm]   # [nl, D]
            np_, nl = ps.shape[0], ls.shape[0]

            # Outer product feature: [np * nl, 2D]
            ps_exp = ps.unsqueeze(1).expand(-1, nl, -1).reshape(np_ * nl, self.D)
            ls_exp = ls.unsqueeze(0).expand(np_, -1, -1).reshape(np_ * nl, self.D)
            pair_feat = torch.cat([ps_exp, ls_exp], dim=-1)

            params = self.pair_ff(pair_feat)  # [np*nl, 3K]
            mdn_params_list.append(params)
            pair_info.append((np_, nl))

        # Auxiliary predictions
        atom_logits = self.atom_head(lig_s)  # [Nl, n_atom_types]
        bond_logits = self.bond_head(lig_s)  # [Nl, n_bond_types]  (node-level approximation)

        return dict(
            mdn_params=mdn_params_list,   # list of [np*nl, 3K] per complex
            pair_info=pair_info,
            atom_logits=atom_logits,
            bond_logits=bond_logits,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────────

    def compute_loss(self, batch, out: dict) -> Tensor:
        """
        L_total = L_MDN + L_atoms + L_bonds

        L_MDN = -log P(d_min | v^p, v^l)  averaged over pairs
        """
        mdn_loss = self._mdn_loss(batch, out["mdn_params"], out["pair_info"])
        atom_loss = F.cross_entropy(out["atom_logits"], batch.lig_atom_types)
        bond_loss = F.cross_entropy(out["bond_logits"], batch.lig_bond_types_node)
        return mdn_loss + 0.5 * atom_loss + 0.5 * bond_loss

    def _mdn_loss(
        self,
        batch,
        mdn_params_list: list,
        pair_info: list,
    ) -> Tensor:
        """
        For each (residue r, atom s) pair, compute the negative log-likelihood
        of the true minimum distance d_rs under the K-component Gaussian mixture.
        """
        total_nll = []
        offset_pro = 0
        offset_lig = 0

        for b, (params, (np_, nl)) in enumerate(zip(mdn_params_list, pair_info)):
            # True positions
            pm = batch.pro_batch == b
            lm = batch.lig_batch == b
            pro_pos = batch.pro_pos[pm]     # [np, 3]  (Cα positions)
            lig_pos = batch.lig_pos[lm]     # [nl, 3]

            # True minimum pairwise distances
            # pro_pos expanded: [np, nl, 3]
            diff = pro_pos.unsqueeze(1) - lig_pos.unsqueeze(0)  # [np, nl, 3]
            d_min = diff.norm(dim=-1).reshape(-1)  # [np*nl]

            # MDN parameters
            K = self.K
            pi_raw = params[:, :K]           # [np*nl, K]
            mu = params[:, K:2*K]            # [np*nl, K]
            log_sigma = params[:, 2*K:3*K]   # [np*nl, K]

            pi = F.softmax(pi_raw, dim=-1)   # [np*nl, K]
            sigma = torch.exp(log_sigma).clamp(min=1e-4)

            # log P(d | π, μ, σ) = log ∑_k π_k N(d; μ_k, σ_k)
            d_exp = d_min.unsqueeze(-1)      # [np*nl, 1]
            log_gauss = (
                -0.5 * ((d_exp - mu) / sigma) ** 2
                - sigma.log()
                - 0.5 * math.log(2 * math.pi)
            )  # [np*nl, K]
            log_p = torch.logsumexp(log_gauss + pi.log(), dim=-1)  # [np*nl]
            total_nll.append(-log_p.mean())

        return torch.stack(total_nll).mean()

    # ──────────────────────────────────────────────────────────────────────
    # Confidence score (inference)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score(self, batch) -> Tensor:
        """
        Compute the confidence score U(x) for a batch of docked complexes.

        Returns [B] tensor of scores. Higher = more confident.
        """
        out = self.forward(batch)
        scores = []

        for b, (params, (np_, nl)) in enumerate(zip(out["mdn_params"], out["pair_info"])):
            pm = batch.pro_batch == b
            lm = batch.lig_batch == b
            pro_pos = batch.pro_pos[pm]
            lig_pos = batch.lig_pos[lm]

            diff = pro_pos.unsqueeze(1) - lig_pos.unsqueeze(0)
            d_min = diff.norm(dim=-1).reshape(-1)

            K = self.K
            pi_raw = params[:, :K]
            mu = params[:, K:2*K]
            log_sigma = params[:, 2*K:3*K]

            pi = F.softmax(pi_raw, dim=-1)
            sigma = torch.exp(log_sigma).clamp(min=1e-4)

            d_exp = d_min.unsqueeze(-1)
            log_gauss = (
                -0.5 * ((d_exp - mu) / sigma) ** 2
                - sigma.log()
                - 0.5 * math.log(2 * math.pi)
            )
            log_p = torch.logsumexp(log_gauss + pi.log(), dim=-1)
            U = -log_p.sum()
            scores.append(-U)  # Higher = better

        return torch.stack(scores)
