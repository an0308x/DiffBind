"""
GVP (Geometric Vector Perceptron) protein backbone encoder.

Encodes protein pocket residues into (scalar, vector) node embeddings
that are equivariant under SE(3) transformations.

Reference: Jing et al., "Learning from Protein Structure with Geometric
Vector Perceptrons", ICLR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


# ---------------------------------------------------------------------------
# GVP primitive
# ---------------------------------------------------------------------------

class GVP(nn.Module):
    """
    Single Geometric Vector Perceptron layer.

    Processes a pair (s, V) where:
        s: scalar features  [*, n_s]
        V: vector features  [*, n_v, 3]

    Returns updated (s_out, V_out).
    """

    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        h_dim: int | None = None,
        activations: Tuple = (F.relu, torch.sigmoid),
        vector_gate: bool = True,
    ):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate

        h_dim = h_dim or max(self.vi, self.vo)

        # Vector branch
        self.W_h = nn.Linear(self.vi, h_dim, bias=False)
        self.W_V = nn.Linear(h_dim, self.vo, bias=False)

        # Scalar branch – receives scalars + norms of hidden vectors
        self.W_s = nn.Linear(self.si + h_dim, self.so)

        if vector_gate and self.vo > 0:
            self.W_gate = nn.Linear(self.so, self.vo)

        self.scalar_act, self.vector_act = activations

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        s, V = x  # s: [B, si], V: [B, vi, 3]

        # Vector branch
        Vh = torch.einsum("...ij,jk->...ik", V, self.W_h.weight.T)  # [B, h, 3]
        Vh_norm = torch.norm(Vh, dim=-1)                              # [B, h]
        V_out = torch.einsum("...ij,jk->...ik", Vh, self.W_V.weight.T)  # [B, vo, 3]

        # Scalar branch
        s_cat = torch.cat([s, Vh_norm], dim=-1)
        s_out = self.W_s(s_cat)
        if self.scalar_act is not None:
            s_out = self.scalar_act(s_out)

        # Optional vector gating
        if self.vector_gate and self.vo > 0:
            gate = torch.sigmoid(self.W_gate(s_out)).unsqueeze(-1)  # [B, vo, 1]
            V_out = V_out * gate
        elif self.vector_act is not None and self.vo > 0:
            V_out = self.vector_act(torch.norm(V_out, dim=-1, keepdim=True)) * V_out

        return s_out, V_out


# ---------------------------------------------------------------------------
# GVP Layer Norm
# ---------------------------------------------------------------------------

class GVPLayerNorm(nn.Module):
    """Layer norm for (scalar, vector) pairs."""

    def __init__(self, dims: Tuple[int, int], eps: float = 1e-8):
        super().__init__()
        self.ns, self.nv = dims
        self.scalar_norm = nn.LayerNorm(self.ns)
        self.eps = eps

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        s, V = x
        s = self.scalar_norm(s)
        if self.nv > 0:
            # Normalise over vector dimension, preserving orientation
            rms = torch.sqrt(torch.mean(V ** 2, dim=(-2, -1), keepdim=True) + self.eps)
            V = V / rms
        return s, V


# ---------------------------------------------------------------------------
# GVP-GNN message passing layer
# ---------------------------------------------------------------------------

class GVPConv(nn.Module):
    """
    One round of GVP-based message passing over a protein graph.

    Node features: (s_node, V_node)
    Edge features: (s_edge, V_edge)  –  V_edge contains unit displacement vectors
    """

    def __init__(
        self,
        node_dims: Tuple[int, int],
        edge_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        n_message: int = 3,
        n_feedforward: int = 2,
        drop_rate: float = 0.1,
        vector_gate: bool = True,
    ):
        super().__init__()

        ns, nv = node_dims
        es, ev = edge_dims
        os, ov = out_dims

        # Message MLPs (GVP stack)
        msg_in_dims = (2 * ns + es, 2 * nv + ev)
        self.message_gvps = nn.Sequential(
            *[GVP(msg_in_dims if i == 0 else (ns, nv),
                  (ns, nv) if i < n_message - 1 else out_dims,
                  vector_gate=vector_gate)
              for i in range(n_message)]
        )

        # Update feedforward
        self.ff_gvps = nn.Sequential(
            *[GVP(out_dims if i == 0 else out_dims,
                  out_dims,
                  vector_gate=vector_gate)
              for i in range(n_feedforward)]
        )

        self.norm1 = GVPLayerNorm(out_dims)
        self.norm2 = GVPLayerNorm(out_dims)
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        node_s: Tensor, node_v: Tensor,
        edge_s: Tensor, edge_v: Tensor,
        edge_index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        node_s: [N, ns]
        node_v: [N, nv, 3]
        edge_s: [E, es]
        edge_v: [E, ev, 3]
        edge_index: [2, E]  (source, target)
        """
        src, dst = edge_index

        # Build message input by concatenating sender + receiver + edge
        msg_s = torch.cat([node_s[src], node_s[dst], edge_s], dim=-1)
        msg_v = torch.cat([node_v[src], node_v[dst], edge_v], dim=-2)

        msg_s, msg_v = self._run_sequential(self.message_gvps, (msg_s, msg_v))

        # Aggregate (sum) messages at each destination node
        agg_s = torch.zeros_like(node_s[:, :msg_s.shape[-1]]) if msg_s.shape[-1] != node_s.shape[-1] else torch.zeros_like(node_s)
        agg_v = torch.zeros(node_s.shape[0], msg_v.shape[-2], 3, device=node_s.device)
        agg_s = agg_s.scatter_add(0, dst.unsqueeze(-1).expand_as(msg_s), msg_s)
        for i in range(msg_v.shape[-2]):
            agg_v[:, i, :] = agg_v[:, i, :].scatter_add(0, dst.unsqueeze(-1).expand(msg_v.shape[0], 3), msg_v[:, i, :])

        # Residual + norm
        # Pad node features if dimensions differ
        if agg_s.shape[-1] == node_s.shape[-1]:
            agg_s = agg_s + node_s
        agg_s, agg_v = self.norm1((agg_s, agg_v))

        # Feedforward
        ff_s, ff_v = self._run_sequential(self.ff_gvps, (agg_s, agg_v))
        ff_s = self.dropout(ff_s)
        out_s = agg_s + ff_s
        out_v = agg_v + ff_v
        out_s, out_v = self.norm2((out_s, out_v))

        return out_s, out_v

    @staticmethod
    def _run_sequential(seq, x):
        for layer in seq:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Full GVP Protein Encoder
# ---------------------------------------------------------------------------

class GVPProteinEncoder(nn.Module):
    """
    Encodes a protein pocket graph into per-residue (scalar, vector) embeddings.

    Input node features (per residue):
        - One-hot amino acid type  [20]
        - Backbone dihedral sines/cosines  [6]
        - Solvent accessible surface  [1]
        Total scalar: 27

    Input edge features (per residue pair):
        - RBF-encoded distance  [16]
        - Orientation quaternion  [4]
        Total scalar: 20, vector: 1 (unit displacement)

    Output: node_s  [N, hidden_dim],  node_v  [N, nv, 3]
    """

    NODE_S_DIM = 27
    NODE_V_DIM = 3       # 3 backbone unit vectors (N→CA, CA→C, CB direction)
    EDGE_S_DIM = 20
    EDGE_V_DIM = 1

    def __init__(
        self,
        hidden_dim: int = 256,
        node_v_dim: int = 16,
        n_layers: int = 3,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        node_out_dims = (hidden_dim, node_v_dim)
        edge_dims = (self.EDGE_S_DIM, self.EDGE_V_DIM)

        # Input embedding
        self.node_embed = GVP(
            (self.NODE_S_DIM, self.NODE_V_DIM),
            node_out_dims,
            vector_gate=True,
        )

        # GVP message-passing layers
        self.layers = nn.ModuleList([
            GVPConv(
                node_dims=node_out_dims,
                edge_dims=edge_dims,
                out_dims=node_out_dims,
                drop_rate=drop_rate,
            )
            for _ in range(n_layers)
        ])

    def forward(self, batch) -> Tuple[Tensor, Tensor]:
        """
        batch must have:
            node_s  [N, NODE_S_DIM]
            node_v  [N, NODE_V_DIM, 3]
            edge_s  [E, EDGE_S_DIM]
            edge_v  [E, EDGE_V_DIM, 3]
            edge_index [2, E]
        Returns:
            node_s  [N, hidden_dim]
            node_v  [N, node_v_dim, 3]
        """
        s, v = self.node_embed((batch.node_s, batch.node_v))
        for layer in self.layers:
            s, v = layer(s, v, batch.edge_s, batch.edge_v, batch.edge_index)
        return s, v


# ---------------------------------------------------------------------------
# Residue featurisation helpers
# ---------------------------------------------------------------------------

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def one_hot_aa(seq: str) -> Tensor:
    """Return [L, 20] one-hot encoding of an amino acid sequence."""
    idx = [AA_TO_IDX.get(aa, 0) for aa in seq]
    return F.one_hot(torch.tensor(idx, dtype=torch.long), num_classes=20).float()


def rbf_encode(d: Tensor, d_min: float = 0., d_max: float = 20., n_bins: int = 16) -> Tensor:
    """Radial basis function distance encoding  [*, n_bins]."""
    centers = torch.linspace(d_min, d_max, n_bins, device=d.device)
    sigma = (d_max - d_min) / n_bins
    return torch.exp(-((d.unsqueeze(-1) - centers) ** 2) / (2 * sigma ** 2))
