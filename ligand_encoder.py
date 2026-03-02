"""
Graph Transformer ligand encoder.

Encodes a ligand molecular graph into per-atom node embeddings.
Follows the architecture of Dwivedi & Bresson (2020) extended with
atom/bond type features used in KarmaDock / DiffBindFR.

Reference: Dwivedi & Bresson, "A Generalization of Transformer Networks
to Graphs", arXiv 2012.09699 (2020).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ---------------------------------------------------------------------------
# Atom / Bond featurisation
# ---------------------------------------------------------------------------

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I",
    "B", "Si", "Se", "Te", "other",
]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
HYBRIDISATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "other"]
DEGREES = list(range(11))                 # 0..10
FORMAL_CHARGES = [-3, -2, -1, 0, 1, 2, 3]
N_ATOM_FEATS = (
    len(ATOM_TYPES)       # atom type
    + len(HYBRIDISATIONS) # hybridisation
    + len(DEGREES)        # degree
    + len(FORMAL_CHARGES) # formal charge
    + 2                   # is_in_ring, is_aromatic
)                         # total ≈ 44

N_BOND_FEATS = len(BOND_TYPES) + 2  # bond type + is_in_ring + is_conjugated


def atom_features(atom) -> Tensor:
    """Return a feature vector for an RDKit atom."""
    def one_hot(val, options):
        idx = options.index(val) if val in options else len(options) - 1
        v = [0] * len(options); v[idx] = 1
        return v

    feats = (
        one_hot(atom.GetSymbol(), ATOM_TYPES)
        + one_hot(str(atom.GetHybridization()), HYBRIDISATIONS)
        + one_hot(atom.GetDegree(), DEGREES)
        + one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + [int(atom.IsInRing()), int(atom.GetIsAromatic())]
    )
    return torch.tensor(feats, dtype=torch.float)


def bond_features(bond) -> Tensor:
    """Return a feature vector for an RDKit bond."""
    from rdkit.Chem import rdchem
    BTYPE = {
        rdchem.BondType.SINGLE: 0,
        rdchem.BondType.DOUBLE: 1,
        rdchem.BondType.TRIPLE: 2,
        rdchem.BondType.AROMATIC: 3,
    }
    bt = BTYPE.get(bond.GetBondType(), 0)
    v = [0] * 4; v[bt] = 1
    v += [int(bond.IsInRing()), int(bond.GetIsConjugated())]
    return torch.tensor(v, dtype=torch.float)


# ---------------------------------------------------------------------------
# Multi-head attention over graph edges (Graph Transformer layer)
# ---------------------------------------------------------------------------

class GTLayer(nn.Module):
    """
    Single Graph Transformer layer with edge features.

    Performs attention-weighted message passing where edge features modulate
    attention logits and are updated by the outer product of sender/receiver.
    """

    def __init__(self, hidden_dim: int, num_heads: int, drop_rate: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Q, K, V projections for nodes
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Edge feature projection into attention bias
        self.e_proj = nn.Linear(hidden_dim, num_heads, bias=False)
        # Edge feature update
        self.e_update = nn.Linear(hidden_dim * 2, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm_e = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: Tensor,          # [N, hidden_dim]
        e: Tensor,          # [E, hidden_dim]
        edge_index: Tensor, # [2, E]
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index  # E
        N = x.shape[0]
        H, D = self.num_heads, self.head_dim

        # Project Q, K, V
        Q = self.q_proj(x[dst]).view(-1, H, D)   # [E, H, D]
        K = self.k_proj(x[src]).view(-1, H, D)   # [E, H, D]
        V = self.v_proj(x[src]).view(-1, H, D)   # [E, H, D]

        # Attention logits + edge bias
        attn = (Q * K).sum(dim=-1) / math.sqrt(D)  # [E, H]
        attn = attn + self.e_proj(e)                # [E, H]

        # Softmax per destination node
        # Build sparse softmax: for each dst node, normalise over incoming edges
        attn_exp = torch.exp(attn - attn.max(dim=0, keepdim=True).values)  # [E, H]
        denom = torch.zeros(N, H, device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(-1).expand(-1, H), attn_exp)
        attn_norm = attn_exp / (denom[dst] + 1e-9)  # [E, H]

        # Aggregate
        agg = (attn_norm.unsqueeze(-1) * V).view(-1, self.hidden_dim)  # [E, hid]
        out = torch.zeros(N, self.hidden_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.hidden_dim), agg)
        out = self.dropout(self.out_proj(out))

        # Residual + norm
        x = self.norm1(x + out)

        # Update edge features (outer product of node pairs)
        e_new = self.e_update(torch.cat([x[src], x[dst]], dim=-1))
        e_new = self.dropout(e_new)
        e = self.norm_e(e + e_new)

        # Feed-forward on nodes
        x = self.norm2(x + self.dropout(self.ff(x)))

        return x, e


# ---------------------------------------------------------------------------
# Full ligand encoder
# ---------------------------------------------------------------------------

class LigandEncoder(nn.Module):
    """
    Graph Transformer encoder for ligand molecules.

    Takes atom and bond features, projects them into a shared hidden space,
    then runs `n_layers` of GTLayer to produce per-atom embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        n_layers: int = 6,
        drop_rate: float = 0.1,
        n_atom_feats: int = N_ATOM_FEATS,
        n_bond_feats: int = N_BOND_FEATS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input embeddings
        self.atom_embed = nn.Sequential(
            nn.Linear(n_atom_feats, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bond_embed = nn.Sequential(
            nn.Linear(n_bond_feats, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList([
            GTLayer(hidden_dim, num_heads, drop_rate)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        atom_feats: Tensor,   # [N, n_atom_feats]
        bond_feats: Tensor,   # [E, n_bond_feats]
        edge_index: Tensor,   # [2, E]
    ) -> Tensor:
        """Returns per-atom embeddings  [N, hidden_dim]."""
        x = self.atom_embed(atom_feats)
        e = self.bond_embed(bond_feats)

        for layer in self.layers:
            x, e = layer(x, e, edge_index)

        return x  # [N, hidden_dim]


# ---------------------------------------------------------------------------
# Utility: build ligand graph from RDKit mol
# ---------------------------------------------------------------------------

def mol_to_graph(mol) -> dict:
    """
    Convert an RDKit molecule to a dict with:
        atom_feats  [N, N_ATOM_FEATS]
        bond_feats  [E, N_BOND_FEATS]
        edge_index  [2, E]   (undirected – each bond appears twice)
        pos         [N, 3]   (if conformer available, else zeros)
    """
    atom_feats = torch.stack([atom_features(a) for a in mol.GetAtoms()])

    edges, edge_feats = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edges += [(i, j), (j, i)]
        edge_feats += [bf, bf]

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).T  # [2, E]
        bond_feats = torch.stack(edge_feats)                   # [E, N_BOND_FEATS]
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        bond_feats = torch.zeros(0, N_BOND_FEATS)

    # 3D positions
    try:
        conf = mol.GetConformer()
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
    except Exception:
        pos = torch.zeros(mol.GetNumAtoms(), 3)

    return dict(
        atom_feats=atom_feats,
        bond_feats=bond_feats,
        edge_index=edge_index,
        pos=pos,
        n_atoms=mol.GetNumAtoms(),
    )
