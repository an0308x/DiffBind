"""
Unit tests for DiffBindFR components.

Run with:
    pytest tests/test_models.py -v
"""

import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ──────────────────────────────────────────────────────────────────────────────
# GVP encoder tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGVP:
    def test_gvp_output_shape(self):
        from models.gvp_encoder import GVP
        layer = GVP((32, 4), (64, 8))
        s = torch.randn(10, 32)
        v = torch.randn(10, 4, 3)
        s_out, v_out = layer((s, v))
        assert s_out.shape == (10, 64)
        assert v_out.shape == (10, 8, 3)

    def test_gvp_equivariance(self):
        """Output vector features must rotate consistently with the input."""
        from models.gvp_encoder import GVP
        torch.manual_seed(0)
        layer = GVP((16, 4), (16, 4))

        s = torch.randn(5, 16)
        v = torch.randn(5, 4, 3)

        # Random rotation
        R = torch.linalg.qr(torch.randn(3, 3))[0]

        s_out, v_out         = layer((s, v))
        s_out_r, v_out_r     = layer((s, torch.einsum("ij,bkj->bki", R, v)))

        # Scalar outputs must be identical (invariant)
        assert torch.allclose(s_out, s_out_r, atol=1e-5), \
            "GVP scalars are not rotation-invariant"

        # Vector outputs must transform as R v
        v_out_expected = torch.einsum("ij,bkj->bki", R, v_out)
        assert torch.allclose(v_out_r, v_out_expected, atol=1e-5), \
            "GVP vectors are not rotation-equivariant"

    def test_gvp_layer_norm(self):
        from models.gvp_encoder import GVPLayerNorm
        norm = GVPLayerNorm((32, 8))
        s = torch.randn(4, 32)
        v = torch.randn(4, 8, 3)
        s_n, v_n = norm((s, v))
        assert s_n.shape == s.shape
        assert v_n.shape == v.shape


# ──────────────────────────────────────────────────────────────────────────────
# Ligand encoder tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLigandEncoder:
    def test_gt_layer_output_shape(self):
        from models.ligand_encoder import GTLayer
        layer = GTLayer(hidden_dim=64, num_heads=4)
        N, E = 12, 30
        x = torch.randn(N, 64)
        e = torch.randn(E, 64)
        src = torch.randint(0, N, (E,))
        dst = torch.randint(0, N, (E,))
        edge_index = torch.stack([src, dst])
        x_out, e_out = layer(x, e, edge_index)
        assert x_out.shape == (N, 64)
        assert e_out.shape == (E, 64)

    def test_ligand_encoder_forward(self):
        from models.ligand_encoder import LigandEncoder, N_ATOM_FEATS, N_BOND_FEATS
        enc = LigandEncoder(hidden_dim=64, num_heads=4, n_layers=2,
                            n_atom_feats=N_ATOM_FEATS, n_bond_feats=N_BOND_FEATS)
        N, E = 10, 20
        atom_f = torch.randn(N, N_ATOM_FEATS)
        bond_f = torch.randn(E, N_BOND_FEATS)
        edge_idx = torch.stack([torch.randint(0, N, (E,)), torch.randint(0, N, (E,))])
        out = enc(atom_f, bond_f, edge_idx)
        assert out.shape == (N, 64)


# ──────────────────────────────────────────────────────────────────────────────
# Diffusion utilities tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDiffusion:
    def test_translation_forward_sample(self):
        from utils.so3 import TranslationDiffusion
        diff = TranslationDiffusion()
        x0 = torch.randn(4, 3)
        t  = torch.rand(4)
        xt, eps = diff.forward_sample(x0, t)
        assert xt.shape == x0.shape
        assert eps.shape == x0.shape

    def test_torsion_score_wrapping(self):
        from utils.so3 import torus_score
        # Angles near π should be wrapped correctly
        theta = torch.tensor([3.0, -3.0, 0.5, -0.5])
        sigma = torch.ones_like(theta)
        score = torus_score(theta, sigma)
        assert score.shape == theta.shape
        # Score should be negative for positive angles (pointing toward 0)
        assert score[2].item() < 0

    def test_so3_exp_identity(self):
        from utils.so3 import so3_exp
        omega = torch.zeros(3, 3)  # zero rotation
        R = so3_exp(omega)
        I = torch.eye(3).unsqueeze(0).expand(3, -1, -1)
        assert torch.allclose(R, I, atol=1e-6)

    def test_rotation_diffusion(self):
        from utils.so3 import RotationDiffusion
        diff = RotationDiffusion()
        # Check that sigma schedule spans expected range
        t0 = torch.tensor([0.0])
        t1 = torch.tensor([1.0])
        assert diff.sigma(t0).item() < diff.sigma(t1).item()


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation metrics tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_rmsd_identical(self):
        from evaluation.metrics import rmsd
        x = torch.randn(10, 3)
        assert rmsd(x, x).item() < 1e-6

    def test_kabsch_rmsd_invariant_to_rotation(self):
        from evaluation.metrics import kabsch_rmsd
        x = torch.randn(10, 3)
        R, _ = torch.linalg.qr(torch.randn(3, 3))
        x_rot = x @ R.T
        kr = kabsch_rmsd(x_rot, x)
        assert kr.item() < 1e-4, f"Kabsch RMSD after rotation: {kr.item()}"

    def test_centroid_distance(self):
        from evaluation.metrics import centroid_distance
        x = torch.randn(5, 3)
        y = x + torch.tensor([1., 0., 0.])
        assert abs(centroid_distance(x, y) - 1.0) < 1e-5

    def test_chi1_accuracy(self):
        from evaluation.metrics import chi1_accuracy
        pred = torch.tensor([0.1, 0.2, 1.0])
        ref  = torch.tensor([0.0, 0.0, 0.0])
        acc  = chi1_accuracy(pred, ref, threshold=15.0)
        # 0.1 and 0.2 rad = 5.7° and 11.5° → both < 15°; 1.0 rad = 57° → fails
        assert abs(acc - 2/3) < 0.01

    def test_success_rate(self):
        from evaluation.metrics import success_rate
        rmsds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        sr = success_rate(rmsds, threshold=2.0)
        assert abs(sr - 3/6) < 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# MDN confidence model test
# ──────────────────────────────────────────────────────────────────────────────

class TestMDN:
    def _make_dummy_batch(self):
        import types
        from models.ligand_encoder import N_ATOM_FEATS, N_BOND_FEATS
        B, Nl, Np = 2, 8, 12
        batch = types.SimpleNamespace(
            node_s     = torch.randn(Np, 27),
            node_v     = torch.randn(Np, 3, 3),
            edge_s     = torch.randn(Np*3, 20),
            edge_v     = torch.randn(Np*3, 1, 3),
            edge_index = torch.randint(0, Np, (2, Np*3)),
            pro_pos    = torch.randn(Np, 3),
            pro_batch  = torch.cat([torch.zeros(Np//2, dtype=torch.long),
                                     torch.ones(Np//2,  dtype=torch.long)]),
            lig_atom_feats = torch.randn(Nl, N_ATOM_FEATS),
            lig_bond_feats = torch.randn(Nl*2, N_BOND_FEATS),
            lig_edge_index = torch.randint(0, Nl, (2, Nl*2)),
            lig_pos    = torch.randn(Nl, 3),
            lig_batch  = torch.cat([torch.zeros(Nl//2, dtype=torch.long),
                                     torch.ones(Nl//2,  dtype=torch.long)]),
            lig_atom_types      = torch.randint(0, 14, (Nl,)),
            lig_bond_types_node = torch.randint(0, 4,  (Nl,)),
        )
        return batch

    def test_mdn_forward(self):
        from models.mdn_confidence import MDNConfidenceModel
        model = MDNConfidenceModel(hidden_dim=64, K=4, n_gvp_layers=1, n_gt_layers=1,
                                    gvp_node_v_dim=4)
        batch = self._make_dummy_batch()
        out = model(batch)
        assert "mdn_params" in out
        assert len(out["mdn_params"]) == 2  # B=2

    def test_mdn_score(self):
        from models.mdn_confidence import MDNConfidenceModel
        model = MDNConfidenceModel(hidden_dim=64, K=4, n_gvp_layers=1, n_gt_layers=1,
                                    gvp_node_v_dim=4)
        batch = self._make_dummy_batch()
        scores = model.score(batch)
        assert scores.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
