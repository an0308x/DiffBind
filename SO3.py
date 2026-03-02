"""
Diffusion utilities for SE(3) flexible docking.

Covers:
  - R³  (translation) diffusion – simple Gaussian
  - SO(3) diffusion via IGSO3 (isotropic Gaussian on SO(3))
  - T^d (torus) diffusion for torsion angles

These are used in both the forward (noising) and reverse (denoising)
processes of DiffBindFR.

References:
  - DiffDock (Corso et al., 2023) – torsion + SO3 noise schedules
  - Leach et al., "Denoising Diffusion Probabilistic Models on SO(3)"
"""

import math
import numpy as np
import torch
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Noise schedules
# ──────────────────────────────────────────────────────────────────────────────

def log_linear_schedule(sigma_min: float, sigma_max: float, T: int) -> Tensor:
    """Linearly spaced sigmas in log space, from sigma_max (t=0) to sigma_min (t=T)."""
    return torch.exp(
        torch.linspace(math.log(sigma_max), math.log(sigma_min), T)
    )


def t_to_sigma(t: Tensor, sigma_min: float, sigma_max: float) -> Tensor:
    """Continuous σ(t) schedule  t ∈ [0, 1]."""
    return sigma_min ** (1 - t) * sigma_max ** t


# ──────────────────────────────────────────────────────────────────────────────
# R³ translation diffusion
# ──────────────────────────────────────────────────────────────────────────────

class TranslationDiffusion:
    """
    Simple isotropic Gaussian diffusion in R³.

    Forward:  x_t = x_0 + σ(t) * ε,     ε ~ N(0, I)
    Score:    ∇ log p(x_t | x_0) = -ε / σ(t)
    """

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 19.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t: Tensor) -> Tensor:
        return t_to_sigma(t, self.sigma_min, self.sigma_max)

    def forward_sample(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Sample x_t and return (x_t, noise ε)."""
        sig = self.sigma(t).view(-1, *([1] * (x0.dim() - 1)))
        eps = torch.randn_like(x0)
        return x0 + sig * eps, eps

    def score(self, eps: Tensor, t: Tensor) -> Tensor:
        sig = self.sigma(t).view(-1, *([1] * (eps.dim() - 1)))
        return -eps / sig

    def reverse_sde_step(
        self, x_t: Tensor, score: Tensor, t: Tensor, dt: float
    ) -> Tensor:
        """One Euler–Maruyama step of the reverse SDE."""
        sig = self.sigma(t).view(-1, *([1] * (x_t.dim() - 1)))
        dsig = sig * math.log(self.sigma_max / self.sigma_min) * dt
        drift = -(sig ** 2) * score
        diffusion = sig * math.sqrt(2 * abs(dt))
        return x_t - drift * dt + diffusion * torch.randn_like(x_t)

    def reverse_ode_step(
        self, x_t: Tensor, score: Tensor, t: Tensor, dt: float
    ) -> Tensor:
        """One step of the probability-flow ODE."""
        sig = self.sigma(t).view(-1, *([1] * (x_t.dim() - 1)))
        drift = -0.5 * (sig ** 2) * score
        return x_t - drift * dt


# ──────────────────────────────────────────────────────────────────────────────
# SO(3) rotation diffusion (IGSO3)
# ──────────────────────────────────────────────────────────────────────────────

def so3_hat(v: Tensor) -> Tensor:
    """Convert a 3-vector to a skew-symmetric matrix (Lie algebra element)."""
    zero = torch.zeros_like(v[..., 0])
    mat = torch.stack([
        torch.stack([  zero,  -v[..., 2],  v[..., 1]], dim=-1),
        torch.stack([ v[..., 2],   zero, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1],  v[..., 0],   zero], dim=-1),
    ], dim=-2)
    return mat


def so3_exp(omega: Tensor) -> Tensor:
    """
    Rodrigues' rotation formula: map axis-angle ω ∈ R³ → R ∈ SO(3).
    ω = angle * axis.
    """
    angle = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = omega / angle
    angle = angle.squeeze(-1)

    K = so3_hat(axis)                       # [..., 3, 3]
    I = torch.eye(3, device=omega.device).expand_as(K)
    c = torch.cos(angle)[..., None, None]
    s = torch.sin(angle)[..., None, None]
    return I + s * K + (1 - c) * (K @ K)


def igso3_sample(sigma: float, n: int, device=None) -> Tensor:
    """
    Sample n rotation matrices from the isotropic Gaussian on SO(3) with
    concentration parameter σ (in radians).

    Uses the axis-angle parameterisation:
        angle ~ IGSO3_angle(σ),  axis ~ Uniform(S²)
    """
    # Sample axis uniformly on S²
    axis = torch.randn(n, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True)

    # Sample angle from the marginal IGSO3 distribution (approximated by
    # a wrapped Normal with std σ, clipped to [0, π])
    angle = torch.randn(n, device=device) * sigma
    angle = angle.abs() % math.pi  # fold into [0, π]

    omega = axis * angle.unsqueeze(-1)
    return so3_exp(omega)  # [n, 3, 3]


def apply_rotation(R: Tensor, coords: Tensor) -> Tensor:
    """Apply rotation R [B, 3, 3] to centred coords [B, N, 3]."""
    return torch.einsum("bij,bnj->bni", R, coords)


class RotationDiffusion:
    """
    IGSO3 diffusion on SO(3).

    Forward:  R_t = R_noise @ R_0,  R_noise ~ IGSO3(σ(t))
    Score approximated as the negative perturbation axis-angle / σ².
    """

    def __init__(self, sigma_min: float = 0.03, sigma_max: float = 1.55):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t: Tensor) -> Tensor:
        return t_to_sigma(t, self.sigma_min, self.sigma_max)

    def forward_sample(self, R0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        R0: [B, 3, 3]
        Returns (R_t, noise_rotation R_eps)
        """
        sig = self.sigma(t).cpu().item() if t.numel() == 1 else self.sigma(t)
        B = R0.shape[0]
        R_eps = igso3_sample(float(sig), B, device=R0.device)
        R_t = R_eps @ R0
        return R_t, R_eps

    def score(self, R_eps: Tensor, t: Tensor) -> Tensor:
        """
        Approximate score (axis-angle of R_eps) / σ(t)².
        Returns axis-angle vector [B, 3].
        """
        # Axis-angle from rotation matrix (log map)
        angle = torch.acos(
            ((R_eps[:, 0, 0] + R_eps[:, 1, 1] + R_eps[:, 2, 2] - 1) / 2
             ).clamp(-1 + 1e-7, 1 - 1e-7)
        )  # [B]
        # Skew-symmetric part
        skew = (R_eps - R_eps.transpose(-1, -2)) / 2  # [B, 3, 3]
        axis = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1)
        safe_sin = torch.sin(angle).clamp(min=1e-7)
        axis = axis / safe_sin.unsqueeze(-1)
        omega = axis * angle.unsqueeze(-1)  # axis-angle [B, 3]

        sig = self.sigma(t)
        return -omega / (sig ** 2).view(-1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Torus (T^d) diffusion for torsion angles
# ──────────────────────────────────────────────────────────────────────────────

def torus_score(theta_noise: Tensor, sigma: Tensor) -> Tensor:
    """
    Score of a wrapped-Normal distribution on the circle.

    θ_noise: noisy torsion delta  [*]
    sigma:   noise level          [*]

    Uses the approximation that dominates for small σ:
        ∇ log p(θ | 0, σ) ≈ -θ / σ²
    with the principal-value θ wrapped to [-π, π].
    """
    theta_wrapped = (theta_noise + math.pi) % (2 * math.pi) - math.pi
    return -theta_wrapped / (sigma ** 2)


class TorsDiffusion:
    """
    Wrapped-Normal (torus) diffusion for torsion angles.

    Forward:  θ_t = θ_0 + σ(t) * ε,  ε ~ N(0,1), wrapped to [-π, π]
    Score approximated via torus_score.
    """

    def __init__(self, sigma_min: float = 0.0314, sigma_max: float = 3.14):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t: Tensor) -> Tensor:
        return t_to_sigma(t, self.sigma_min, self.sigma_max)

    def forward_sample(self, theta0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        theta0: [*, d] torsion angles in radians
        Returns (theta_t, eps)
        """
        sig = self.sigma(t).view(-1, *([1] * (theta0.dim() - 1)))
        eps = torch.randn_like(theta0)
        theta_t = (theta0 + sig * eps + math.pi) % (2 * math.pi) - math.pi
        return theta_t, eps

    def score(self, eps: Tensor, t: Tensor) -> Tensor:
        sig = self.sigma(t).view(-1, *([1] * (eps.dim() - 1)))
        return torus_score(sig * eps, sig)

    def reverse_sde_step(
        self, theta_t: Tensor, score: Tensor, t: Tensor, dt: float
    ) -> Tensor:
        sig = self.sigma(t).view(-1, *([1] * (theta_t.dim() - 1)))
        drift = -(sig ** 2) * score
        diffusion = sig * math.sqrt(2 * abs(dt))
        theta_new = theta_t - drift * dt + diffusion * torch.randn_like(theta_t)
        return (theta_new + math.pi) % (2 * math.pi) - math.pi
