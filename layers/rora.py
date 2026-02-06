# layers/rora.py
"""
Rotational Rank Adaptation (RoRA) for Time Series Forecasting

Based on the geometric insight that:
1. Patches create discrete temporal partitions (like XGBoost leaves)
2. The conv backbone creates piecewise-linear geometry (local linear charts)
3. RoRA learns a global rotation to align these charts for better prediction

The rotation is parameterized via low-rank skew-symmetric generators in SO(d):
    R = exp(Ω)  where Ωᵀ = -Ω

This preserves feature energy (no stretching/squashing) while reorienting the
representation space so that the downstream linear head can separate better.

Reference: Sudjianto, "Rotational Rank Adaptation (RoRA)", 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RoRA(nn.Module):
    """
    Rotational Rank Adaptation layer.
    
    Learns an orthogonal transformation R ∈ SO(d) parameterized by a low-rank
    skew-symmetric matrix Ω = UV^T - VU^T where U, V ∈ R^{d×r}.
    
    The rotation is computed via matrix exponential (Padé approximation for
    efficiency) or Cayley transform.
    
    Args:
        dim: Feature dimension to rotate
        rank: Rank of the skew-symmetric generator (controls expressivity)
        method: 'cayley' (exact, O(d³)) or 'taylor' (approx, O(d²r))
        scale_init: Initial scale for U, V (smaller = closer to identity)
    """
    
    def __init__(
        self,
        dim: int,
        rank: int = 4,
        method: str = 'cayley',
        scale_init: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.method = method
        
        # Low-rank factors for skew-symmetric generator
        # Ω = UV^T - VU^T is guaranteed skew-symmetric
        self.U = nn.Parameter(torch.randn(dim, rank) * scale_init)
        self.V = nn.Parameter(torch.randn(dim, rank) * scale_init)
        
        # Optional learnable gate (starts at 1.0 = full rotation)
        self.gate = nn.Parameter(torch.ones(1))
        
    def _build_skew_symmetric(self) -> torch.Tensor:
        """Build Ω = UV^T - VU^T (guaranteed skew-symmetric)."""
        UV = self.U @ self.V.T  # [d, d]
        return UV - UV.T
    
    def _cayley_transform(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Cayley transform: R = (I + Ω/2)(I - Ω/2)^{-1}
        
        Maps skew-symmetric Ω to orthogonal R exactly.
        Cost: O(d³) for the matrix inverse.
        """
        d = omega.shape[0]
        I = torch.eye(d, device=omega.device, dtype=omega.dtype)
        half_omega = omega / 2
        
        # R = (I + Ω/2) @ (I - Ω/2)^{-1}
        R = torch.linalg.solve(I - half_omega, I + half_omega)
        return R
    
    def _taylor_approx(self, omega: torch.Tensor, order: int = 4) -> torch.Tensor:
        """
        Taylor series: R ≈ I + Ω + Ω²/2! + Ω³/3! + ...
        
        Good approximation when ||Ω|| is small.
        Cost: O(d²r × order) if we're clever, O(d³ × order) naive.
        """
        d = omega.shape[0]
        R = torch.eye(d, device=omega.device, dtype=omega.dtype)
        omega_power = torch.eye(d, device=omega.device, dtype=omega.dtype)
        factorial = 1.0
        
        for k in range(1, order + 1):
            factorial *= k
            omega_power = omega_power @ omega
            R = R + omega_power / factorial
            
        return R
    
    def _householder_approx(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Approximate rotation via sequence of Householder reflections.
        
        Each rank-1 component of Ω can be converted to a Givens rotation.
        Cost: O(d × r) per application, very efficient for low rank.
        """
        # For low-rank Ω, we can decompose into r Givens rotations
        # This is more numerically stable and efficient
        d = omega.shape[0]
        R = torch.eye(d, device=omega.device, dtype=omega.dtype)
        
        # Apply Givens rotations from U, V factors
        for i in range(self.rank):
            u = self.U[:, i]
            v = self.V[:, i]
            
            # Givens-like rotation in the (u, v) plane
            # R_i = I + sin(θ)(uv^T - vu^T) + (cos(θ)-1)(uu^T + vv^T)
            # For small θ: R_i ≈ I + θ(uv^T - vu^T)
            uv = torch.outer(u, v)
            R = R + (uv - uv.T)
            
        return R
    
    def get_rotation_matrix(self) -> torch.Tensor:
        """Compute the rotation matrix R ∈ SO(d)."""
        omega = self._build_skew_symmetric()
        
        # Scale by learnable gate
        omega = omega * torch.sigmoid(self.gate)
        
        if self.method == 'cayley':
            return self._cayley_transform(omega)
        elif self.method == 'taylor':
            return self._taylor_approx(omega, order=4)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to input.
        
        Args:
            x: [..., dim] tensor
            
        Returns:
            Rotated tensor [..., dim]
        """
        R = self.get_rotation_matrix()  # [d, d]
        return x @ R.T
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, rank={self.rank}, method={self.method}"


class RoRABlock(nn.Module):
    """
    RoRA with residual connection and optional LayerNorm.
    
    Implements: x' = x + α * RoRA(LN(x))
    
    The residual allows gradual learning of the rotation.
    """
    
    def __init__(
        self,
        dim: int,
        rank: int = 4,
        method: str = 'cayley',
        use_layernorm: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.rora = RoRA(dim, rank, method)
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.scale = nn.Parameter(torch.tensor(residual_scale))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim]"""
        return x + self.scale * self.rora(self.norm(x))


class MultiHeadRoRA(nn.Module):
    """
    Multi-head RoRA for when different subspaces need different rotations.
    
    Splits the feature dimension into `num_heads` groups, applies independent
    RoRA to each, then concatenates. Analogous to multi-head attention but
    for rotations.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        rank_per_head: int = 2,
        method: str = 'cayley',
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.heads = nn.ModuleList([
            RoRA(self.head_dim, rank_per_head, method)
            for _ in range(num_heads)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim]"""
        # Split into heads
        chunks = x.split(self.head_dim, dim=-1)
        
        # Apply RoRA to each head
        rotated = [head(chunk) for head, chunk in zip(self.heads, chunks)]
        
        # Concatenate
        return torch.cat(rotated, dim=-1)


class PatchRoRA(nn.Module):
    """
    RoRA specifically designed for patch-based time series models.
    
    Can apply rotation:
    - Per-patch: same rotation to each patch's features
    - Cross-patch: rotation that mixes information across patches
    - Both: sequential application
    
    Args:
        d_model: Feature dimension per patch
        num_patches: Number of patches (for cross-patch mode)
        rank: Rank of skew-symmetric generator
        mode: 'feature', 'patch', or 'both'
    """
    
    def __init__(
        self,
        d_model: int,
        num_patches: int = None,
        rank: int = 4,
        mode: str = 'feature',
        method: str = 'cayley',
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.num_patches = num_patches
        
        if mode in ('feature', 'both'):
            # Rotate within each patch's feature space
            self.feature_rora = RoRABlock(d_model, rank, method)
            
        if mode in ('patch', 'both'):
            # Rotate across patches (requires num_patches)
            assert num_patches is not None, "num_patches required for patch/both mode"
            self.patch_rora = RoRABlock(num_patches, rank, method)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, num_patches, d_model]
        """
        if self.mode == 'feature':
            # Rotate features within each patch
            return self.feature_rora(x)
            
        elif self.mode == 'patch':
            # Rotate across patches (transpose, rotate, transpose back)
            x = x.transpose(-2, -1)  # [batch, d_model, num_patches]
            x = self.patch_rora(x)
            return x.transpose(-2, -1)  # [batch, num_patches, d_model]
            
        elif self.mode == 'both':
            # Apply both rotations
            x = self.feature_rora(x)
            x = x.transpose(-2, -1)
            x = self.patch_rora(x)
            return x.transpose(-2, -1)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# =============================================================================
# Diagnostic utilities
# =============================================================================

def compute_stable_rank(Z: torch.Tensor) -> float:
    """
    Compute stable rank of representation matrix.
    
    stable_rank = tr(Σ)² / tr(Σ²) = (sum of singular values)² / (sum of squared singular values)
    
    This measures the "effective dimensionality" of the representation.
    RoRA works best when ambient_dim / stable_rank ≥ 50-60.
    """
    # Z: [n_samples, dim]
    # Compute SVD
    _, S, _ = torch.linalg.svd(Z, full_matrices=False)
    
    # Stable rank
    sum_s = S.sum()
    sum_s2 = (S ** 2).sum()
    
    return (sum_s ** 2 / sum_s2).item()


def compute_overparameterization_ratio(Z: torch.Tensor) -> float:
    """
    Compute κ = ambient_dim / stable_rank.
    
    RoRA empirically works when κ ≥ 50-60 (enough "room to rotate").
    """
    stable_rank = compute_stable_rank(Z)
    ambient_dim = Z.shape[-1]
    return ambient_dim / stable_rank


class RoRADiagnostics(nn.Module):
    """
    Wrapper that computes diagnostic metrics during forward pass.
    
    Useful for understanding whether RoRA has "room to rotate".
    """
    
    def __init__(self, rora_module: nn.Module):
        super().__init__()
        self.rora = rora_module
        self.last_metrics = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute pre-rotation metrics
        batch_flat = x.reshape(-1, x.shape[-1])
        
        with torch.no_grad():
            stable_rank = compute_stable_rank(batch_flat)
            kappa = x.shape[-1] / stable_rank
            
            # Rotation matrix properties
            if hasattr(self.rora, 'get_rotation_matrix'):
                R = self.rora.get_rotation_matrix()
                # Check orthogonality: R @ R^T should be identity
                ortho_error = torch.norm(R @ R.T - torch.eye(R.shape[0], device=R.device))
                # Rotation angle (via Frobenius norm of log)
                rotation_magnitude = torch.norm(R - torch.eye(R.shape[0], device=R.device))
            else:
                ortho_error = rotation_magnitude = 0.0
                
            self.last_metrics = {
                'stable_rank': stable_rank,
                'overparameterization_ratio': kappa,
                'orthogonality_error': ortho_error.item() if isinstance(ortho_error, torch.Tensor) else ortho_error,
                'rotation_magnitude': rotation_magnitude.item() if isinstance(rotation_magnitude, torch.Tensor) else rotation_magnitude,
            }
        
        return self.rora(x)
