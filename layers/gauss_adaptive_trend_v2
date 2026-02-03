"""
Adaptive Gaussian Trend Extraction V2.

Key improvements over V1:
1. Centered (non-causal) symmetric Gaussian kernels - eliminates phase delay
2. Sequence-length-scaled sigma bank - proper coverage across different seq_lens
3. Enhanced conditioning features - z-score, log-variance, AND local slope
4. Kernel caching - avoids rebuilding kernels every forward pass
5. Numerical stability guards - prevents NaN on edge cases
6. Returns both trend and residual for convenience

A conditioning network selects mixture weights over a bank of Gaussian kernels
based on local signal statistics. The kernels adapt per-timestep based on
local signal characteristics.

Author: [Your Name]
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, List, Dict, Optional, Union
import math


# =============================================================================
# Helper Functions
# =============================================================================

def _build_centered_gaussian_kernel(
    sigma: float,
    truncate: float,
    seq_len: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build a centered (symmetric) Gaussian kernel with size clamping.

    k[n] ∝ exp(-(n²)/(2σ²)), n = -R..R, R = min(ceil(truncate * σ), (seq_len-1)//2)
    Normalized to sum=1.

    Args:
        sigma: Standard deviation of the Gaussian
        truncate: Truncation factor (kernel extends to truncate * sigma)
        seq_len: Input sequence length (for clamping kernel size)
        dtype: Target dtype
        device: Target device

    Returns:
        Kernel tensor of shape [2R+1] where 2R+1 <= seq_len
    """
    # Desired radius
    R_desired = max(1, int(truncate * sigma + 0.5))

    # Clamp: centered kernel length 2R+1 <= seq_len
    max_R = max(1, (seq_len - 1) // 2)
    R = min(R_desired, max_R)

    # Create symmetric kernel: -R to +R
    n = torch.arange(-R, R + 1, device=device, dtype=dtype)

    # Gaussian with numerical stability
    sigma_safe = max(float(sigma), 1e-6)
    k = torch.exp(-0.5 * (n / sigma_safe) ** 2)

    # Normalize to sum=1
    k = k / (k.sum() + 1e-12)

    return k


def _depthwise_conv1d_centered(
    x_btC: torch.Tensor,
    kernel_1d: torch.Tensor,
) -> torch.Tensor:
    """
    Depthwise centered 1D convolution with appropriate padding.

    Uses reflection padding where possible (better edge behavior),
    falls back to replicate padding for very large kernels.

    Args:
        x_btC: Input tensor [B, T, C]
        kernel_1d: 1D kernel [K] (same kernel applied to all channels)

    Returns:
        Filtered output [B, T, C] (same shape as input)
    """
    B, T, C = x_btC.shape
    x_bCt = x_btC.transpose(1, 2)  # [B, C, T]

    # Prepare kernel for depthwise conv
    k = kernel_1d.to(x_bCt.dtype).to(x_bCt.device)  # [K]
    K = k.numel()
    k = k.view(1, 1, K).expand(C, 1, K)  # [C, 1, K] for depthwise

    # Padding: symmetric for centered convolution
    pad = K // 2

    # Use reflection padding if possible (better than replicate at edges)
    # Reflection requires pad < T
    if pad < T:
        x_pad = F.pad(x_bCt, (pad, pad), mode='reflect')
    else:
        x_pad = F.pad(x_bCt, (pad, pad), mode='replicate')

    # Depthwise convolution
    y = F.conv1d(x_pad, k, bias=None, stride=1, padding=0, groups=C)

    return y.transpose(1, 2)  # [B, T, C]


def _compute_local_statistics(
    x: torch.Tensor,
    window: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute local mean, variance, and slope using centered windows.

    For positions near edges, uses available samples via padding.

    Args:
        x: Input tensor [B, T, C]
        window: Window size for statistics computation
        eps: Small constant for numerical stability

    Returns:
        Tuple of (mean, variance, slope), each [B, T, C]
    """
    B, T, C = x.shape
    device, dtype = x.device, x.dtype

    # Clamp window to sequence length (must be at least 2 for variance)
    win = max(2, min(window, T))

    # For centered windows of size `win`, we need asymmetric padding:
    # - left_pad: positions before index 0
    # - right_pad: positions after index T-1
    # This ensures unfold produces exactly T windows
    left_pad = (win - 1) // 2
    right_pad = win // 2

    # Transpose for padding along time dimension: [B, C, T]
    x_bCt = x.transpose(1, 2)

    # Apply padding (reflection if possible, replicate otherwise)
    if left_pad < T and right_pad < T:
        x_pad = F.pad(x_bCt, (left_pad, right_pad), mode='reflect')
    else:
        x_pad = F.pad(x_bCt, (left_pad, right_pad), mode='replicate')

    # Back to [B, T_padded, C]
    x_pad = x_pad.transpose(1, 2)  # [B, T + left_pad + right_pad, C]

    # Unfold to get windows: [B, T, C, win]
    # After padding, length is T + (win-1), unfold gives T windows
    x_windows = x_pad.unfold(dimension=1, size=win, step=1)  # [B, T, C, win]

    # Verify shape (this should now always be correct)
    assert x_windows.shape[1] == T, f"Window count mismatch: got {x_windows.shape[1]}, expected {T}"

    # Compute statistics over window dimension
    mean = x_windows.mean(dim=-1)  # [B, T, C]
    var = x_windows.var(dim=-1, unbiased=False).clamp_min(0.0)  # [B, T, C]

    # Compute local slope using linear regression over window
    # slope = cov(t, x) / var(t) where t = 0, 1, ..., win-1
    t = torch.arange(win, device=device, dtype=dtype)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    # Centered time indices
    t_centered = t - t_mean  # [win]

    # x_centered: [B, T, C, win]
    x_centered = x_windows - x_windows.mean(dim=-1, keepdim=True)

    # Covariance and slope
    cov = (x_centered * t_centered).sum(dim=-1)  # [B, T, C]
    slope = cov / (t_var + eps)  # [B, T, C]

    return mean, var, slope


class KernelCache:
    """
    Simple cache for precomputed kernels.

    Avoids rebuilding Gaussian kernels every forward pass when
    sequence length remains constant (common during training/inference).
    """

    def __init__(self, max_size: int = 32):
        """
        Args:
            max_size: Maximum number of cached kernel sets
        """
        self.max_size = max_size
        self._cache: Dict[Tuple, List[torch.Tensor]] = {}
        self._access_order: List[Tuple] = []

    def get_key(
        self,
        seq_len: int,
        sigmas: Tuple[float, ...],
        truncate: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple:
        """Generate cache key from parameters."""
        return (seq_len, sigmas, truncate, str(dtype), str(device))

    def get(self, key: Tuple) -> Optional[List[torch.Tensor]]:
        """Retrieve kernels from cache if available."""
        if key in self._cache:
            # Move to end of access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: Tuple, kernels: List[torch.Tensor]) -> None:
        """Store kernels in cache."""
        if key in self._cache:
            return

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = kernels
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


# =============================================================================
# Main Module
# =============================================================================

class AdaptiveGaussianTrendV2(nn.Module):
    """
    Adaptive Gaussian Trend Extraction V2.

    Architecture:
    1. Predefine K centered symmetric Gaussian low-pass kernels (scaled σ values)
    2. Apply K depthwise convolutions to input → Y[..., k]
    3. Compute local features (z-score, log-var, slope) → MLP → softmax over K
    4. Trend = Σ_k w_k * Y_k (per timestep, per channel)
    5. Residual = x - trend

    The conditioning network learns HOW to adapt filter selection based on
    local signal characteristics. At inference, filter weights vary per-window
    based on input statistics.

    Args:
        base_sigmas: Base sigma values for kernel bank (will be scaled by seq_len)
        reference_seq_len: Reference sequence length for sigma scaling (default: 512)
        truncate: Kernel truncation factor (kernel extends to truncate * sigma)
        cond_hidden: Hidden dimension for conditioning MLP
        stat_window: Window size for computing local statistics
        softmax_temp: Temperature for softmax (lower = sharper selection)
        use_slope: Whether to include local slope in conditioning features
        return_residual: Whether to return residual along with trend
        cache_kernels: Whether to cache computed kernels
        eps: Small constant for numerical stability

    Input/Output:
        x: [B, T, C] → (trend: [B, T, C], residual: [B, T, C]) or just trend

    Example:
        >>> model = AdaptiveGaussianTrendV2(
        ...     base_sigmas=(2, 4, 8, 16, 32),
        ...     reference_seq_len=512,
        ... )
        >>> x = torch.randn(32, 96, 7)  # batch=32, seq_len=96, channels=7
        >>> trend, residual = model(x)
        >>> print(trend.shape, residual.shape)  # [32, 96, 7], [32, 96, 7]
    """

    def __init__(
        self,
        base_sigmas: Sequence[float] = (2.0, 4.0, 8.0, 16.0, 32.0),
        reference_seq_len: int = 512,
        truncate: float = 4.0,
        cond_hidden: int = 32,
        stat_window: int = 16,
        softmax_temp: float = 0.7,
        use_slope: bool = True,
        return_residual: bool = True,
        cache_kernels: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Store configuration
        self.base_sigmas = tuple(base_sigmas)
        self.reference_seq_len = reference_seq_len
        self.truncate = float(truncate)
        self.stat_window = int(stat_window)
        self.softmax_temp = float(softmax_temp)
        self.use_slope = bool(use_slope)
        self.return_residual = bool(return_residual)
        self.eps = float(eps)

        self.num_kernels = len(base_sigmas)

        # Conditioning MLP
        # Features: z-score (1) + log-var (1) + optional slope (1)
        in_feats = 3 if self.use_slope else 2

        self.cond = nn.Sequential(
            nn.Linear(in_feats, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, self.num_kernels),
        )

        # Initialize final layer with small weights for soft initial selection
        nn.init.xavier_uniform_(self.cond[-1].weight, gain=0.1)
        nn.init.zeros_(self.cond[-1].bias)

        # Kernel cache
        self._kernel_cache = KernelCache() if cache_kernels else None

        # Statistics storage for monitoring
        self._last_gate_weights: Optional[torch.Tensor] = None
        self._last_entropy: Optional[float] = None
        self._last_sigma_usage: Optional[List[float]] = None

    def _get_scaled_sigmas(self, seq_len: int) -> Tuple[float, ...]:
        """
        Scale sigma bank based on sequence length.

        Longer sequences can support larger sigmas (more smoothing options).
        Shorter sequences need smaller sigmas to maintain resolution.

        Args:
            seq_len: Current input sequence length

        Returns:
            Tuple of scaled sigma values
        """
        scale = seq_len / self.reference_seq_len
        return tuple(s * scale for s in self.base_sigmas)

    def _build_kernels(
        self,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Build all kernels for current sequence length.

        Uses caching to avoid recomputation when seq_len is constant.

        Args:
            seq_len: Input sequence length
            dtype: Target dtype
            device: Target device

        Returns:
            List of kernel tensors, one per sigma
        """
        sigmas = self._get_scaled_sigmas(seq_len)

        # Check cache
        if self._kernel_cache is not None:
            key = self._kernel_cache.get_key(seq_len, sigmas, self.truncate, dtype, device)
            cached = self._kernel_cache.get(key)
            if cached is not None:
                return cached

        # Build kernels
        kernels = [
            _build_centered_gaussian_kernel(
                sigma=s,
                truncate=self.truncate,
                seq_len=seq_len,
                dtype=dtype,
                device=device,
            )
            for s in sigmas
        ]

        # Store in cache
        if self._kernel_cache is not None:
            self._kernel_cache.put(key, kernels)

        return kernels

    def _compute_conditioning_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute conditioning features for adaptive filter selection.

        Features:
        - Z-score: scale-invariant measure of local deviation
        - Log-variance: indicates local volatility
        - Local slope (optional): indicates trend direction/strength

        Args:
            x: Input tensor [B, T, C]

        Returns:
            Features tensor [B, T, C, F] where F is number of features
        """
        B, T, C = x.shape

        # Compute local statistics
        stat_win = min(self.stat_window, T)
        mean, var, slope = _compute_local_statistics(x, stat_win, self.eps)

        # Z-score (scale-invariant deviation from local mean)
        std = (var + self.eps).sqrt()
        z_score = (x - mean) / std

        # Clamp z-score to prevent extreme values
        z_score = z_score.clamp(-10.0, 10.0)

        # Log-variance (indicates volatility level)
        log_var = (var + self.eps).log()

        # Normalize log_var to roughly [-1, 1] range for better MLP input
        # Using running statistics would be better, but this is simpler
        log_var = log_var / 10.0  # Empirical scaling

        # Assemble features
        if self.use_slope:
            # Normalize slope by std for scale invariance
            norm_slope = slope / (std + self.eps)
            norm_slope = norm_slope.clamp(-10.0, 10.0)
            features = torch.stack([z_score, log_var, norm_slope], dim=-1)
        else:
            features = torch.stack([z_score, log_var], dim=-1)

        return features  # [B, T, C, F]

    def forward(
        self,
        x: torch.Tensor,
        return_gate_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Extract adaptive trend from input signal.

        Args:
            x: Input tensor [B, T, C]
            return_gate_weights: If True, also return gate weights [B, T, C, K]

        Returns:
            If return_residual=False and return_gate_weights=False:
                trend: [B, T, C]
            If return_residual=True and return_gate_weights=False:
                (trend, residual): each [B, T, C]
            If return_gate_weights=True:
                Adds gate_weights [B, T, C, K] to the returned tuple
        """
        B, T, C = x.shape

        # Input validation
        if torch.isnan(x).any():
            raise ValueError("NaN detected in input tensor")

        # Build kernels (cached for efficiency)
        kernels = self._build_kernels(T, x.dtype, x.device)

        # Apply all kernels via depthwise convolution
        # Y: [B, T, C, K] where K is number of kernels
        Y = torch.stack([
            _depthwise_conv1d_centered(x, k) for k in kernels
        ], dim=-1)

        # Compute conditioning features
        features = self._compute_conditioning_features(x)  # [B, T, C, F]

        # Run conditioning MLP to get logits
        logits = self.cond(features)  # [B, T, C, K]

        # Softmax with temperature for filter selection
        gate_weights = F.softmax(logits / self.softmax_temp, dim=-1)  # [B, T, C, K]

        # Weighted combination of filtered signals
        trend = (Y * gate_weights).sum(dim=-1)  # [B, T, C]

        # Store statistics for monitoring
        self._store_gate_statistics(gate_weights)

        # Compute residual
        residual = x - trend

        # Return based on configuration
        if return_gate_weights:
            if self.return_residual:
                return trend, residual, gate_weights
            else:
                return trend, gate_weights
        else:
            if self.return_residual:
                return trend, residual
            else:
                return trend

    def _store_gate_statistics(self, gate_weights: torch.Tensor) -> None:
        """Store gate statistics for monitoring/analysis."""
        with torch.no_grad():
            w = gate_weights.detach()

            # Entropy of gate distribution (higher = more uniform selection)
            entropy = -(w * (w.clamp_min(1e-8)).log()).sum(dim=-1).mean()

            # Mean weight per sigma (which kernels are preferred)
            sigma_usage = [float(w[..., i].mean()) for i in range(self.num_kernels)]

            self._last_gate_weights = w
            self._last_entropy = float(entropy)
            self._last_sigma_usage = sigma_usage

    def get_gate_statistics(self) -> Dict[str, Union[float, List[float]]]:
        """
        Get statistics about gating behavior for analysis.

        Returns:
            Dictionary containing:
            - entropy: Average entropy of gate distribution
            - max_weight_mean: Mean of maximum weight per position
            - max_weight_std: Std of maximum weight per position
            - sigma_usage: List of mean weights per sigma
            - scaled_sigmas: Current scaled sigma values (if available)
        """
        if self._last_gate_weights is None:
            return {}

        w = self._last_gate_weights

        return {
            "entropy": self._last_entropy,
            "max_weight_mean": float(w.max(dim=-1).values.mean()),
            "max_weight_std": float(w.max(dim=-1).values.std()),
            "sigma_usage": self._last_sigma_usage,
            "base_sigmas": list(self.base_sigmas),
        }

    def clear_cache(self) -> None:
        """Clear the kernel cache."""
        if self._kernel_cache is not None:
            self._kernel_cache.clear()

    def extra_repr(self) -> str:
        return (
            f"base_sigmas={self.base_sigmas}, "
            f"reference_seq_len={self.reference_seq_len}, "
            f"truncate={self.truncate}, "
            f"stat_window={self.stat_window}, "
            f"softmax_temp={self.softmax_temp}, "
            f"use_slope={self.use_slope}"
        )


# =============================================================================
# Convenience Wrapper with RevIN
# =============================================================================

class AdaptiveGaussianTrendWithRevIN(nn.Module):
    """
    AdaptiveGaussianTrendV2 wrapped with Reversible Instance Normalization.

    RevIN helps handle distribution shift between training and inference,
    which is common in time series forecasting.

    Args:
        num_features: Number of input channels (C)
        affine: Whether RevIN has learnable affine parameters
        **trend_kwargs: Arguments passed to AdaptiveGaussianTrendV2
    """

    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        **trend_kwargs,
    ):
        super().__init__()

        self.num_features = num_features
        self.affine = affine

        # RevIN parameters
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

        # Trend extractor
        self.trend_extractor = AdaptiveGaussianTrendV2(**trend_kwargs)

        self.eps = 1e-5

    def forward(
        self,
        x: torch.Tensor,
        return_gate_weights: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input [B, T, C]
            return_gate_weights: Whether to return gate weights

        Returns:
            (trend, residual) or (trend, residual, gate_weights)
            All in original (denormalized) scale
        """
        # Instance normalization
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + self.eps
        x_norm = (x - mean) / std

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        # Extract trend
        result = self.trend_extractor(x_norm, return_gate_weights=return_gate_weights)

        # Denormalize outputs
        if return_gate_weights:
            if self.trend_extractor.return_residual:
                trend_norm, residual_norm, gate_weights = result
            else:
                trend_norm, gate_weights = result
                residual_norm = x_norm - trend_norm
        else:
            if self.trend_extractor.return_residual:
                trend_norm, residual_norm = result
            else:
                trend_norm = result
                residual_norm = x_norm - trend_norm

        # Reverse normalization
        if self.affine:
            trend_norm = (trend_norm - self.affine_bias) / self.affine_weight
            residual_norm = (residual_norm - self.affine_bias) / self.affine_weight

        trend = trend_norm * std + mean
        residual = residual_norm * std  # Residual has zero mean by construction

        if return_gate_weights:
            return trend, residual, gate_weights
        else:
            return trend, residual

