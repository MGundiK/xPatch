"""
Adaptive Gaussian Trend Extraction V2 (Centered).

Improvements over V1 (causal):
1. Centered (non-causal) symmetric Gaussian kernels — eliminates phase delay
2. Sequence-length-scaled sigma bank — proper coverage across different seq_lens
3. Enhanced conditioning: z-score + log-variance + local slope
4. Kernel caching — avoids rebuilding every forward pass
5. Numerical stability guards — prevents NaN on edge cases (illness, solar)
6. Reflection padding where possible — better edge behavior

Interface:
    forward(x: [B, T, C]) → trend: [B, T, C]

    Designed to drop into DECOMP.ma — decomp.py computes seasonal = x - trend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, List, Dict, Optional
import math


# =============================================================================
# Helpers
# =============================================================================

def _build_centered_gaussian_kernel(
    sigma: float,
    truncate: float,
    seq_len: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Centered symmetric Gaussian kernel with size clamping.

    k[n] ∝ exp(−n² / 2σ²),  n = −R … +R
    R = min(⌈truncate·σ⌉, (seq_len−1)//2)
    Normalized so Σk = 1.
    """
    R_desired = max(1, int(truncate * sigma + 0.5))
    max_R = max(1, (seq_len - 1) // 2)
    R = min(R_desired, max_R)

    n = torch.arange(-R, R + 1, device=device, dtype=dtype)
    sigma_safe = max(float(sigma), 1e-6)
    k = torch.exp(-0.5 * (n / sigma_safe) ** 2)
    k = k / (k.sum() + 1e-12)
    return k


def _depthwise_conv1d_centered(
    x_btC: torch.Tensor,
    kernel_1d: torch.Tensor,
) -> torch.Tensor:
    """
    Depthwise centered 1-D convolution.

    Uses reflection padding when pad < T, else replicate.
    """
    B, T, C = x_btC.shape
    x_bCt = x_btC.transpose(1, 2)                     # [B, C, T]

    k = kernel_1d.to(x_bCt.dtype).to(x_bCt.device)    # [K]
    K = k.numel()
    k = k.view(1, 1, K).expand(C, 1, K)               # [C, 1, K]

    pad = K // 2
    if pad < T:
        x_pad = F.pad(x_bCt, (pad, pad), mode='reflect')
    else:
        x_pad = F.pad(x_bCt, (pad, pad), mode='replicate')

    y = F.conv1d(x_pad, k, bias=None, stride=1, padding=0, groups=C)
    return y.transpose(1, 2)                            # [B, T, C]


def _compute_local_statistics(
    x: torch.Tensor,
    window: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Centered local mean, variance, and slope via sliding window.

    Returns (mean, var, slope) each [B, T, C].
    """
    B, T, C = x.shape
    device, dtype = x.device, x.dtype

    win = max(2, min(window, T))

    # Asymmetric padding so unfold yields exactly T windows
    left_pad = (win - 1) // 2
    right_pad = win // 2

    x_bCt = x.transpose(1, 2)
    if left_pad < T and right_pad < T:
        x_pad = F.pad(x_bCt, (left_pad, right_pad), mode='reflect')
    else:
        x_pad = F.pad(x_bCt, (left_pad, right_pad), mode='replicate')
    x_pad = x_pad.transpose(1, 2)                      # [B, T+win-1, C]

    x_windows = x_pad.unfold(dimension=1, size=win, step=1)  # [B, T, C, win]

    mean = x_windows.mean(dim=-1)                       # [B, T, C]
    var  = x_windows.var(dim=-1, unbiased=False).clamp_min(0.0)

    # Local slope via OLS:  slope = cov(t, x) / var(t)
    t = torch.arange(win, device=device, dtype=dtype)
    t_mean = t.mean()
    t_var  = ((t - t_mean) ** 2).sum()
    t_centered = t - t_mean

    x_centered = x_windows - x_windows.mean(dim=-1, keepdim=True)
    cov   = (x_centered * t_centered).sum(dim=-1)
    slope = cov / (t_var + eps)

    return mean, var, slope


class _KernelCache:
    """LRU cache for precomputed kernel lists."""

    def __init__(self, max_size: int = 16):
        self.max_size = max_size
        self._cache: Dict[Tuple, List[torch.Tensor]] = {}
        self._order: List[Tuple] = []

    def _make_key(self, seq_len, sigmas, truncate, dtype, device):
        return (seq_len, sigmas, truncate, str(dtype), str(device))

    def get(self, seq_len, sigmas, truncate, dtype, device):
        key = self._make_key(seq_len, sigmas, truncate, dtype, device)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        return None

    def put(self, seq_len, sigmas, truncate, dtype, device, kernels):
        key = self._make_key(seq_len, sigmas, truncate, dtype, device)
        if key in self._cache:
            return
        if len(self._cache) >= self.max_size:
            del self._cache[self._order.pop(0)]
        self._cache[key] = kernels
        self._order.append(key)

    def clear(self):
        self._cache.clear()
        self._order.clear()


# =============================================================================
# Main module
# =============================================================================

class AdaptiveGaussianTrendV2(nn.Module):
    """
    Adaptive Gaussian Trend (centered, V2).

    1.  Build K centered Gaussian LP kernels (σ scaled by seq_len).
    2.  Depthwise-convolve x with each kernel → Y[…, k].
    3.  Compute local features (z-score, log-var, slope) → MLP → softmax → w.
    4.  trend = Σ_k  w_k · Y_k   (per timestep, per channel).

    forward(x) → trend   (single tensor, compatible with decomp.py)

    Args
    ----
    base_sigmas        : sigma values at reference_seq_len
    reference_seq_len  : sequence length at which base_sigmas are exact
    truncate           : kernel truncation factor
    cond_hidden        : hidden dim for conditioning MLP
    stat_window        : window for local statistics
    softmax_temp       : softmax temperature (lower = sharper)
    use_slope          : include local slope as conditioning feature
    cache_kernels      : cache kernels across forward calls
    eps                : numerical stability constant
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
        cache_kernels: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.base_sigmas       = tuple(base_sigmas)
        self.reference_seq_len = int(reference_seq_len)
        self.truncate          = float(truncate)
        self.stat_window       = int(stat_window)
        self.softmax_temp      = float(softmax_temp)
        self.use_slope         = bool(use_slope)
        self.eps               = float(eps)
        self.num_kernels       = len(self.base_sigmas)

        # ---- Conditioning MLP ----
        in_feats = 3 if self.use_slope else 2
        self.cond = nn.Sequential(
            nn.Linear(in_feats, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, self.num_kernels),
        )
        # small init → near-uniform selection at start of training
        nn.init.xavier_uniform_(self.cond[-1].weight, gain=0.1)
        nn.init.zeros_(self.cond[-1].bias)

        # ---- Kernel cache ----
        self._kcache = _KernelCache() if cache_kernels else None

        # ---- Monitoring (not saved in state_dict) ----
        self._last_gate_weights: Optional[torch.Tensor] = None
        self._last_entropy:      Optional[float]        = None

    # ------------------------------------------------------------------ helpers

    def _scaled_sigmas(self, seq_len: int) -> Tuple[float, ...]:
        s = seq_len / self.reference_seq_len
        return tuple(round(b * s, 4) for b in self.base_sigmas)

    def _get_kernels(self, seq_len, dtype, device):
        sigmas = self._scaled_sigmas(seq_len)
        if self._kcache is not None:
            hit = self._kcache.get(seq_len, sigmas, self.truncate, dtype, device)
            if hit is not None:
                return hit
        kernels = [
            _build_centered_gaussian_kernel(s, self.truncate, seq_len, dtype, device)
            for s in sigmas
        ]
        if self._kcache is not None:
            self._kcache.put(seq_len, sigmas, self.truncate, dtype, device, kernels)
        return kernels

    def _conditioning_features(self, x: torch.Tensor) -> torch.Tensor:
        """→ [B, T, C, F]"""
        B, T, C = x.shape
        stat_win = min(self.stat_window, T)
        mean, var, slope = _compute_local_statistics(x, stat_win, self.eps)

        std     = (var + self.eps).sqrt()
        z_score = ((x - mean) / std).clamp(-10.0, 10.0)
        log_var = (var + self.eps).log() / 10.0          # rough scaling

        if self.use_slope:
            norm_slope = (slope / (std + self.eps)).clamp(-10.0, 10.0)
            return torch.stack([z_score, log_var, norm_slope], dim=-1)
        else:
            return torch.stack([z_score, log_var], dim=-1)

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, C]  →  trend : [B, T, C]
        """
        B, T, C = x.shape

        # 1. filtered bank
        kernels = self._get_kernels(T, x.dtype, x.device)
        Y = torch.stack(
            [_depthwise_conv1d_centered(x, k) for k in kernels], dim=-1
        )                                                   # [B, T, C, K]

        # 2. conditioning
        feats  = self._conditioning_features(x)             # [B, T, C, F]
        logits = self.cond(feats)                           # [B, T, C, K]
        w      = F.softmax(logits / self.softmax_temp, dim=-1)

        # 3. mix
        trend = (Y * w).sum(dim=-1)                         # [B, T, C]

        # monitoring (detached, not part of graph)
        with torch.no_grad():
            self._last_gate_weights = w.detach()
            self._last_entropy = float(
                -(w * w.clamp_min(1e-8).log()).sum(-1).mean()
            )

        return trend

    # ------------------------------------------------------------------ extras

    def get_gate_statistics(self) -> dict:
        if self._last_gate_weights is None:
            return {}
        w = self._last_gate_weights
        return {
            "entropy":         self._last_entropy,
            "max_weight_mean": float(w.max(-1).values.mean()),
            "max_weight_std":  float(w.max(-1).values.std()),
            "sigma_usage":     [float(w[..., i].mean()) for i in range(self.num_kernels)],
        }

    def clear_cache(self):
        if self._kcache is not None:
            self._kcache.clear()

    def extra_repr(self) -> str:
        return (
            f"base_sigmas={self.base_sigmas}, "
            f"ref_seq_len={self.reference_seq_len}, "
            f"truncate={self.truncate}, "
            f"stat_window={self.stat_window}, "
            f"softmax_temp={self.softmax_temp}, "
            f"use_slope={self.use_slope}"
        )
