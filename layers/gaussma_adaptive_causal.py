"""
Adaptive Gaussian Trend Extraction (Causal).

A conditioning network selects mixture weights over a bank of Gaussian kernels
based on local signal statistics. The kernels are causal (no future leakage).

Handles edge cases where kernel size would exceed input sequence length
by dynamically clamping the kernel size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, List


def _causal_gauss_kernel1d_safe(
    sigma: float,
    truncate: float,
    seq_len: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Build a causal (one-sided) half-Gaussian kernel with size clamping.
    
    k[n] ∝ exp(-(n²)/(2σ²)), n = 0..R, R = min(ceil(truncate * σ), seq_len-1)
    Normalized to sum=1.
    
    Args:
        sigma: Standard deviation
        truncate: Truncation factor
        seq_len: Input sequence length (for clamping)
        dtype: Target dtype
        device: Target device
        
    Returns:
        Kernel [K] where K <= seq_len
    """
    # Desired radius
    R_desired = max(1, int(truncate * sigma + 0.5))
    
    # Clamp: causal kernel length K = R + 1 <= seq_len
    max_R = max(1, seq_len - 1)
    R = min(R_desired, max_R)
    
    n = torch.arange(0, R + 1, device=device, dtype=dtype)  # 0..R (causal lags only)
    k = torch.exp(-0.5 * (n / max(float(sigma), 1e-6)) ** 2)
    k = k / (k.sum() + 1e-12)
    return k


def _depthwise_conv1d_causal_safe(
    x_btC: torch.Tensor,
    kernel_1d: torch.Tensor,
) -> torch.Tensor:
    """
    Depthwise causal 1D convolution (no future leakage) with safe padding.
    
    Args:
        x_btC: Input [B, T, C]
        kernel_1d: 1D kernel [K] (same for all channels)
        
    Returns:
        Filtered output [B, T, C]
    """
    B, T, C = x_btC.shape
    x_bCt = x_btC.transpose(1, 2)  # [B, C, T]
    
    k = kernel_1d.to(x_bCt.dtype).to(x_bCt.device)  # [K]
    K = k.numel()
    k = k.view(1, 1, -1).repeat(C, 1, 1)  # [C, 1, K]
    
    # Causal: left pad only
    padL = K - 1
    x_pad = F.pad(x_bCt, (padL, 0), mode='replicate')
    
    y = F.conv1d(x_pad, k, bias=None, stride=1, padding=0, groups=C)
    return y.transpose(1, 2)  # [B, T, C]


def _causal_mean_var(
    x: torch.Tensor,
    win: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Causal running mean/variance via depthwise conv with an all-ones kernel.
    
    Args:
        x: Input [B, T, C]
        win: Window size (>=1)
        
    Returns:
        (mean, var) each [B, T, C], causal (uses only t' <= t)
    """
    B, T, C = x.shape
    x_bCt = x.transpose(1, 2)  # [B, C, T]
    
    # Clamp window to sequence length
    win = min(win, T)
    
    ones = torch.ones(1, 1, win, device=x.device, dtype=x.dtype)
    ones = ones.repeat(C, 1, 1)  # [C, 1, win] depthwise
    padL = win - 1

    # Sum over causal window
    sum_x = F.conv1d(F.pad(x_bCt, (padL, 0), mode='replicate'), ones, groups=C)
    sum_x2 = F.conv1d(F.pad(x_bCt ** 2, (padL, 0), mode='replicate'), ones, groups=C)

    # Effective window length at each t (1..win) to handle warm-up without bias
    eff = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).clamp_max(win)
    eff = eff.view(1, 1, T).expand(B, C, T)  # [B, C, T]

    mean = (sum_x / (eff + 1e-12)).transpose(1, 2)   # [B, T, C]
    mean2 = (sum_x2 / (eff + 1e-12)).transpose(1, 2)  # [B, T, C]
    var = (mean2 - mean ** 2).clamp_min(0.0)          # [B, T, C]
    return mean, var


class AdaptiveGaussianTrendCausal(nn.Module):
    """
    Causal Adaptive Gaussian Trend (AGF, causal).
    
    Architecture:
    1. Predefine K causal half-Gaussian low-pass kernels (different σ)
    2. Apply K depthwise causal convs to x → Y[..., k]
    3. Compute causal local features (z-score, log-var) → tiny MLP → softmax over K
    4. Trend = Σ_k w_k * Y_k (per time, per channel)
    
    The conditioning network learns HOW to adapt, not what params to use.
    At inference, filter weights vary per-window based on input statistics.
    
    Args:
        sigmas: Tuple of sigma values for the kernel bank
        truncate: Kernel truncation factor
        cond_hidden: Hidden size for conditioning MLP
        stat_window: Window size for computing local statistics
        add_x_feature: Include raw x in conditioner features
        softmax_temp: Temperature for softmax (lower = sharper selection)
        use_zscore: Use z-score instead of raw x for conditioning
        entropy_reg: Entropy regularization weight (optional)
    
    Input/Output:
        x: [B, T, C] → trend: [B, T, C]
    """
    def __init__(
        self,
        sigmas: Sequence[float] = (2.5, 4.0, 6.0, 9.0, 14.0),
        truncate: float = 4.0,
        cond_hidden: int = 32,
        stat_window: int = 16,
        add_x_feature: bool = False,
        softmax_temp: float = 0.7,
        use_zscore: bool = True,
        entropy_reg: float = 0.0,
    ):
        super().__init__()
        self.sigmas = list(sigmas)
        self.truncate = float(truncate)
        self.stat_window = int(stat_window)
        self.add_x_feature = bool(add_x_feature)
        self.softmax_temp = float(softmax_temp)
        self.use_zscore = bool(use_zscore)
        self.entropy_reg = float(entropy_reg)

        # Features: z-score and log-var (2). Optionally raw x (+1).
        in_feats = 2 + (1 if self.add_x_feature else 0)
        
        self.cond = nn.Sequential(
            nn.Linear(in_feats, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, len(self.sigmas)),
        )
        
        # Store last entropy for monitoring
        self._last_entropy = None
        self._last_gate_weights = None

    def _build_kernels(
        self,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Build all kernels with size clamped to sequence length.
        
        Args:
            seq_len: Input sequence length
            dtype: Target dtype
            device: Target device
            
        Returns:
            List of kernels, one per sigma
        """
        return [
            _causal_gauss_kernel1d_safe(
                sigma=s,
                truncate=self.truncate,
                seq_len=seq_len,
                dtype=dtype,
                device=device,
            )
            for s in self.sigmas
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract adaptive trend.
        
        Args:
            x: Input [B, T, C]
            
        Returns:
            Trend [B, T, C] (causal; uses only past/present)
        """
        B, T, C = x.shape
        
        # Build kernels dynamically based on sequence length
        kernels = self._build_kernels(T, x.dtype, x.device)

        # K causal depthwise LP responses
        Y = torch.stack([
            _depthwise_conv1d_causal_safe(x, k) for k in kernels
        ], dim=-1)  # [B, T, C, K]

        # Causal local stats
        stat_win = min(self.stat_window, T)
        m, v = _causal_mean_var(x, win=stat_win)  # [B, T, C]
        
        # Build conditioning features
        z = (x - m) / (v.add(1e-6).sqrt())  # z-score (scale-invariant)
        logv = (v + 1e-6).log()             # log-variance

        feats = [z if self.use_zscore else x, logv]  # 2 features
        if self.add_x_feature:
            feats.append(x)  # optional 3rd feature
        feats = torch.stack(feats, dim=-1)  # [B, T, C, F]

        # Conditioner (pointwise MLP over B, T, C)
        logits = self.cond(feats)  # [B, T, C, K]
        w = torch.softmax(logits / self.softmax_temp, dim=-1)

        # Weighted combination of filtered signals
        trend = (Y * w).sum(dim=-1)  # [B, T, C]

        # Store for monitoring/regularization
        self._last_gate_weights = w.detach()
        self._last_entropy = -(w * (w.clamp_min(1e-8)).log()).sum(dim=-1).mean().detach()

        return trend
    
    def get_gate_stats(self) -> dict:
        """
        Get statistics about gating behavior (for analysis).
        
        Returns:
            Dict with entropy, max_weight, etc.
        """
        if self._last_gate_weights is None:
            return {}
        
        w = self._last_gate_weights
        return {
            "entropy": float(self._last_entropy),
            "max_weight_mean": float(w.max(dim=-1).values.mean()),
            "max_weight_std": float(w.max(dim=-1).values.std()),
            "weights_per_sigma": [float(w[..., i].mean()) for i in range(len(self.sigmas))],
        }
    
    def extra_repr(self) -> str:
        return (f"sigmas={self.sigmas}, truncate={self.truncate}, "
                f"stat_window={self.stat_window}, softmax_temp={self.softmax_temp}")


class AdaptiveGaussianTrendCentered(nn.Module):
    """
    Centered (non-causal) version of AdaptiveGaussianTrend.
    
    Uses full symmetric Gaussian kernels (looks at both past and future).
    Suitable for offline analysis where causality is not required.
    
    Same interface as AdaptiveGaussianTrendCausal.
    """
    def __init__(
        self,
        sigmas: Sequence[float] = (2.5, 4.0, 6.0, 9.0, 14.0),
        truncate: float = 4.0,
        cond_hidden: int = 32,
        stat_window: int = 16,
        softmax_temp: float = 0.7,
    ):
        super().__init__()
        self.sigmas = list(sigmas)
        self.truncate = float(truncate)
        self.stat_window = int(stat_window)
        self.softmax_temp = float(softmax_temp)
        
        self.cond = nn.Sequential(
            nn.Linear(2, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, len(self.sigmas)),
        )
    
    def _build_centered_kernel(
        self,
        sigma: float,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build centered Gaussian kernel with clamping."""
        R_desired = max(1, int(self.truncate * sigma + 0.5))
        max_R = max(1, (seq_len - 1) // 2)
        R = min(R_desired, max_R)
        
        x = torch.arange(-R, R + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
        k = k / (k.sum() + 1e-12)
        return k
    
    def _centered_conv(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Apply centered convolution with appropriate padding."""
        B, T, C = x.shape
        x_bCt = x.transpose(1, 2)
        
        K = k.numel()
        pad = K // 2
        
        if pad < T:
            x_pad = F.pad(x_bCt, (pad, pad), mode='reflect')
        else:
            x_pad = F.pad(x_bCt, (pad, pad), mode='replicate')
        
        kw = k.view(1, 1, K).repeat(C, 1, 1)
        y = F.conv1d(x_pad, kw, groups=C)
        return y.transpose(1, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Build and apply kernels
        Y = torch.stack([
            self._centered_conv(x, self._build_centered_kernel(s, T, x.dtype, x.device))
            for s in self.sigmas
        ], dim=-1)  # [B, T, C, K]
        
        # Local stats (using centered window here would be ideal, but causal is fine)
        m, v = _causal_mean_var(x, min(self.stat_window, T))
        z = (x - m) / (v.add(1e-6).sqrt())
        logv = (v + 1e-6).log()
        feats = torch.stack([z, logv], dim=-1)
        
        # Gating
        logits = self.cond(feats)
        w = torch.softmax(logits / self.softmax_temp, dim=-1)
        
        return (Y * w).sum(dim=-1)
