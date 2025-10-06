import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple

def _gauss_kernel1d(sigma: float, truncate: float = 4.0, dtype=torch.float32, device=None):
    radius = max(1, int(truncate * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / (k.sum() + 1e-12)
    return k  # [K]

def _depthwise_conv1d_centered(x_btC: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    # x: [B,T,C]  -> Conv1d expects [B,C,T]
    B, T, C = x_btC.shape
    x_bCt = x_btC.transpose(1, 2)  # [B,C,T]
    k = kernel_1d.to(x_bCt.dtype).to(x_bCt.device)  # [K]
    k = k.view(1, 1, -1).repeat(C, 1, 1)           # [C,1,K] depthwise
    pad = (k.shape[-1] - 1) // 2
    y = F.conv1d(x_bCt, k, bias=None, stride=1, padding=pad, groups=C)
    return y.transpose(1, 2)  # [B,T,C]

class AdaptiveGaussianTrend(nn.Module):
    """
    Trend(x): mixture of K centered Gaussians with data-driven time-varying weights.
    Conditioner: local variance via AvgPool + small MLP -> softmax over K per (B,T,C).
    Fast: K depthwise convs + per-time convex mixture.
    """
    def __init__(
        self,
        sigmas: Sequence[float] = (2.5, 4.0, 6.0, 9.0, 14.0),
        truncate: float = 4.0,
        cond_hidden: int = 32,
        pool: int = 16,
    ):
        super().__init__()
        self.sigmas = list(sigmas)
        self.truncate = truncate
        self.pool = pool
        # conditioner maps [local_mean, local_var, x] -> K logits (per channel, per time)
        in_ch = 3
        self.cond = nn.Sequential(
            nn.Linear(in_ch, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, len(self.sigmas))
        )

    @torch.no_grad()
    def _precompute_kernels(self, x: torch.Tensor):
        dtype, device = x.dtype, x.device
        kernels = [ _gauss_kernel1d(s, self.truncate, dtype=dtype, device=device) for s in self.sigmas ]
        return kernels  # list of [K_i]

    def _local_mean_var(self, x: torch.Tensor, k: int):
        # x [B,T,C] -> use AvgPool1d over time with "same" output length
        B, T, C = x.shape
        x_bCt = x.transpose(1, 2)  # [B,C,T]
    
        # Asymmetric "same" padding for any k (odd or even)
        padL = (k - 1) // 2
        padR = k - 1 - padL  # ensures padL + padR = k - 1
        x_pad = F.pad(x_bCt, (padL, padR), mode='reflect')
    
        mean  = F.avg_pool1d(x_pad, kernel_size=k, stride=1)          # [B,C,T]
        mean2 = F.avg_pool1d(x_pad * x_pad, kernel_size=k, stride=1)  # [B,C,T]
        var = (mean2 - mean * mean).clamp_min(0.0)
        return mean.transpose(1, 2), var.transpose(1, 2)  # [B,T,C]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: trend [B,T,C]
        """
        B,T,C = x.shape
        kernels = self._precompute_kernels(x)

        # K depthwise convs
        yKs = []
        for k in kernels:
            yKs.append(_depthwise_conv1d_centered(x, k))  # each [B,T,C]
        Y = torch.stack(yKs, dim=-1)  # [B,T,C,K]

        # Conditioner: local stats + raw x
        m, v = self._local_mean_var(x, k=self.pool)
        feats = torch.stack([m, v, x], dim=-1)  # [B,T,C,3]
        logits = self.cond(feats)               # [B,T,C,K]
        w = torch.softmax(logits, dim=-1)       # convex weights over K
        trend = (Y * w).sum(dim=-1)             # [B,T,C]
        return trend
