import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple

class EMA_Trend(nn.Module):
    """Fast EMA trend with scalar alpha (can be made learnable if desired)."""
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,C], causal EMA computed exactly via cumulative formula
        B,T,C = x.shape
        device, dtype = x.device, x.dtype
        t_idx = torch.arange(T, dtype=torch.float64, device=device)
        w = (1.0 - self.alpha) ** torch.flip(t_idx, dims=(0,))     # [T]
        w = w.to(dtype)
        w_div = w.clone()
        w[1:] *= self.alpha
        w = w.view(1, T, 1)
        w_div = w_div.view(1, T, 1)
        num = torch.cumsum(x * w, dim=1)
        ema = (num / (w_div + 1e-12)).to(dtype)
        return ema

class DoG_Seasonal(nn.Module):
    """Seasonal(x) = (Gσ1 - Gσ2)*x with centered Gaussian LPs (depthwise)."""
    def __init__(self, sigma1: float = 4.2, sigma2: float = 96.0, truncate: float = 4.0):
        super().__init__()
        assert sigma2 > sigma1, "Require sigma2 > sigma1"
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.truncate = float(truncate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k1 = _gauss_kernel1d(self.sigma1, self.truncate, x.dtype, x.device)
        k2 = _gauss_kernel1d(self.sigma2, self.truncate, x.dtype, x.device)
        y1 = _depthwise_conv1d_centered(x, k1)  # LP small sigma (passes daily & above)
        y2 = _depthwise_conv1d_centered(x, k2)  # LP large sigma (trendier)
        seasonal = y1 - y2
        return seasonal

class HybridEMA_DoG(nn.Module):
    """Returns (trend, seasonal) so you can wire directly into xPatch streams."""
    def __init__(self, alpha: float = 0.3, sigma1: float = 4.2, sigma2: float = 96.0, truncate: float = 4.0):
        super().__init__()
        self.ema = EMA_Trend(alpha)
        self.dog = DoG_Seasonal(sigma1, sigma2, truncate)

    def forward(self, x: torch.Tensor):
        trend = self.ema(x)
        seasonal = self.dog(x)   # compute on raw x (as we analyzed)
        return trend, seasonal
