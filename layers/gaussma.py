# layers/gaussma.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- helpers ---

def sigma_from_alpha(alpha: float) -> float:
    # EMA alpha -> approx Gaussian sigma (variance-matched geometric kernel)
    a = float(alpha)
    a = max(min(a, 0.9999), 1e-6)
    return math.sqrt(1.0 - a) / a

def gaussian_sigma_from_period(P: int, mult: float = 0.75) -> float:
    return float(P) * float(mult)

def _gaussian_kernel_1d(sigma: torch.Tensor, truncate: float = 4.0) -> torch.Tensor:
    sigma = sigma.float().clamp_min(1e-3)
    R = int((truncate * sigma).round().clamp_min(1).item())
    x = torch.arange(-R, R + 1, device=sigma.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k  # [K], odd length

def _depthwise_conv1d_reflect(x_bct: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    # x_bct: [B, C, T]
    K = kernel_1d.numel()
    pad = (K // 2, K // 2)
    x_pad = F.pad(x_bct, pad=(pad[0], pad[1]), mode="reflect")
    w = kernel_1d.view(1, 1, K).repeat(x_bct.shape[1], 1, 1)  # [C,1,K], depthwise
    return F.conv1d(x_pad, w, stride=1, padding=0, groups=x_bct.shape[1])

# --- single-pass Gaussian moving average ---

class GaussianMA(nn.Module):
    """
    Centered Gaussian moving average for inputs [B, T, C].
    Forward returns the smoothed series (trend), like EMA.forward(x).
    """
    def __init__(self, sigma: float, learnable: bool = False, truncate: float = 4.0):
        super().__init__()
        self.truncate = float(truncate)
        s = torch.tensor(float(sigma), dtype=torch.float32)
        if learnable:
            self.sigma = nn.Parameter(s)
        else:
            self.register_buffer("sigma", s)

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]  (xPatch convention)
        k = _gaussian_kernel_1d(self.sigma.to(x_btc.device), truncate=self.truncate)
        y_bct = _depthwise_conv1d_reflect(x_btc.transpose(1, 2), k)
        return y_bct.transpose(1, 2)  # [B, T, C]

# --- two-pass Gaussian decomposition (daily→weekly etc.) ---

class GaussianMA2Pass(nn.Module):
    """
    Two-pass Gaussian decomposition returning (seasonal_total, trend_final).
    Pass 1: trend1 = G(sigma1)*x; seasonal1 = x - trend1
    Pass 2 on x_res = x - seasonal1: trend2 = G(sigma2)*x_res; seasonal2 = x_res - trend2
    Output: seasonal_total = seasonal1 + seasonal2, trend_final = trend2
    """
    def __init__(self,
                 sigma1: float | None = None,
                 sigma2: float | None = None,
                 P1: int | None = None, mult1: float | None = None,
                 P2: int | None = None, mult2: float | None = None,
                 learnable: bool = False, truncate: float = 4.0):
        super().__init__()
        if sigma1 is None:
            if P1 is None or mult1 is None:
                raise ValueError("Provide sigma1 or (P1, mult1).")
            sigma1 = gaussian_sigma_from_period(P1, mult1)
        if sigma2 is None:
            if P2 is None or mult2 is None:
                raise ValueError("Provide sigma2 or (P2, mult2).")
            sigma2 = gaussian_sigma_from_period(P2, mult2)

        self.ma1 = GaussianMA(sigma1, learnable=learnable, truncate=truncate)
        self.ma2 = GaussianMA(sigma2, learnable=learnable, truncate=truncate)

    def forward(self, x: torch.Tensor):
        trend1 = self.ma1(x)          # [B,T,C]
        seasonal1 = x - trend1
        x_res = x - seasonal1
        trend2 = self.ma2(x_res)
        seasonal2 = x_res - trend2
        return seasonal1 + seasonal2, trend2
