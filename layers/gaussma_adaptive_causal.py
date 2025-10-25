import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple

def _causal_gauss_kernel1d(
    sigma: float, truncate: float = 4.0,
    dtype: torch.dtype = torch.float32, device: torch.device = None
) -> torch.Tensor:
    """
    Build a causal (one-sided) half-Gaussian kernel:
      k[n] ∝ exp(-(n^2)/(2σ^2)),  n = 0..R,  R = ceil(truncate * σ)
    Normalized to sum=1. Returns shape [K].
    """
    R = max(1, int(truncate * sigma))
    n = torch.arange(0, R + 1, device=device, dtype=dtype)  # 0..R (causal lags only)
    k = torch.exp(-0.5 * (n / float(sigma))**2)
    k = k / (k.sum() + 1e-12)
    return k


def _depthwise_conv1d_causal(x_btC: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    """
    Depthwise causal 1D conv (no future leakage).
      x: [B, T, C]
      kernel_1d: [K] (same for all channels)
    Returns y: [B, T, C]
    """
    B, T, C = x_btC.shape
    x_bCt = x_btC.transpose(1, 2)  # [B, C, T]
    k = kernel_1d.to(x_bCt.dtype).to(x_bCt.device)      # [K]
    k = k.view(1, 1, -1).repeat(C, 1, 1)                # [C, 1, K]
    padL = k.shape[-1] - 1
    # causal: left pad only
    y = F.conv1d(F.pad(x_bCt, (padL, 0), mode='replicate'),
                 k, bias=None, stride=1, padding=0, groups=C)
    return y.transpose(1, 2)  # [B, T, C]


def _causal_mean_var(x: torch.Tensor, win: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Causal running mean/variance via depthwise conv with an all-ones kernel.
      x:  [B, T, C]
      win: window size (>=1)
    Returns (mean, var) each [B, T, C], causal (uses only t' <= t).
    """
    B, T, C = x.shape
    x_bCt = x.transpose(1, 2)  # [B, C, T]
    ones = torch.ones(1, 1, win, device=x.device, dtype=x.dtype)
    ones = ones.repeat(C, 1, 1)  # [C,1,win] depthwise
    padL = win - 1

    # Sum over causal window
    sum_x  = F.conv1d(F.pad(x_bCt,  (padL, 0), mode='replicate'), ones, groups=C)
    sum_x2 = F.conv1d(F.pad(x_bCt**2, (padL, 0), mode='replicate'), ones, groups=C)

    # Effective window length at each t (1..win) to handle warm-up without bias
    # Build a ramp [1,2,...,win] then flat at 'win'
    eff = torch.arange(1, win + 1, device=x.device, dtype=x.dtype)
    eff = F.pad(eff.view(1,1,-1), (T-win, 0), mode='constant', value=win)  # [1,1,T]
    eff = eff.expand(B, 1, T).expand(B, C, T)  # [B,C,T]

    mean = (sum_x / (eff + 1e-12)).transpose(1, 2)      # [B,T,C]
    mean2= (sum_x2/ (eff + 1e-12)).transpose(1, 2)      # [B,T,C]
    var  = (mean2 - mean**2).clamp_min(0.0)             # [B,T,C]
    return mean, var

'''
class AdaptiveGaussianTrendCausal(nn.Module):
    """
    Causal Adaptive Gaussian Trend (AGF, causal):
      - Predefine K causal half-Gaussian low-pass kernels (different σ).
      - Apply K depthwise causal convs to x  -> Y[..., k]
      - Compute causal local features (mean, var, x)  -> tiny MLP -> softmax over K
      - Trend = Σ_k w_k * Y_k   (per time, per channel)

    x, trend: [B, T, C]
    """
    def __init__(
        self,
        sigmas: Sequence[float] = (2.5, 4.0, 6.0, 9.0, 14.0),
        truncate: float = 4.0,
        cond_hidden: int = 32,
        stat_window: int = 16,     # causal window for mean/var
        add_x_feature: bool = True # include raw x in conditioner
    ):
        super().__init__()
        self.sigmas = list(sigmas)
        self.truncate = float(truncate)
        self.stat_window = int(stat_window)
        self.add_x_feature = bool(add_x_feature)

        in_feats = 2 + (1 if add_x_feature else 0)  # [mean, var, (optional x)]
        self.cond = nn.Sequential(
            nn.Linear(in_feats, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, len(self.sigmas))
        )

    @torch.no_grad()
    def _precompute_kernels(self, x: torch.Tensor):
        dtype, device = x.dtype, x.device
        return [
            _causal_gauss_kernel1d(s, self.truncate, dtype=dtype, device=device)
            for s in self.sigmas
        ]  # list of [K_i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            trend: [B, T, C] (causal; uses only past/present)
        """
        B, T, C = x.shape
        kernels = self._precompute_kernels(x)

        # K causal depthwise LP responses
        yKs = [ _depthwise_conv1d_causal(x, k) for k in kernels ]   # each [B,T,C]
        Y   = torch.stack(yKs, dim=-1)                              # [B,T,C,K]

        # Causal local stats (mean/var) over stat_window
        m, v = _causal_mean_var(x, win=self.stat_window)            # [B,T,C]
        feats = [m, v]
        if self.add_x_feature:
            feats.append(x)
        feats = torch.stack(feats, dim=-1)                          # [B,T,C,F]

        # Tiny MLP conditioner (pointwise over B,T,C)
        logits = self.cond(feats)                                   # [B,T,C,K]
        w = torch.softmax(logits, dim=-1)                           # convex weights over K

        trend = (Y * w).sum(dim=-1)                                 # [B,T,C]
        return trend
'''

class AdaptiveGaussianTrendCausal(nn.Module):
    """
    Causal Adaptive Gaussian Trend (AGF, causal):
      - Predefine K causal half-Gaussian low-pass kernels (different σ).
      - Apply K depthwise causal convs to x  -> Y[..., k]
      - Compute causal local features (mean, var, x)  -> tiny MLP -> softmax over K
      - Trend = Σ_k w_k * Y_k   (per time, per channel)

    x, trend: [B, T, C]
    """
    def __init__(
        self,
        sigmas=(2.5, 4.0, 6.0, 9.0, 14.0),
        truncate=4.0,
        cond_hidden=32,
        stat_window=16,
        add_x_feature=False,          # default False now
        softmax_temp=0.7,             # NEW: temperature
        use_zscore=True,              # NEW: z-score instead of raw x
        entropy_reg=0.0               # NEW: optional gating entropy penalty
    ):
        super().__init__()
        self.sigmas = list(sigmas)
        self.truncate = float(truncate)
        self.stat_window = int(stat_window)
        self.add_x_feature = bool(add_x_feature)
        self.softmax_temp = float(softmax_temp)
        self.use_zscore = bool(use_zscore)
        self.entropy_reg = float(entropy_reg)

        # features: z and logvar (2). Optionally raw x (+1).
        in_feats = 2 + (1 if self.add_x_feature else 0)
        self.cond = nn.Sequential(
            nn.Linear(in_feats, cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, len(self.sigmas))
        )

    @torch.no_grad()
    def _precompute_kernels(self, x):
        dtype, device = x.dtype, x.device
        return [
            _causal_gauss_kernel1d(s, self.truncate, dtype=dtype, device=device)
            for s in self.sigmas
        ]

    def forward(self, x):
        B, T, C = x.shape
        kernels = self._precompute_kernels(x)

        # K causal depthwise LP responses
        Y = torch.stack([_depthwise_conv1d_causal(x, k) for k in kernels], dim=-1)  # [B,T,C,K]

        # Causal local stats
        m, v = _causal_mean_var(x, win=self.stat_window)  # [B,T,C]
        z = (x - m) / (v.add(1e-6).sqrt())                # scale-invariant
        logv = (v + 1e-6).log()

        feats = [z if self.use_zscore else x, logv]       # 2 feats
        if self.add_x_feature:
            feats.append(x)                                # optional 3rd feat
        feats = torch.stack(feats, dim=-1)                 # [B,T,C,F]

        # Conditioner (pointwise MLP)
        logits = self.cond(feats)                          # [B,T,C,K]
        w = torch.softmax(logits / self.softmax_temp, dim=-1)

        trend = (Y * w).sum(dim=-1)                        # [B,T,C]

        # If you want to use entropy_reg, return it for the training loop to add:
        # ent = -(w * (w.clamp_min(1e-8)).log()).sum(dim=-1).mean()
        # stash for external collection if desired
        self._last_entropy = -(w * (w.clamp_min(1e-8)).log()).sum(dim=-1).mean().detach()

        return trend
