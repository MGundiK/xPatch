# layers/decomp.py
import torch
from torch import nn

# Base smoothers
from layers.ema import EMA
from layers.dema import DEMA

# Gaussian smoother
from layers.gaussma import GaussianMA, sigma_from_alpha

# New modules
from layers.gaussma_adaptive import AdaptiveGaussianTrend   # adaptive multi-scale Gaussian
from layers.doghybrid import HybridEMA_DoG                  # EMA + DoG combo
from layers.learnablelp import LearnableLP                  # Learnable LP kernel
from layers.tc_smoother import TCSmoother                   # Temporal ConvNet smoother


class DECOMP(nn.Module):
    """
    Series decomposition block for xPatch.
    Selectable smoother via `ma_type`.
    Returns: (seasonal, trend)
    """

    def __init__(self,
                 ma_type: str,
                 alpha: float = 0.3,
                 beta: float = 0.3,
                 # Optional Gaussian parameters
                 gauss_sigma1: float | None = None,
                 gauss_P1: int | None = None,
                 gauss_mult1: float | None = None,
                 gauss_learnable: bool = False,
                 gauss_truncate: float = 4.0,
                 # Optional for adaptive Gaussian
                 gauss_sigmas: list[float] | None = None,
                 gauss_pool: int = 16,
                 gauss_cond_hidden: int = 32,
                 # Optional for learnable LP / TCN
                 channels: int | None = None,
                 lp_kernel: int = 21,
                 lp_mode: str = "centered",
                 tcn_blocks: int = 2,
                 tcn_beta: float = 0.3):
        super().__init__()
        self.ma_type = ma_type.lower()

        if self.ma_type == 'ema':
            self.ma = EMA(alpha)

        elif self.ma_type == 'dema':
            self.ma = DEMA(alpha, beta)

        elif self.ma_type == 'gauss':
            # Use provided sigma or map from alpha
            sigma = gauss_sigma1 or sigma_from_alpha(alpha)
            self.ma = GaussianMA(sigma, learnable=gauss_learnable, truncate=gauss_truncate)

        elif self.ma_type == 'gauss_adaptive':
            sigmas = gauss_sigmas or [2.5, 4.0, 6.0, 9.0, 14.0]
            self.ma = AdaptiveGaussianTrend(sigmas=sigmas, truncate=gauss_truncate,
                                            cond_hidden=gauss_cond_hidden, pool=gauss_pool)

        elif self.ma_type == 'doghybrid':
            sigma1 = gauss_sigma1 or 4.0
            sigma2 = gauss_P1 or 96.0
            self.ma = HybridEMA_DoG(alpha=alpha, sigma1=sigma1, sigma2=sigma2, truncate=gauss_truncate)

        elif self.ma_type == 'lp_learnable':
            assert channels is not None, "channels must be provided for LearnableLP"
            self.ma = LearnableLP(channels=channels, kernel_size=lp_kernel, mode=lp_mode, ema_alpha=alpha)

        elif self.ma_type == 'tcn_trend':
            assert channels is not None, "channels must be provided for TCSmoother"
            self.ma = TCSmoother(channels=channels, n_blocks=tcn_blocks, beta=tcn_beta)

        else:
            raise ValueError(f"Unknown ma_type: {ma_type}")

    def forward(self, x: torch.Tensor):
        """
        Input:  x [B, T, C]
        Output: (seasonal, trend)
        """
        out = self.ma(x)
        if isinstance(out, tuple):
            # Already decomposed (e.g. HybridEMA_DoG)
            return out
        else:
            # Single trend smoother
            trend = out
            seasonal = x - trend
            return seasonal, trend
