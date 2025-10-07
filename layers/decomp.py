# layers/decomp.py
import torch
from torch import nn
from typing import Optional, List

# Base smoothers
from layers.ema import EMA
from layers.dema import DEMA

# Gaussian (single-pass)
from layers.gaussma import GaussianMA, sigma_from_alpha

# New modules
from layers.gaussma_adaptive_causal import AdaptiveGaussianTrendCausal  # NEW causal adaptive
from layers.doghybrid import HybridEMA_DoG
from layers.learnablelp import LearnableLP
from layers.tc_smoother import TCSmoother


class DECOMP(nn.Module):
    """
    Series decomposition block for xPatch.
    Always returns (seasonal, trend).

    Supported ma_type:
      - 'reg' (handled in Model)
      - 'ema', 'dema'
      - 'gauss'
      - 'gauss_adaptive_causal'
      - 'doghybrid'
      - 'lp_learnable'
      - 'tcn_trend'
    """

    def __init__(
        self,
        ma_type: str,
        alpha: float = 0.3,
        beta: float = 0.3,

        # ---------- Gaussian (single-pass) ----------
        gauss_sigma1: Optional[float] = None,
        gauss_P1: Optional[int] = None,
        gauss_mult1: Optional[float] = None,
        gauss_learnable: bool = False,
        gauss_truncate: float = 4.0,

        # ---------- Adaptive Gaussian (causal) ----------
        # Adaptive Gaussian (CAUSAL)
        adaptive_sigmas: Optional[List[float]] = None,
        adaptive_truncate: float = 4.0,
        adaptive_cond_hidden: int = 32,
        adaptive_stat_window: int = 16,          # <-- use this
        adaptive_add_x_feature: bool = True,     # <-- and this

        # ---------- DoG Hybrid ----------
        dog_sigma1: float = 4.2,
        dog_sigma2: float = 96.0,
        dog_truncate: float = 4.0,

        # ---------- Learnable LP ----------
        channels: Optional[int] = None,
        lp_kernel: int = 21,
        lp_mode: str = "centered",
        lp_ema_alpha: float = 0.3,

        # ---------- TCN smoother ----------
        tcn_hidden_mult: float = 1.0,
        tcn_blocks: int = 2,
        tcn_kernel: int = 7,
        tcn_beta: float = 0.3,
        tcn_final_avg: int = 0,
    ):
        super().__init__()
        self.ma_type = ma_type.lower()

        if self.ma_type == 'ema':
            self.ma = EMA(alpha)

        elif self.ma_type == 'dema':
            self.ma = DEMA(alpha, beta)

        elif self.ma_type == 'gauss':
            if gauss_sigma1 is not None:
                sigma = float(gauss_sigma1)
            elif (gauss_P1 is not None) and (gauss_mult1 is not None):
                sigma = float(gauss_P1) * float(gauss_mult1)
            else:
                sigma = sigma_from_alpha(alpha)
            self.ma = GaussianMA(sigma, learnable=gauss_learnable, truncate=gauss_truncate)

        elif self.ma_type == 'gauss_adaptive_causal':
            sigmas = adaptive_sigmas or [2.5, 4.0, 6.0, 9.0, 14.0]
            self.ma = AdaptiveGaussianTrendCausal(
                sigmas=sigmas,
                truncate=adaptive_truncate,
                cond_hidden=adaptive_cond_hidden,
                stat_window=adaptive_stat_window,   # <-- renamed
                add_x_feature=adaptive_add_x_feature,  # <-- new toggle
            )


        elif self.ma_type == 'doghybrid':
            self.ma = HybridEMA_DoG(
                alpha=alpha,
                sigma1=dog_sigma1,
                sigma2=dog_sigma2,
                truncate=dog_truncate,
            )

        elif self.ma_type == 'lp_learnable':
            if channels is None:
                raise ValueError("DECOMP(lp_learnable): 'channels' must be provided (configs.enc_in).")
            self.ma = LearnableLP(
                channels=channels,
                kernel_size=lp_kernel,
                mode=lp_mode,
                ema_alpha=lp_ema_alpha,
            )

        elif self.ma_type == 'tcn_trend':
            if channels is None:
                raise ValueError("DECOMP(tcn_trend): 'channels' must be provided (configs.enc_in).")
            self.ma = TCSmoother(
                channels=channels,
                hidden_mult=tcn_hidden_mult,
                n_blocks=tcn_blocks,
                kernel=tcn_kernel,
                beta=tcn_beta,
                final_avg=tcn_final_avg,
            )

        else:
            raise ValueError(f"Unknown ma_type: {ma_type}")

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, C]
        returns: (seasonal, trend)
        """
        out = self.ma(x)
        if isinstance(out, tuple):
            return out  # (seasonal, trend)
        else:
            trend = out
            seasonal = x - trend
            return seasonal, trend
