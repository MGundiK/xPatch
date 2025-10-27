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
from layers.causal_window import CausalWindowTrend

# NEW: the factory for your new fast learnable modules
from layers.trend_bank import build_trend_module



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
         # Gaussian (single-pass)
         gauss_sigma1: Optional[float] = None,
         gauss_P1: Optional[int] = None,
         gauss_mult1: Optional[float] = None,
         gauss_learnable: bool = False,
         gauss_truncate: float = 4.0,
         # Adaptive Gaussian (CAUSAL)
         adaptive_sigmas: Optional[List[float]] = None,
         adaptive_truncate: float = 4.0,
         adaptive_cond_hidden: int = 32,
         adaptive_stat_window: int = 16,
         adaptive_add_x_feature: bool = True,
        # NEW:
         adaptive_softmax_temp: float = 0.7,
         adaptive_use_zscore: bool = False,
         # DoG hybrid
         dog_sigma1: float = 4.2,
         dog_sigma2: float = 96.0,
         dog_truncate: float = 4.0,
         # LP, TCN (need channels)
         channels: Optional[int] = None,
         lp_kernel: int = 21,
         lp_mode: str = "centered",
         lp_ema_alpha: float = 0.3,
         tcn_hidden_mult: float = 1.0,
         tcn_blocks: int = 2,
         tcn_kernel: int = 7,
         tcn_beta: float = 0.3,
         tcn_final_avg: int = 0,
        # ---------- Causal Window Smoother ----------
         cw_kind: str = "hann",
         cw_L:    int = 33,
         cw_beta: float = 8.0,
         cw_a:    int = 2,
         cw_per_channel: bool = False,

        # ---- NEW: pass-through kwargs for trend_bank modules ----
        **trend_kwargs

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
                stat_window=adaptive_stat_window,
                add_x_feature=adaptive_add_x_feature,
                # NEW pass-through
                softmax_temp=adaptive_softmax_temp,
                use_zscore=adaptive_use_zscore,
            )
            print("[DECOMP] AGF params:",
                  dict(sigmas=sigmas,
                       truncate=adaptive_truncate,
                       cond_hidden=adaptive_cond_hidden,
                       stat_window=adaptive_stat_window,
                       add_x_feature=adaptive_add_x_feature,
                       softmax_temp=adaptive_softmax_temp,
                       use_zscore=adaptive_use_zscore))

        elif self.ma_type == 'doghybrid':
            self.ma = HybridEMA_DoG(alpha=alpha, sigma1=dog_sigma1, sigma2=dog_sigma2, truncate=dog_truncate)

        elif self.ma_type == 'lp_learnable':
            if channels is None:
                raise ValueError("lp_learnable requires 'channels' (configs.enc_in).")
            self.ma = LearnableLP(channels=channels, kernel_size=lp_kernel, mode=lp_mode, ema_alpha=lp_ema_alpha)

        elif self.ma_type == 'tcn_trend':
            if channels is None:
                raise ValueError("tcn_trend requires 'channels' (configs.enc_in).")
            self.ma = TCSmoother(channels=channels, hidden_mult=tcn_hidden_mult, n_blocks=tcn_blocks,
                                 kernel=tcn_kernel, beta=tcn_beta, final_avg=tcn_final_avg)

        elif self.ma_type in ('window_hann','window_kaiser','window_lanczos','window_hann_poisson'):
            # reuse your existing causal-window knobs (cw_*) you already added earlier
            kind = self.ma_type.split('_', 1)[1]  # 'hann'|'kaiser'|'lanczos'|'hann_poisson'
            self.ma = CausalWindowTrend(kind=kind, L=cw_L, beta=cw_beta, a=cw_a, per_channel=cw_per_channel)

        # ---- NEW: your fast learnable family via the factory ----
        elif self.ma_type in {
            "fast_ema", "alpha_beta",
            "kaiser_fir", "hann_poisson_fir",
            "ewrls_fast", "huber_ema",
            "fast_multi_ema"
        }:
            if channels is None:
                raise ValueError(f"{self.ma_type} requires 'channels' (configs.enc_in).")
            self.ma = build_trend_module(self.ma_type, channels=channels, **trend_kwargs)

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
