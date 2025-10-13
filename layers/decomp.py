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

from layers.trend_bank import (
    LearnableEMA, MultiEMAMixture, DebiasedEMA, AlphaBetaFilter,
    EWRLSLevel, EWMedian, AlphaCutoffFilter, CausalWindowTrend, OneEuro
)




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
        # ---------- Causal Window Smoother ----------
        cw_kind: str = "hann",
        cw_L:    int = 33,
        cw_beta: float = 8.0,
        cw_a:    int = 2,
        cw_per_channel: bool = False,
        # ---------- Learnable EMA ----------
        lem_init_alpha: float = 0.9,
        
        # ---------- Multi-EMA ----------
        mema_K: int = 3,
        mema_init_alphas: Optional[List[float]] = None,  # CSV -> list via run.py
        
        # ---------- Debiased EMA ----------
        dema_alpha: float = 0.9,
        dema_learnable: bool = False,
        
        # ---------- Alpha-Beta ----------
        ab_init_alpha: float = 0.5,
        ab_init_beta: float = 0.1,
        
        # ---------- EWRLS ----------
        ewrls_init_lambda: float = 0.98,
        ewrls_learnable: bool = True,
        ewrls_init_P: float = 1.0,
        
        # ---------- EW-Median ----------
        ewm_step: float = 0.05,
        ewm_tau_temp: float = 0.01,
        ewm_learnable_step: bool = False,
        
        # ---------- Alpha-Cutoff ----------
        ac_fs: float = 1.0,
        ac_init_fc: float = 0.05,
        ac_learnable_fc: bool = True,
        ac_fc_low: float = 1e-4,
        ac_fc_high: float = 0.5,
        
        # ---------- One-Euro ----------
        oe_min_cutoff: float = 1.0,
        oe_beta: float = 0.007,
        oe_dcutoff: float = 1.0,
        oe_fs: float = 1.0,

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
        elif self.ma_type == 'causal_window':
            self.ma = CausalWindowTrend(
                kind=cw_kind,
                L=cw_L,
                beta=cw_beta,
                a=cw_a,
                per_channel=cw_per_channel,
            )

        elif self.ma_type == 'learnable_ema':
            if channels is None: raise ValueError("learnable_ema needs 'channels'")
            self.ma = LearnableEMA(channels=channels, init_alpha=lem_init_alpha)
        
        elif self.ma_type == 'multi_ema':
            if channels is None: raise ValueError("multi_ema needs 'channels'")
            init_alphas = mema_init_alphas or [0.8, 0.9, 0.98]
            self.ma = MultiEMAMixture(channels=channels, K=mema_K, init_alphas=init_alphas)
        
        elif self.ma_type == 'debiased_ema':
            if channels is None: raise ValueError("debiased_ema needs 'channels'")
            self.ma = DebiasedEMA(channels=channels, alpha=dema_alpha, learnable=dema_learnable)
        
        elif self.ma_type == 'alpha_beta':
            if channels is None: raise ValueError("alpha_beta needs 'channels'")
            self.ma = AlphaBetaFilter(channels=channels, init_alpha=ab_init_alpha, init_beta=ab_init_beta)
        
        elif self.ma_type == 'ewrls':
            if channels is None: raise ValueError("ewrls needs 'channels'")
            self.ma = EWRLSLevel(channels=channels, init_lambda=ewrls_init_lambda,
                                 learnable=ewrls_learnable, init_P=ewrls_init_P)
        
        elif self.ma_type == 'ew_median':
            if channels is None: raise ValueError("ew_median needs 'channels'")
            self.ma = EWMedian(channels=channels, step=ewm_step,
                               tau_temp=ewm_tau_temp, learnable_step=ewm_learnable_step)
        
        elif self.ma_type == 'alpha_cutoff':
            if channels is None: raise ValueError("alpha_cutoff needs 'channels'")
            self.ma = AlphaCutoffFilter(channels=channels, fs=ac_fs,
                                        init_fc=ac_init_fc, learnable_fc=ac_learnable_fc,
                                        fc_bounds=(ac_fc_low, ac_fc_high))
        
        elif self.ma_type in ('window_hann','window_kaiser','window_lanczos','window_hann_poisson'):
            # reuse your existing causal-window knobs (cw_*) you already added earlier
            kind = self.ma_type.split('_', 1)[1]  # 'hann'|'kaiser'|'lanczos'|'hann_poisson'
            self.ma = CausalWindowTrend(kind=kind, L=cw_L, beta=cw_beta, a=cw_a, per_channel=cw_per_channel)
        
        elif self.ma_type == 'one_euro':
            self.ma = OneEuro(min_cutoff=oe_min_cutoff, beta=oe_beta,
                              dcutoff=oe_dcutoff, fs=oe_fs)
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
