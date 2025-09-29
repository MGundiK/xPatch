# layers/decomp.py
import torch
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA

# NEW:
from layers.gaussma import (
    GaussianMA, GaussianMA2Pass,
    sigma_from_alpha,
)

class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta,
                 # Optional Gaussian configs
                 gauss_sigma1: float | None = None,
                 gauss_sigma2: float | None = None,
                 gauss_P1: int | None = None, gauss_mult1: float | None = None,
                 gauss_P2: int | None = None, gauss_mult2: float | None = None,
                 gauss_learnable: bool = False,
                 gauss_truncate: float = 4.0):
        super(DECOMP, self).__init__()
        self.ma_type = ma_type

        if ma_type == 'ema':
            self.ma = EMA(alpha)            # returns trend
            self._mode = 'single'
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)     # returns trend
            self._mode = 'single'
        elif ma_type == 'gauss':
            # If user doesn’t pass sigma, map EMA alpha -> sigma for parity
            sigma = gauss_sigma1 if gauss_sigma1 is not None else sigma_from_alpha(alpha)
            self.ma = GaussianMA(sigma, learnable=gauss_learnable, truncate=gauss_truncate)
            self._mode = 'single'
        elif ma_type == 'gauss2':
            # Prefer explicit (P,mult). If absent, fall back to alpha→sigma and set sigma2 ~ 4×sigma1
            if gauss_sigma1 is None and (gauss_P1 is None or gauss_mult1 is None):
                gauss_sigma1 = sigma_from_alpha(alpha)
            if gauss_sigma2 is None and (gauss_P2 is None or gauss_mult2 is None):
                if gauss_sigma1 is not None:
                    gauss_sigma2 = 4.0 * gauss_sigma1
            self.ma = GaussianMA2Pass(
                sigma1=gauss_sigma1, sigma2=gauss_sigma2,
                P1=gauss_P1, mult1=gauss_mult1, P2=gauss_P2, mult2=gauss_mult2,
                learnable=gauss_learnable, truncate=gauss_truncate
            )
            self._mode = 'double'
        else:
            raise ValueError(f"Unknown ma_type: {ma_type}")

    def forward(self, x):
        # x: [Batch, Input, Channel]
        if self.ma_type in ('ema', 'dema', 'gauss'):
            moving_average = self.ma(x)      # trend
            res = x - moving_average         # seasonal
            return res, moving_average
        elif self.ma_type == 'gauss2':
            seasonal_total, trend_final = self.ma(x)
            return seasonal_total, trend_final
