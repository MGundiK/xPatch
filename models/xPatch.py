import torch
import torch.nn as nn
import math

from layers.decomp import DECOMP
from layers.network import Network
from layers.revin import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_in = configs.enc_in

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        # Decomposition method
        self.ma_type = configs.ma_type.lower()
        alpha = getattr(configs, "alpha", 0.3)
        beta  = getattr(configs, "beta", 0.3)

        # --- Optional Gaussian parameters ---
        gauss_kwargs = dict(
            gauss_sigma1   = getattr(configs, "gauss_sigma1", None),
            gauss_P1       = getattr(configs, "gauss_P1", None),
            gauss_mult1    = getattr(configs, "gauss_mult1", None),
            gauss_learnable= getattr(configs, "gauss_learnable", False),
            gauss_truncate = getattr(configs, "gauss_truncate", 4.0),
        )

        # --- Optional parameters for other methods ---
        adaptive_kwargs = dict(
            sigmas    = getattr(configs, "adaptive_sigmas", (2.5, 4.0, 6.0, 9.0, 14.0)),
            truncate  = getattr(configs, "adaptive_truncate", 4.0),
            cond_hidden = getattr(configs, "adaptive_cond_hidden", 32),
            pool      = getattr(configs, "adaptive_pool", 16),
        )

        dog_kwargs = dict(
            sigma1   = getattr(configs, "dog_sigma1", 4.2),
            sigma2   = getattr(configs, "dog_sigma2", 96.0),
            truncate = getattr(configs, "dog_truncate", 4.0),
        )

        lp_kwargs = dict(
            channels     = c_in,
            kernel_size  = getattr(configs, "lp_kernel_size", 21),
            mode         = getattr(configs, "lp_mode", "centered"),
            ema_alpha    = getattr(configs, "lp_ema_alpha", 0.3),
        )

        tcn_kwargs = dict(
            channels    = c_in,
            hidden_mult = getattr(configs, "tcn_hidden_mult", 1.0),
            n_blocks    = getattr(configs, "tcn_n_blocks", 2),
            kernel      = getattr(configs, "tcn_kernel", 7),
            beta        = getattr(configs, "tcn_beta", 0.3),
            final_avg   = getattr(configs, "tcn_final_avg", 0),
        )

        # --- Build DECOMP ---
        self.decomp = DECOMP(
            ma_type=self.ma_type,
            alpha=alpha,
            beta=beta,
            **gauss_kwargs,
            **adaptive_kwargs,
            **dog_kwargs,
            **lp_kwargs,
            **tcn_kwargs
        )

        # Main forecaster network
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)


    def forward(self, x):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        # Decomposition
        if self.ma_type == 'reg':   # no decomposition
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
