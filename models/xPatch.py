import torch
import torch.nn as nn
import math

from layers.decomp import DECOMP
from layers.network import Network
# from layers.network_mlp import NetworkMLP
# from layers.network_cnn import NetworkCNN
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

        # Moving Average / Decomposition
        self.ma_type = configs.ma_type
        alpha = configs.alpha                   # EMA smoothing
        beta  = configs.beta                    # DEMA smoothing

        # --- NEW: read optional Gaussian params from configs (if present) ---
        gauss_kwargs = dict(
            gauss_sigma1   = getattr(configs, "gauss_sigma1", None),
            gauss_sigma2   = getattr(configs, "gauss_sigma2", None),
            gauss_P1       = getattr(configs, "gauss_P1", None),
            gauss_mult1    = getattr(configs, "gauss_mult1", None),
            gauss_P2       = getattr(configs, "gauss_P2", None),
            gauss_mult2    = getattr(configs, "gauss_mult2", None),
            gauss_learnable= getattr(configs, "gauss_learnable", False),
            gauss_truncate = getattr(configs, "gauss_truncate", 4.0),
        )

        # Construct DECOMP (EMA/DEMA unchanged; GAUSS/GAUSS2 use gauss_kwargs)
        self.decomp = DECOMP(self.ma_type, alpha, beta, **gauss_kwargs)

        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)
        # self.net_mlp = NetworkMLP(seq_len, pred_len)
        # self.net_cnn = NetworkCNN(seq_len, pred_len, patch_len, stride, padding_patch)

    def forward(self, x):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':   # no decomposition
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
