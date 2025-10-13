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

        # --- Gaussian (single-pass) ---
        gauss_kwargs = dict(
            gauss_sigma1    = getattr(configs, "gauss_sigma1", None),
            gauss_P1        = getattr(configs, "gauss_P1", None),
            gauss_mult1     = getattr(configs, "gauss_mult1", None),
            gauss_learnable = getattr(configs, "gauss_learnable", False),
            gauss_truncate  = getattr(configs, "gauss_truncate", 4.0),
        )


        # --- Adaptive Gaussian (CAUSAL) ---
        adaptive_kwargs = dict(
            adaptive_sigmas        = getattr(configs, "adaptive_sigmas", (2.5, 4.0, 6.0, 9.0, 14.0)),
            adaptive_truncate      = getattr(configs, "adaptive_truncate", 4.0),
            adaptive_cond_hidden   = getattr(configs, "adaptive_cond_hidden", 32),
            adaptive_stat_window   = getattr(configs, "adaptive_stat_window", 16),  # <-- was adaptive_pool
            adaptive_add_x_feature = getattr(configs, "adaptive_add_x_feature", True),  # <-- new
        )


        # --- Hybrid EMA + DoG (namespaced) ---
        dog_kwargs = dict(
            dog_sigma1   = getattr(configs, "dog_sigma1", 4.2),
            dog_sigma2   = getattr(configs, "dog_sigma2", 96.0),
            dog_truncate = getattr(configs, "dog_truncate", 4.0),
        )

        # Learnable LP (no 'channels' here)
        lp_kwargs = dict(
            lp_kernel    = getattr(configs, "lp_kernel_size", 21),
            lp_mode      = getattr(configs, "lp_mode", "centered"),
            lp_ema_alpha = getattr(configs, "lp_ema_alpha", 0.3),
        )

        # TCN smoother (no 'channels' here)
        tcn_kwargs = dict(
            tcn_hidden_mult = getattr(configs, "tcn_hidden_mult", 1.0),
            tcn_blocks      = getattr(configs, "tcn_n_blocks", 2),
            tcn_kernel      = getattr(configs, "tcn_kernel", 7),
            tcn_beta        = getattr(configs, "tcn_beta", 0.3),
            tcn_final_avg   = getattr(configs, "tcn_final_avg", 0),
        )

        # --- Causal Window Smoother (namespaced) ---
        cw_kwargs = dict(
            cw_kind        = getattr(configs, "cw_kind", "hann"),
            cw_L           = getattr(configs, "cw_L", 33),
            cw_beta        = getattr(configs, "cw_beta", 8.0),
            cw_a           = getattr(configs, "cw_a", 2),
            cw_per_channel = getattr(configs, "cw_per_channel", False),
        )

        # --- Learnable EMA ---
        lem_kwargs = dict(
            lem_init_alpha = getattr(configs, "lem_init_alpha", 0.9),
        )
        
        # --- Multi-EMA ---
        mema_kwargs = dict(
            mema_K           = getattr(configs, "mema_K", 3),
            mema_init_alphas = getattr(configs, "mema_init_alphas", None),  # list or None
        )
        
        # --- Debiased EMA ---
        dema_kwargs = dict(
            dema_alpha     = getattr(configs, "dema_alpha", 0.9),
            dema_learnable = getattr(configs, "dema_learnable", False),
        )
        
        # --- Alpha-Beta ---
        ab_kwargs = dict(
            ab_init_alpha = getattr(configs, "ab_init_alpha", 0.5),
            ab_init_beta  = getattr(configs, "ab_init_beta", 0.1),
        )
        
        # --- EWRLS ---
        ewrls_kwargs = dict(
            ewrls_init_lambda = getattr(configs, "ewrls_init_lambda", 0.98),
            ewrls_learnable   = getattr(configs, "ewrls_learnable", True),
            ewrls_init_P      = getattr(configs, "ewrls_init_P", 1.0),
        )
        
        # --- EW-Median ---
        ewm_kwargs = dict(
            ewm_step            = getattr(configs, "ewm_step", 0.05),
            ewm_tau_temp        = getattr(configs, "ewm_tau_temp", 0.01),
            ewm_learnable_step  = getattr(configs, "ewm_learnable_step", False),
        )
        
        # --- Alpha-Cutoff ---
        ac_kwargs = dict(
            ac_fs           = getattr(configs, "ac_fs", 1.0),
            ac_init_fc      = getattr(configs, "ac_init_fc", 0.05),
            ac_learnable_fc = getattr(configs, "ac_learnable_fc", True),
            ac_fc_low       = getattr(configs, "ac_fc_low", 1e-4),
            ac_fc_high      = getattr(configs, "ac_fc_high", 0.5),
        )
        
        # --- One-Euro ---
        oe_kwargs = dict(
            oe_min_cutoff = getattr(configs, "oe_min_cutoff", 1.0),
            oe_beta       = getattr(configs, "oe_beta", 0.007),
            oe_dcutoff    = getattr(configs, "oe_dcutoff", 1.0),
            oe_fs         = getattr(configs, "oe_fs", 1.0),
        )
        
        # --- Build DECOMP ---
        self.decomp = DECOMP(
            ma_type=self.ma_type,
            alpha=alpha,
            beta=beta,
            channels=c_in,
            **gauss_kwargs,
            **adaptive_kwargs,
            **dog_kwargs,
            **lp_kwargs,
            **tcn_kwargs,
            **cw_kwargs,       # you already added earlier
            **lem_kwargs,
            **mema_kwargs,
            **dema_kwargs,
            **ab_kwargs,
            **ewrls_kwargs,
            **ewm_kwargs,
            **ac_kwargs,
            **oe_kwargs,
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
