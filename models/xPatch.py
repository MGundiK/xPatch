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
            adaptive_stat_window   = getattr(configs, "adaptive_stat_window", 16),
            adaptive_add_x_feature = getattr(configs, "adaptive_add_x_feature", False),
            # NEW:
            adaptive_softmax_temp  = getattr(configs, "adaptive_softmax_temp", 0.7),
            adaptive_use_zscore    = getattr(configs, "adaptive_use_zscore", False),
        )

        # --- Adaptive Gaussian V2 (Centered) ---
        adaptive_v2_kwargs = dict(
            adaptive_v2_base_sigmas   = getattr(configs, "adaptive_v2_base_sigmas", None),
            adaptive_v2_ref_seq_len   = getattr(configs, "adaptive_v2_ref_seq_len", 512),
            adaptive_v2_truncate      = getattr(configs, "adaptive_v2_truncate", 4.0),
            adaptive_v2_cond_hidden   = getattr(configs, "adaptive_v2_cond_hidden", 32),
            adaptive_v2_stat_window   = getattr(configs, "adaptive_v2_stat_window", 16),
            adaptive_v2_softmax_temp  = getattr(configs, "adaptive_v2_softmax_temp", 0.7),
            adaptive_v2_use_slope     = getattr(configs, "adaptive_v2_use_slope", True),
        )


        # --- Hybrid EMA + DoG ---
        dog_kwargs = dict(
            dog_sigma1   = getattr(configs, "dog_sigma1", 4.2),
            dog_sigma2   = getattr(configs, "dog_sigma2", 96.0),
            dog_truncate = getattr(configs, "dog_truncate", 4.0),
        )

        # Learnable LP
        lp_kwargs = dict(
            lp_kernel    = getattr(configs, "lp_kernel_size", 21),
            lp_mode      = getattr(configs, "lp_mode", "centered"),
            lp_ema_alpha = getattr(configs, "lp_ema_alpha", 0.3),
        )

        # TCN smoother
        tcn_kwargs = dict(
            tcn_hidden_mult = getattr(configs, "tcn_hidden_mult", 1.0),
            tcn_blocks      = getattr(configs, "tcn_n_blocks", 2),
            tcn_kernel      = getattr(configs, "tcn_kernel", 7),
            tcn_beta        = getattr(configs, "tcn_beta", 0.3),
            tcn_final_avg   = getattr(configs, "tcn_final_avg", 0),
        )

        # --- Causal Window Smoother ---
        cw_kwargs = dict(
            cw_kind        = getattr(configs, "cw_kind", "hann"),   # hann | kaiser | lanczos | hann_poisson
            cw_L           = getattr(configs, "cw_L", 33),
            cw_beta        = getattr(configs, "cw_beta", 8.0),      # kaiser only
            cw_a           = getattr(configs, "cw_a", 2),           # lanczos only
            cw_per_channel = getattr(configs, "cw_per_channel", False),
        )

        # ===============================
        # Trend-bank (new fast learnables)
        # ===============================
        fastema_kwargs = dict(
            fastema_init_alpha = getattr(configs, "fastema_init_alpha", 0.9),
            fastema_debias     = getattr(configs, "fastema_debias", False),
        )

        ab_kwargs = dict(
            ab_init_alpha = getattr(configs, "ab_init_alpha", 0.5),
            ab_init_beta  = getattr(configs, "ab_init_beta", 0.1),
        )

        kaiser_kwargs = dict(
            kaiser_L             = getattr(configs, "kaiser_L", 129),
            kaiser_num_kernels   = getattr(configs, "kaiser_num_kernels", 1),
            kaiser_init_beta     = getattr(configs, "kaiser_init_beta", 6.0),
            kaiser_learnable_mix = getattr(configs, "kaiser_learnable_mix", False),
        )

        hannp_kwargs = dict(
            hannp_L               = getattr(configs, "hannp_L", 129),
            hannp_num_kernels     = getattr(configs, "hannp_num_kernels", 1),
            hannp_init_lambda     = getattr(configs, "hannp_init_lambda", 0.02),
            hannp_learnable_mix   = getattr(configs, "hannp_learnable_mix", False),
        )

        ewrls_kwargs = dict(
            ewrls_init_lambda = getattr(configs, "ewrls_init_lambda", 0.98),
        )

        huber_kwargs = dict(
            huber_init_alpha = getattr(configs, "huber_init_alpha", 0.9),
            huber_delta      = getattr(configs, "huber_delta", 1.0),
        )

        # --- Build DECOMP ---
        self.decomp = DECOMP(
            ma_type=self.ma_type,
            alpha=alpha,
            beta=beta,
            channels=c_in,             # required by several modules
            # legacy / existing:
            **gauss_kwargs,
            **adaptive_kwargs,
            **adaptive_v2_kwargs,
            **dog_kwargs,
            **lp_kwargs,
            **tcn_kwargs,
            **cw_kwargs,
            # trend-bank new modules:
            **fastema_kwargs,
            **ab_kwargs,
            **kaiser_kwargs,
            **hannp_kwargs,
            **ewrls_kwargs,
            **huber_kwargs,
        )

        # Main forecaster network
        #self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)
        # === Trend head plumbing (safe fallback happens inside Network) ===
        # ===============================
        # Trend head (optional, minimal)
        # ===============================

        # ---------------- Trend head config (assemble kwargs only) ----------------
        trend_head = getattr(configs, "trend_head", None)
        
        # FIR head
        fir_kwargs = dict(
            k_list    = getattr(configs, "fir_k_list", None),
            d_list    = getattr(configs, "fir_d_list", None),
            channels  = getattr(configs, "fir_channels", None),
            gelu      = getattr(configs, "fir_gelu", None),      # 0/1
            aa_pool   = getattr(configs, "fir_aa_pool", None),   # 0/1
            smooth_l2 = getattr(configs, "fir_smooth_l2", None),
        )
        if isinstance(fir_kwargs["k_list"], str) and fir_kwargs["k_list"]:
            fir_kwargs["k_list"] = [int(x) for x in fir_kwargs["k_list"].split(",")]
        if isinstance(fir_kwargs["d_list"], str) and fir_kwargs["d_list"]:
            fir_kwargs["d_list"] = [int(x) for x in fir_kwargs["d_list"].split(",")]
        if fir_kwargs["gelu"] is not None:
            fir_kwargs["gelu"] = bool(int(fir_kwargs["gelu"]))
        if fir_kwargs["aa_pool"] is not None:
            fir_kwargs["aa_pool"] = bool(int(fir_kwargs["aa_pool"]))
        
        # Basis head
        basis_kwargs = dict(
            poly_degree = getattr(configs, "basis_poly_degree", None),
            fourier_k   = getattr(configs, "basis_fourier_k", None),
            normalize_t = getattr(configs, "basis_normalize_t", None),  # 0/1
        )
        if basis_kwargs["normalize_t"] is not None:
            basis_kwargs["normalize_t"] = bool(int(basis_kwargs["normalize_t"]))
        
        # Local linear head
        local_lin_kwargs = dict(
            k = getattr(configs, "local_lin_k", None),
        )
        
        # Delta head
        delta_kwargs = dict(
            mode   = getattr(configs, "delta_mode", None),  # 'last' | 'lin'
            k      = getattr(configs, "delta_k", None),
            hidden = getattr(configs, "delta_hidden", None),
        )
        if delta_kwargs["mode"] is not None:
            m = str(delta_kwargs["mode"]).lower()
            if m not in ("last", "lin"):
                delta_kwargs["mode"] = None
        
        # Downsampled MLP head
        ds_mlp_kwargs = dict(
            stride = getattr(configs, "ds_mlp_stride", None),
            hidden = getattr(configs, "ds_mlp_hidden", None),
            hann   = getattr(configs, "ds_mlp_hann", None),    # 0/1
            causal = getattr(configs, "ds_mlp_causal", None),  # 0/1
        )
        if ds_mlp_kwargs["hann"] is not None:
            ds_mlp_kwargs["hann"] = bool(int(ds_mlp_kwargs["hann"]))
        if ds_mlp_kwargs["causal"] is not None:
            ds_mlp_kwargs["causal"] = bool(int(ds_mlp_kwargs["causal"]))
        
        # Choose which kwargs to pass
        trend_cfg = None
        if isinstance(trend_head, str) and trend_head.strip():
            if trend_head == "fir":
                trend_cfg = {k: v for k, v in fir_kwargs.items() if v is not None}
            elif trend_head == "basis":
                trend_cfg = {k: v for k, v in basis_kwargs.items() if v is not None}
            elif trend_head == "local_lin":
                trend_cfg = {k: v for k, v in local_lin_kwargs.items() if v is not None}
            elif trend_head == "delta":
                trend_cfg = {k: v for k, v in delta_kwargs.items() if v is not None}
            elif trend_head == "ds_mlp":
                trend_cfg = {k: v for k, v in ds_mlp_kwargs.items() if v is not None}
        # else leave None to trigger baseline in Network
        
        # ---------------- Create Network (Network handles registry + fallback) ---------------
        self.net = Network(
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            trend_head=trend_head,
            trend_cfg=trend_cfg,
        )


    def forward(self, x):
        # x: [Batch, Input, Channel]
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':   # no decomposition
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
