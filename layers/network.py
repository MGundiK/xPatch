# layers/network.py
# xPatch Network with configurable trend stream, optional multi-scale patching,
# optional cross-patch attention, and optional RoRA (Rotational Rank Adaptation).
#
# Backward compatible: when use_multiscale=False, use_cross_attn=False, and
# use_rora=False, the architecture is IDENTICAL to the original.
#
# RoRA provides geometric reorientation of patch representations, based on the
# insight that the conv backbone creates piecewise-linear geometry that may
# need global rotation for the linear head to separate effectively.

import torch
from torch import nn

from layers.trend_heads import (
    BaselineMLPTrendHead,
    FIRTrendHead,
    BasisTrendHead,
    LocalLinearTrendHead,
    DeltaTrendHead,
    DownsampledMLPTrendHead,
)

# Import RoRA (optional, graceful fallback if not available)
try:
    from layers.rora import PatchRoRA, RoRABlock
    RORA_AVAILABLE = True
except ImportError:
    RORA_AVAILABLE = False


_TREND_FACTORY = {
    "mlp_baseline": BaselineMLPTrendHead,
    "fir":          FIRTrendHead,
    "basis":        BasisTrendHead,
    "local_lin":    LocalLinearTrendHead,
    "delta":        DeltaTrendHead,
    "ds_mlp":       DownsampledMLPTrendHead,
}


class Network(nn.Module):
    """
    xPatch forecasting network.

    Architecture (seasonal stream):
        patches -> fc1 (expand) -> depthwise conv (compress) + residual
        -> pointwise conv -> [optional attention] -> [optional RoRA] -> flatten head

    Options:
        use_multiscale : extract patches at multiple resolutions
        use_cross_attn : add lightweight self-attention after conv backbone
        use_rora       : add rotational rank adaptation (geometric reorientation)

    When multi-scale is enabled, patches from all scales are projected to
    d_model = patch_len and concatenated along the token dimension.  The
    expand->compress backbone reuses the same layer structure -- only the
    number of tokens (patch_num) changes.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        patch_len,
        stride,
        padding_patch,
        trend_head=None,
        trend_cfg=None,
        # ---- Phase 2: multi-scale patching ----
        use_multiscale=False,
        ms_patch_lens=None,       # e.g. [8, 16, 32]
        ms_patch_strides=None,    # e.g. [4, 8, 16]; default: pl // 2
        # ---- Phase 2: cross-patch attention ----
        use_cross_attn=False,
        attn_heads=4,
        attn_dropout=0.1,
        attn_use_ffn=False,
        # ---- Phase 3: RoRA (Rotational Rank Adaptation) ----
        use_rora=False,
        rora_rank=4,
        rora_mode='feature',      # 'feature', 'patch', or 'both'
        rora_method='cayley',     # 'cayley' or 'taylor'
    ):
        super(Network, self).__init__()

        # Core parameters
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.use_multiscale = use_multiscale and (ms_patch_lens is not None)
        self.use_cross_attn = use_cross_attn
        self.use_rora = use_rora and RORA_AVAILABLE
        
        if use_rora and not RORA_AVAILABLE:
            print("[xPatch][Network] Warning: use_rora=True but RoRA module not available. "
                  "Ensure layers/rora.py exists. Continuing without RoRA.")

        # Working dimension = patch_len (reuses original backbone exactly)
        d_model = patch_len
        self.d_model = d_model
        self.dim = d_model * d_model          # expansion dim (= patch_len**2)

        # ==================================================================
        # Patch Extraction
        # ==================================================================
        if self.use_multiscale:
            # --- Multi-scale patching ---
            if ms_patch_strides is None:
                ms_patch_strides = [pl // 2 for pl in ms_patch_lens]
            assert len(ms_patch_lens) == len(ms_patch_strides), (
                f"patch_lens ({len(ms_patch_lens)}) and "
                f"strides ({len(ms_patch_strides)}) must match"
            )
            self._ms_patch_lens = list(ms_patch_lens)
            self._ms_strides = list(ms_patch_strides)

            # Per-scale: padding + linear projection to d_model
            self.ms_pads = nn.ModuleList()
            self.ms_projs = nn.ModuleList()
            total_patches = 0
            for pl, st in zip(ms_patch_lens, ms_patch_strides):
                n_patches = (seq_len - pl) // st + 1
                if padding_patch == 'end':
                    self.ms_pads.append(nn.ReplicationPad1d((0, st)))
                    n_patches += 1
                else:
                    self.ms_pads.append(nn.Identity())
                self.ms_projs.append(nn.Linear(pl, d_model))
                total_patches += n_patches

            self.patch_num = total_patches
        else:
            # --- Original single-scale patching (unchanged) ---
            self.patch_num = (seq_len - patch_len) // stride + 1
            if padding_patch == 'end':
                self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
                self.patch_num += 1

        # ==================================================================
        # Non-linear Stream  (seasonal / residual)
        # ==================================================================
        # Patch Embedding (expand: d_model -> dim = d_model**2)
        self.fc1 = nn.Linear(d_model, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise (compress: dim -> d_model)
        self.conv1 = nn.Conv1d(
            self.patch_num, self.patch_num,
            d_model, d_model,
            groups=self.patch_num,
        )
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream (dim -> d_model)
        self.fc2 = nn.Linear(self.dim, d_model)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # ==================================================================
        # Optional Cross-Patch Attention  (inserted after conv backbone)
        # ==================================================================
        if use_cross_attn:
            # Ensure head count divides d_model
            eff_heads = min(attn_heads, d_model)
            while d_model % eff_heads != 0 and eff_heads > 1:
                eff_heads -= 1

            self.attn_norm = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=eff_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            self.attn_drop = nn.Dropout(attn_dropout)

            if attn_use_ffn:
                ffn_dim = d_model * 4
                self.attn_ffn = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(attn_dropout),
                    nn.Linear(ffn_dim, d_model),
                    nn.Dropout(attn_dropout),
                )
            else:
                self.attn_ffn = None

        # ==================================================================
        # Optional RoRA (Rotational Rank Adaptation)
        # ==================================================================
        if self.use_rora:
            self.rora = PatchRoRA(
                d_model=d_model,
                num_patches=self.patch_num,
                rank=rora_rank,
                mode=rora_mode,
                method=rora_method,
            )

        # ==================================================================
        # Flatten Head
        # ==================================================================
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * d_model, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # ==================================================================
        # Trend Stream (configurable with safe fallback)
        # ==================================================================
        if not trend_head or (isinstance(trend_head, str) and
                              trend_head.strip() == ""):
            selected_head = "mlp_baseline"
        elif trend_head in _TREND_FACTORY:
            selected_head = trend_head
        else:
            print(
                f"[xPatch][Network] Warning: unknown trend_head='{trend_head}'. "
                f"Falling back to 'mlp_baseline'. "
                f"Options: {list(_TREND_FACTORY.keys())}"
            )
            selected_head = "mlp_baseline"

        if trend_cfg is None:
            trend_cfg = {}
        if selected_head != (trend_head or "mlp_baseline"):
            trend_cfg = {}

        self.trend = _TREND_FACTORY[selected_head](
            seq_len=seq_len, pred_len=pred_len, **trend_cfg
        )

        # ==================================================================
        # Streams Concatenation
        # ==================================================================
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    # ------------------------------------------------------------------
    def forward(self, s, t):
        """
        Args:
            s: seasonal component  [Batch, Input, Channel]
            t: trend component     [Batch, Input, Channel]
        Returns:
            prediction             [Batch, Output, Channel]
        """
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend

        s = s.permute(0, 2, 1)  # to [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # to [Batch, Channel, Input]

        # Channel split for channel independence
        B = s.shape[0]  # Batch size
        C = s.shape[1]  # Channel size
        I = s.shape[2]  # Input size
        s = torch.reshape(s, (B * C, I))  # [Batch*Channel, Input]
        t = torch.reshape(t, (B * C, I))  # [Batch*Channel, Input]

        # ============== Patch Extraction ==============
        if self.use_multiscale:
            # Multi-scale: per-scale pad -> unfold -> project -> concatenate
            tokens = []
            for i, (pl, st) in enumerate(
                zip(self._ms_patch_lens, self._ms_strides)
            ):
                s_i = self.ms_pads[i](s)                               # [BC, I+pad]
                patches = s_i.unfold(dimension=-1, size=pl, step=st)   # [BC, N_i, pl]
                tokens.append(self.ms_projs[i](patches))               # [BC, N_i, d_model]
            s = torch.cat(tokens, dim=1)                               # [BC, N_total, d_model]
        else:
            # Original single-scale
            # Patching
            if self.padding_patch == 'end':
                s = self.padding_patch_layer(s)
            s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            # s: [Batch*Channel, Patch_num, Patch_len]

        # ============== Patch Embedding ==============
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # ============== CNN Depthwise ==============
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # ============== Residual Stream ==============
        res = self.fc2(res)
        s = s + res

        # ============== CNN Pointwise ==============
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)
        # s: [BC, patch_num, d_model]

        # ============== Optional Cross-Patch Attention ==============
        if self.use_cross_attn:
            residual = s
            s_norm = self.attn_norm(s)
            s_attn, _ = self.attn(s_norm, s_norm, s_norm)
            s = residual + self.attn_drop(s_attn)
            if self.attn_ffn is not None:
                s = s + self.attn_ffn(s)

        # ============== Optional RoRA (Geometric Reorientation) ==============
        if self.use_rora:
            s = self.rora(s)

        # ============== Flatten Head ==============
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)  # [BC, P]

        # ---------- Trend Stream (configurable/fallback) ----------
        t = self.trend(t)  # [BC, P]

        # ---------- Streams Concatenation ----------
        x = torch.cat((s, t), dim=1)  # [BC, 2P]
        x = self.fc8(x)               # [BC, P]

        # Channel concatenation
        x = torch.reshape(x, (B, C, self.pred_len))  # [Batch, Channel, Output]
        x = x.permute(0, 2, 1)  # to [Batch, Output, Channel]

        return x
