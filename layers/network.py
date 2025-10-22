# layers/network.py
# xPatch Network with configurable trend stream (heads in layers/trend_heads.py)
# Falls back to baseline trend head if no/unknown trend_head is provided.

import torch
from torch import nn

from layers.trend_heads import (
    BaselineMLPTrendHead,
    FIRTrendHead,
    BasisTrendHead,
    LocalLinearTrendHead,
    DeltaTrendHead,
    DownsampledMLPTrendHead
)


_TREND_FACTORY = {
    "mlp_baseline": BaselineMLPTrendHead,
    "fir":          FIRTrendHead,
    "basis":        BasisTrendHead,
    "local_lin":    LocalLinearTrendHead,
    "delta":        DeltaTrendHead,
    "ds_mlp":       DownsampledMLPTrendHead,
}


class Network(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        patch_len,
        stride,
        padding_patch,
        trend_head=None,      # may be None or empty -> fallback to baseline
        trend_cfg=None,       # optional dict
    ):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len

        # ---------- Non-linear Stream (seasonality) ----------
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise
        self.conv1 = nn.Conv1d(
            self.patch_num, self.patch_num, patch_len, patch_len, groups=self.patch_num
        )
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # ---------- Trend Stream (configurable with safe fallback) ----------
        if not trend_head or (isinstance(trend_head, str) and trend_head.strip() == ""):
            # No trend head provided -> baseline
            selected_head = "mlp_baseline"
        elif trend_head in _TREND_FACTORY:
            selected_head = trend_head
        else:
            # Unknown head string -> fallback to baseline with a warning
            print(f"[xPatch][Network] Warning: unknown trend_head='{trend_head}'. "
                  f"Falling back to 'mlp_baseline'. Options: {list(_TREND_FACTORY.keys())}")
            selected_head = "mlp_baseline"

        if trend_cfg is None:
            trend_cfg = {}

        if selected_head != (trend_head or "mlp_baseline"):
            trend_cfg = {}

        self.trend = _TREND_FACTORY[selected_head](
            seq_len=seq_len, pred_len=pred_len, **trend_cfg
        )

        # ---------- Streams Concatenation ----------
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
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

        # ---------- Non-linear Stream (seasonality) ----------
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch*Channel, Patch_num, Patch_len]

        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
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
