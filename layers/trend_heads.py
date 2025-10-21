# layers/trend_heads.py
# Pluggable trend heads for xPatch.
# All heads map [B*C, seq_len] -> [B*C, pred_len].

import math
import torch
from torch import nn


# ---------------- Base ----------------
class TrendHeadBase(nn.Module):
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------- 1) Baseline (original xPatch linear stream) ----------------
class BaselineMLPTrendHead(TrendHeadBase):
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__(seq_len, pred_len)
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.fc5(t)                             # [BC, 4P]
        x = self.avgpool1(x.unsqueeze(1)).squeeze(1)  # [BC, 2P]
        x = self.ln1(x)                             # [BC, 2P]
        x = self.fc6(x)                             # [BC, P]
        x = self.avgpool2(x.unsqueeze(1)).squeeze(1)  # [BC, P//2]
        x = self.ln2(x)                             # [BC, P//2]
        x = self.fc7(x)                             # [BC, P]
        return x


# ---------------- 2) FIRTrendHead: multi-rate causal conv (learnable low-pass) ----------------
class FIRTrendHead(TrendHeadBase):
    def __init__(self, seq_len: int, pred_len: int,
                 k_list=None, d_list=None, channels: int = 16,
                 gelu: bool = True, aa_pool: bool = True, smooth_l2: float = 1e-5):
        super().__init__(seq_len, pred_len)

        if k_list is None: k_list = [32, 64]
        if d_list is None: d_list = [1, 4]
        assert len(k_list) == len(d_list)

        self.aa_pool = aa_pool

        layers = []
        in_ch = 1
        ch = int(channels)
        self._convs = []

        for i, (k, d) in enumerate(zip(k_list, d_list)):
            k = int(k); d = int(d)
            pad = (k - 1) * d  # causal padding
            conv = nn.Conv1d(in_ch if i == 0 else ch, ch, kernel_size=k, dilation=d, padding=pad)
            layers.append(conv)
            self._convs.append(conv)
            if gelu:
                layers.append(nn.GELU())

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(pred_len) if aa_pool else None
        if not aa_pool:
            self.to_feat = nn.Linear(seq_len, pred_len)
        self.refine = nn.Conv1d(ch, 1, kernel_size=1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t.unsqueeze(1)       # [BC, 1, L]
        x = self.conv(x)         # [BC, ch, L]
        if self.pool is not None:
            x = self.pool(x)     # [BC, ch, P]
            x = self.refine(x).squeeze(1)  # [BC, P]
            return x
        else:
            x = x.flatten(-2)    # [BC, ch*L]
            return self.to_feat(x)  # [BC, P]


# ---------------- 3) BasisTrendHead: polynomial + ultra-low-frequency Fourier ----------------
class BasisTrendHead(TrendHeadBase):
    def __init__(self, seq_len: int, pred_len: int,
                 poly_degree: int = 2, fourier_k: int = 4,
                 normalize_t: bool = True, l2_curv: float = 0.0):
        super().__init__(seq_len, pred_len)

        t_pred = torch.linspace(0, 1, pred_len) if normalize_t else torch.arange(pred_len, dtype=torch.float32)
        Phi = []
        for d in range(int(poly_degree) + 1):
            Phi.append(t_pred ** d)
        for k in range(1, int(fourier_k) + 1):
            Phi.append(torch.sin(2 * math.pi * k * t_pred))
            Phi.append(torch.cos(2 * math.pi * k * t_pred))
        Phi_pred = torch.stack(Phi, dim=0)  # [nbasis, P]
        self.nbasis = int(Phi_pred.shape[0])
        self.register_buffer("Phi_pred", Phi_pred, persistent=False)

        self.to_coeff = nn.Linear(seq_len, self.nbasis)
        self.residual = nn.Sequential(nn.LayerNorm(self.nbasis), nn.Linear(self.nbasis, pred_len))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        c = self.to_coeff(t)           # [BC, nbasis]
        y = c @ self.Phi_pred          # [BC, P]
        y = y + self.residual(c)       # [BC, P]
        return y

# LocalLinearTrendHead (closed-form slope, zero trainable params)
# fit a line on the last K points of the EMA trend (per channel, per batch) 
# and extrapolate it. This is the classic “local level + slope” extrapolator; 
# very robust when the input still contains seasonality leakage.
# the slope over the last K points ignores most of that and extrapolates the slower drift.
class LocalLinearTrendHead(nn.Module):
    """
    Closed-form local linear extrapolation on the last K points.
    IO: [BC, seq_len] -> [BC, pred_len]
    """
    def __init__(self, seq_len:int, pred_len:int, k:int=32):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = max(2, int(k))
        # Precompute normal equations for x = 0..k-1
        t = torch.arange(self.k, dtype=torch.float32)          # [k]
        X = torch.stack([torch.ones_like(t), t], dim=1)        # [k,2]
        XtX_inv = torch.inverse(X.t() @ X)                     # [2,2]
        self.register_buffer('W_beta', XtX_inv @ X.t(), persistent=False)  # [2,k]
        t_pred = torch.arange(self.k, self.k + pred_len, dtype=torch.float32)  # future steps
        Phi = torch.stack([torch.ones_like(t_pred), t_pred], dim=0)            # [2,P]
        self.register_buffer('Phi_pred', Phi, persistent=False)                # [2,P]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        xk = t[:, -self.k:]             # [BC,k]
        beta = xk @ self.W_beta.t()     # [BC,2]  (intercept, slope)
        y = beta @ self.Phi_pred        # [BC,P]
        return y

# ------------ DeltaTrendHead (predict only a small correction on top of a naive) ---------- #
# compute a naive trend forecast from t (repeat-last or short linear), and the head predicts Δ only:
# y_trend = naive(t)+fθ​(t)
class DeltaTrendHead(nn.Module):
    """
    Predict a small correction on top of a naive trend forecast.
    mode: 'last' or 'lin'; k used for 'lin'
    """
    def __init__(self, seq_len:int, pred_len:int, mode:str='last', k:int=16, hidden:int=128):
        super().__init__()
        self.seq_len, self.pred_len, self.mode, self.k = seq_len, pred_len, mode, max(2,int(k))
        self.corr = nn.Sequential(nn.Linear(seq_len, hidden), nn.GELU(), nn.Linear(hidden, pred_len))
        if self.mode == 'lin':
            t = torch.arange(self.k, dtype=torch.float32)
            X = torch.stack([torch.ones_like(t), t], dim=1)
            XtX_inv = torch.inverse(X.t() @ X)
            self.register_buffer('W_beta', XtX_inv @ X.t(), persistent=False)
            t_pred = torch.arange(self.k, self.k + pred_len, dtype=torch.float32)
            self.register_buffer('Phi_pred', torch.stack([torch.ones_like(t_pred), t_pred], dim=0), persistent=False)

    def _naive(self, t: torch.Tensor) -> torch.Tensor:
        if self.mode == 'last':
            return t[:, -1:].repeat(1, self.pred_len)
        # linear
        xk = t[:, -self.k:]
        beta = xk @ self.W_beta.t()
        return beta @ self.Phi_pred

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        base = self._naive(t)
        delta = self.corr(t)
        return base + delta


# ------- DownsampledMLPTrendHead (anti-alias → tiny MLP → upsample) ------------- #
class DownsampledMLPTrendHead(nn.Module):
    """
    Anti-aliased downsample -> small MLP -> linear upsample to pred_len.
    """
    def __init__(self, seq_len:int, pred_len:int, stride:int=4, hidden:int=128):
        super().__init__()
        self.seq_len, self.pred_len, self.stride = seq_len, pred_len, int(stride)
        self.pool = nn.AvgPool1d(kernel_size=self.stride, stride=self.stride, ceil_mode=False)
        Lc = math.floor(seq_len / self.stride)
        self.mlp = nn.Sequential(
            nn.Linear(Lc, hidden),
            nn.GELU(),
            nn.Linear(hidden, pred_len)
        )
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.pool(t.unsqueeze(1)).squeeze(1)  # [BC, Lc]
        return self.mlp(x)                         # [BC, P]


