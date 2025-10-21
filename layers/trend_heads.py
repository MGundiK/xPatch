# layers/trend_heads.py
# Pluggable trend heads for xPatch.
# All heads map [B*C, seq_len] -> [B*C, pred_len] and expose .regularization().

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
        """
        Args:
            t: Tensor of shape [B*C, seq_len]
        Returns:
            Tensor of shape [B*C, pred_len]
        """
        raise NotImplementedError

    def regularization(self) -> torch.Tensor:
        """Optional extra loss term (e.g., smoothness)."""
        device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else t.device
        return torch.tensor(0.0, device=device)


# ---------------- 1) Baseline (original xPatch linear stream) ----------------
class BaselineMLPTrendHead(TrendHeadBase):
    """
    Faithful reimplementation of xPatch "Linear Stream":
      fc5 -> AvgPool1d(k=2) -> LN(pred_len*2)
      -> fc6 -> AvgPool1d(k=2) -> LN(pred_len//2)
      -> fc7
    """
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
        # t: [BC, seq_len]
        x = self.fc5(t)                        # [BC, 4P]
        x = self.avgpool1(x.unsqueeze(1))      # [BC, 1, 2P]
        x = x.squeeze(1)                       # [BC, 2P]
        x = self.ln1(x)                        # [BC, 2P]

        x = self.fc6(x)                        # [BC, P]
        x = self.avgpool2(x.unsqueeze(1))      # [BC, 1, P//2]
        x = x.squeeze(1)                       # [BC, P//2]
        x = self.ln2(x)                        # [BC, P//2]

        x = self.fc7(x)                        # [BC, P]
        return x


# ---------------- 2) FIRTrendHead: multi-rate causal conv (learnable low-pass) ----------------
class FIRTrendHead(TrendHeadBase):
    """
    Learn a small bank of long, causal FIR filters across time; adapt length to pred_len.

    Config kwargs:
        k_list     : list of kernel sizes (e.g., [32, 64])
        d_list     : list of dilations (e.g., [1, 4]) -- same length as k_list
        channels   : internal conv channels (e.g., 8 or 16)
        gelu       : use GELU between convs (default True)
        aa_pool    : use AdaptiveAvgPool1d(pred_len) to adjust length (default True)
        smooth_l2  : L2 penalty on first differences of conv taps (default 1e-5)
    """
    def __init__(self, seq_len: int, pred_len: int,
                 k_list=None, d_list=None, channels: int = 16,
                 gelu: bool = True, aa_pool: bool = True, smooth_l2: float = 1e-5):
        super().__init__(seq_len, pred_len)

        if k_list is None:
            k_list = [32, 64]
        if d_list is None:
            d_list = [1, 4]
        assert len(k_list) == len(d_list), "k_list and d_list must have the same length."

        self.gelu = gelu
        self.aa_pool = aa_pool
        self.smooth_l2 = float(smooth_l2)

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
            if self.gelu:
                layers.append(nn.GELU())

        self.conv = nn.Sequential(*layers)

        # Time length adapter
        self.pool = nn.AdaptiveAvgPool1d(pred_len) if self.aa_pool else None
        if not self.aa_pool:
            self.to_feat = nn.Linear(seq_len, pred_len)

        # final refinement along "channel" dim
        self.refine = nn.Conv1d(ch, 1, kernel_size=1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t.unsqueeze(1)               # [BC, 1, L]
        x = self.conv(x)                 # [BC, ch, L]
        if self.pool is not None:
            x = self.pool(x)             # [BC, ch, P]
            x = self.refine(x).squeeze(1)  # [BC, P]
            return x
        else:
            # fallback: flatten and project to horizon
            x = x.flatten(-2)            # [BC, ch*L]
            return self.to_feat(x)       # [BC, P]

    def regularization(self) -> torch.Tensor:
        if self.smooth_l2 <= 0:
            # create a device-consistent 0
            for p in self.parameters():
                return torch.zeros((), device=p.device)
            return torch.zeros(())
        reg = 0.0
        for conv in self._convs:
            w = conv.weight  # [out_ch, in_ch, K]
            if w.shape[-1] > 1:
                diff = w[..., 1:] - w[..., :-1]
                reg = reg + diff.pow(2).mean()
        # ensure tensor on correct device
        device = self._convs[0].weight.device if self._convs else torch.device("cpu")
        return torch.as_tensor(reg * self.smooth_l2, device=device)


# ---------------- 3) BasisTrendHead: polynomial + ultra-low-frequency Fourier ----------------
class BasisTrendHead(TrendHeadBase):
    """
    Predict coefficients for a small set of trend bases and synthesize the horizon.

    Config kwargs:
        poly_degree : int (default 2)
        fourier_k   : int (default 4)  -> very low-frequency terms
        normalize_t : bool (default True) use t in [0,1]
        l2_curv     : float (default 1e-5) curvature penalty on basis curvature
    """
    def __init__(self, seq_len: int, pred_len: int,
                 poly_degree: int = 2, fourier_k: int = 4,
                 normalize_t: bool = True, l2_curv: float = 1e-5):
        super().__init__(seq_len, pred_len)
        self.l2_curv = float(l2_curv)

        # Build decoder basis Phi_pred: [nbasis, pred_len]
        if normalize_t:
            t_pred = torch.linspace(0, 1, pred_len)
        else:
            t_pred = torch.arange(pred_len, dtype=torch.float32)

        Phi_list = []
        # Polynomial
        for d in range(int(poly_degree) + 1):
            Phi_list.append(t_pred ** d)
        # Very low-frequency Fourier
        for k in range(1, int(fourier_k) + 1):
            Phi_list.append(torch.sin(2 * math.pi * k * t_pred))
            Phi_list.append(torch.cos(2 * math.pi * k * t_pred))

        Phi_pred = torch.stack(Phi_list, dim=0)  # [nbasis, P]
        self.nbasis = int(Phi_pred.shape[0])
        self.register_buffer("Phi_pred", Phi_pred, persistent=False)

        # Coefficient regressor
        self.to_coeff = nn.Linear(seq_len, self.nbasis)

        # Tiny residual to catch leftovers
        self.residual = nn.Sequential(
            nn.LayerNorm(self.nbasis),
            nn.Linear(self.nbasis, pred_len)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        coeff = self.to_coeff(t)              # [BC, nbasis]
        y = coeff @ self.Phi_pred             # [BC, P]
        y = y + self.residual(coeff)          # [BC, P]
        return y

    def regularization(self) -> torch.Tensor:
        if self.l2_curv <= 0:
            # produce a device-correct zero
            for p in self.parameters():
                return torch.zeros((), device=p.device)
            return torch.zeros(())
        P = self.pred_len
        device = self.Phi_pred.device
        D2 = torch.zeros(P - 2, P, device=device)
        rows = torch.arange(P - 2, device=device)
        D2[rows, rows] = 1.0
        D2[rows, rows + 1] = -2.0
        D2[rows, rows + 2] = 1.0
        # curvature energy of the basis (proxy for output smoothness)
        curv = (D2 @ self.Phi_pred.T).pow(2).mean()
        return curv * self.l2_curv
