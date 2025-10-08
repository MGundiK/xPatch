import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- window builders (symmetric, length L) ----------

def _build_window(kind: str, L: int, beta: float = 8.0, a: int = 2, dtype=torch.float32, device=None):
    """
    Return a symmetric window of length L (odd recommended).
    kind: 'hann' | 'kaiser' | 'lanczos' | 'hann_poisson'
    Kaiser:  beta controls sidelobes (8~12 good defaults)
    Lanczos: a controls taper sharpness (2~3 common)
    """
    n = torch.arange(L, device=device, dtype=dtype)

    if kind == "hann":
        w = 0.5 - 0.5*torch.cos(2*math.pi*(n/(L-1)))
    elif kind == "kaiser":
        # Kaiser via I0; for speed, use a short series (OK for beta <= ~12)
        x = (2*n/(L-1) - 1.0)  # [-1,1]
        t = (1 - x**2).clamp_min(0)
        def I0(z):
            y = z/2
            s = torch.ones_like(y)
            term = torch.ones_like(y)
            for k in range(1,8):
                term = term * (y*y)/(k*k)
                s = s + term
            return s
        w = I0(beta*torch.sqrt(t)) / I0(torch.tensor(beta, device=device, dtype=dtype))
    elif kind == "lanczos":
        # sinc(x)*sinc(x/a) centered at (L-1)/2
        x = (n - (L-1)/2) / ((L-1)/2) * a  # scale to [-a, a]
        def sinc(t):
            out = torch.ones_like(t)
            nz = t != 0
            out[nz] = torch.sin(math.pi*t[nz])/(math.pi*t[nz])
            return out
        w = sinc(x) * sinc(x/a)
        # ensure nonnegative (to behave like a smoother kernel)
        w = (w - w.min()).clamp_min(0)
    elif kind == "hann_poisson":
        # Hann multiplied by a symmetric exponential (Poisson) envelope
        hann = 0.5 - 0.5*torch.cos(2*math.pi*(n/(L-1)))
        tau = max(1.0, L/6.0)
        pois = torch.exp(-torch.abs(n - (L-1)/2)/tau)
        w = hann * pois
    else:
        raise ValueError(f"Unknown window kind: {kind}")

    # normalize area = 1 for consistent smoothing strength before halving
    w = w / (w.sum() + 1e-12)
    return w  # [L]

# ---------- causal half-window depthwise smoother ----------

class CausalWindowTrend(nn.Module):
    """
    Causal FIR smoother built from a symmetric window by taking the causal half (center..end).
    Intended as a drop-in replacement for the EMA trend block in xPatch.

    Args
    ----
    kind: str = 'hann' | 'kaiser' | 'lanczos' | 'hann_poisson'
    L: int     full window length (use odd; effective causal kernel length K = (L+1)//2)
    beta: float  Kaiser parameter (only used if kind='kaiser')
    a: int       Lanczos parameter (only used if kind='lanczos')
    per_channel: bool  if True, learns a tiny scalar gain per channel after smoothing (off by default)

    I/O
    ---
    forward(x: [B,T,C]) -> trend: [B,T,C]
    """
    def __init__(self, kind: str = "hann", L: int = 33, beta: float = 8.0, a: int = 2, per_channel: bool = False):
        super().__init__()
        assert L % 2 == 1, "Use an odd L so the center is well-defined"
        self.kind = kind
        self.L = int(L)
        self.beta = float(beta)
        self.a = int(a)
        self.per_channel = per_channel
        self._gain = None  # lazily created if per_channel=True

    @property
    def group_delay(self) -> int:
        """Approximate causal delay in samples (for half-window FIR, ≈ K-1 = (L-1)//2)."""
        return (self.L - 1) // 2

    def _build_half_kernel(self, C: int, dtype, device):
        # symmetric window -> take causal half (center..end), normalize to sum=1
        w_full = _build_window(self.kind, self.L, self.beta, self.a, dtype=dtype, device=device)  # [L]
        mid = (self.L - 1) // 2
        k = w_full[mid:].clone()                                         # [K]
        k = k / (k.sum() + 1e-12)
        k = k.view(1,1,-1).repeat(C,1,1)                                 # [C,1,K] depthwise kernel
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,C]  -> trend: [B,T,C]
        """
        B, T, C = x.shape
        device, dtype = x.device, x.dtype
        k = self._build_half_kernel(C, dtype, device)                    # [C,1,K]
        padL = k.shape[-1] - 1
        y_bct = F.conv1d( F.pad(x.transpose(1,2), (padL, 0), mode="replicate"),
                          k, bias=None, stride=1, padding=0, groups=C )  # [B,C,T]
        y = y_bct.transpose(1,2)                                         # [B,T,C]

        if self.per_channel:
            if self._gain is None:
                # simple affine gain per channel (init 1.0)
                self._gain = nn.Parameter(torch.ones(1,1,C, device=device, dtype=dtype))
            y = y * self._gain
        return y
