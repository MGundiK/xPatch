import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# Small utilities
# --------------------
def _as_device_dtype(x, v):
    return torch.as_tensor(v, device=x.device, dtype=x.dtype)

def _init_like_first(x):
    y = torch.zeros_like(x)
    y[:, 0, :] = x[:, 0, :]
    return y

# ============================================================
# 0) Causal Half-Window FIR (Hann / Kaiser / Lanczos / Hann-Poisson)
# ============================================================

def _build_window(kind: str, L: int, beta: float = 8.0, a: int = 2,
                  dtype=torch.float32, device=None):
    n = torch.arange(L, device=device, dtype=dtype)
    if kind == "hann":
        w = 0.5 - 0.5*torch.cos(2*math.pi*(n/(L-1)))
    elif kind == "kaiser":
        x = (2*n/(L-1) - 1.0)
        t = (1 - x**2).clamp_min(0)
        def I0(z):
            y = z/2
            s = torch.ones_like(y)
            term = torch.ones_like(y)
            for k in range(1, 8):
                term = term * (y*y)/(k*k)
                s = s + term
            return s
        w = I0(beta*torch.sqrt(t)) / I0(torch.tensor(beta, device=device, dtype=dtype))
    elif kind == "lanczos":
        x = (n - (L-1)/2) / ((L-1)/2) * a
        def sinc(t):
            out = torch.ones_like(t)
            nz = t != 0
            out[nz] = torch.sin(math.pi*t[nz])/(math.pi*t[nz])
            return out
        w = sinc(x)*sinc(x/a)
        w = (w - w.min()).clamp_min(0)  # keep it smoothing
    elif kind == "hann_poisson":
        hann = 0.5 - 0.5*torch.cos(2*math.pi*(n/(L-1)))
        tau = max(1.0, L/6.0)
        pois = torch.exp(-torch.abs(n - (L-1)/2)/tau)
        w = hann * pois
    else:
        raise ValueError(f"Unknown window kind: {kind}")
    w = w / (w.sum() + 1e-12)
    return w

class CausalWindowTrend(nn.Module):
    """
    Causal FIR smoother from a symmetric window by taking the causal half (center..end).
    Forward: x[B,T,C] -> trend[B,T,C]
    """
    def __init__(self, kind="hann", L=33, beta=9.0, a=2, per_channel=False):
        super().__init__()
        assert L % 2 == 1, "Use odd L"
        self.kind, self.L, self.beta, self.a = kind, int(L), float(beta), int(a)
        self.per_channel = per_channel
        self._gain = None  # optional per-channel affine gain

    @property
    def group_delay(self):
        # causal half-window delay ≈ (L-1)//2
        return (self.L - 1)//2

    def _half_kernel(self, C, dtype, device):
        w_full = _build_window(self.kind, self.L, self.beta, self.a, dtype=dtype, device=device)
        mid = (self.L - 1)//2
        k = w_full[mid:].clone()
        k = k / (k.sum() + 1e-12)
        return k.view(1,1,-1).repeat(C,1,1)  # [C,1,K]

    def forward(self, x):
        B,T,C = x.shape
        k = self._half_kernel(C, x.dtype, x.device)
        padL = k.shape[-1] - 1
        y = F.conv1d(F.pad(x.transpose(1,2), (padL,0), mode="replicate"),
                     k, groups=C).transpose(1,2)
        if self.per_channel:
            if self._gain is None:
                self._gain = nn.Parameter(torch.ones(1,1,C, device=x.device, dtype=x.dtype))
            y = y * self._gain
        return y

# ============================================================
# 1) Learnable EMA (α per channel)
# ============================================================

class LearnableEMA(nn.Module):
    """
    y_t = α ⊙ y_{t-1} + (1-α) ⊙ x_t, α ∈ (0,1) per channel.
    """
    def __init__(self, channels, init_alpha=0.9, clamp=(1e-4, 1-1e-4)):
        super().__init__()
        init = torch.logit(torch.tensor(init_alpha).clamp(*clamp))
        self.logit_alpha = nn.Parameter(init.repeat(channels))
        self.clamp = clamp

    def forward(self, x):
        B,T,C = x.shape
        a = torch.sigmoid(self.logit_alpha).clamp(*self.clamp).to(x.device, x.dtype).view(1,1,C)
        y = _init_like_first(x)
        for t in range(1, T):
            y[:, t, :] = a * y[:, t-1, :] + (1 - a) * x[:, t, :]
        return y

# ============================================================
# 2) Multi-α EMA mixture (K time constants + learned mixing)
# ============================================================

class MultiEMAMixture(nn.Module):
    """
    K parallel EMAs with per-channel α_k and learned convex mixing per channel.
    """
    def __init__(self, channels, K=3, init_alphas=(0.8, 0.9, 0.98), clamp=(1e-4, 1-1e-4)):
        super().__init__()
        import numpy as np
        self.K = int(K)
        if len(init_alphas) < self.K:
            extra = np.linspace(0.7, 0.99, self.K - len(init_alphas))
            init_alphas = list(init_alphas) + list(extra)
        init_a = torch.stack([torch.logit(torch.tensor(a).clamp(*clamp)) for a in init_alphas[:self.K]], dim=0)
        self.logit_alpha = nn.Parameter(init_a.unsqueeze(-1).repeat(1, channels))  # [K,C]
        self.mix_logits  = nn.Parameter(torch.zeros(channels, self.K))             # [C,K]
        self.clamp = clamp

    def forward(self, x):
        B,T,C = x.shape
        a = torch.sigmoid(self.logit_alpha).clamp(*self.clamp).to(x.device, x.dtype)  # [K,C]
        w = F.softmax(self.mix_logits.to(x.device, x.dtype), dim=-1).transpose(0,1)   # [K,C]
        # states per (k,c): initialize at x0
        yk = x[:, 0, :].unsqueeze(1).repeat(1, self.K, 1)  # [B,K,C]
        out = torch.zeros_like(x)
        out[:, 0, :] = (w.unsqueeze(0) * yk).sum(dim=1)
        for t in range(1, T):
            xt = x[:, t, :].unsqueeze(1)                          # [B,1,C]
            yk = a.unsqueeze(0) * yk + (1 - a.unsqueeze(0)) * xt  # [B,K,C]
            out[:, t, :] = (w.unsqueeze(0) * yk).sum(dim=1)
        return out

# ============================================================
# 3) Debiased EMA (start-up bias correction)
# ============================================================

class DebiasedEMA(nn.Module):
    """
    \hat{y}_t = y_t / (1 - α^t). α can be fixed or learnable per channel.
    """
    def __init__(self, channels, alpha=0.9, learnable=False, clamp=(1e-4, 1-1e-4)):
        super().__init__()
        self.learnable = learnable
        self.clamp = clamp
        if learnable:
            init = torch.logit(torch.tensor(alpha).clamp(*clamp))
            self.logit_alpha = nn.Parameter(init.repeat(channels))
        else:
            self.register_buffer("alpha_fixed", torch.tensor(alpha))

    def forward(self, x):
        B,T,C = x.shape
        if self.learnable:
            a = torch.sigmoid(self.logit_alpha).clamp(*self.clamp).to(x.device, x.dtype).view(1,1,C)
        else:
            a = _as_device_dtype(x, self.alpha_fixed).clamp(*self.clamp).view(1,1,1).expand(1,1,C)
        y = _init_like_first(x)
        for t in range(1, T):
            y[:, t, :] = a * y[:, t-1, :] + (1 - a) * x[:, t, :]
        # bias correction
        t_idx = torch.arange(1, T+1, device=x.device, dtype=x.dtype).view(T, 1)  # [T,1]
        aC = a.view(1, C).expand(T, C)
        denom = (1 - torch.pow(aC, t_idx)).clamp_min(1e-6).unsqueeze(0)         # [1,T,C]
        return y / denom

# ============================================================
# 4) Alpha-Beta Filter (level + slope)
# ============================================================

class AlphaBetaFilter(nn.Module):
    """
    Level-slope filter (Holt). α,β learnable per channel. Output is level.
    """
    def __init__(self, channels, init_alpha=0.5, init_beta=0.1, clamp=(1e-4, 1-1e-4)):
        super().__init__()
        self.logit_alpha = nn.Parameter(torch.logit(torch.tensor(init_alpha)).repeat(channels))
        self.logit_beta  = nn.Parameter(torch.logit(torch.tensor(init_beta)).repeat(channels))
        self.clamp = clamp

    def forward(self, x):
        B,T,C = x.shape
        a = torch.sigmoid(self.logit_alpha).clamp(*self.clamp).to(x.device, x.dtype).view(1,1,C)
        b = torch.sigmoid(self.logit_beta ).clamp(*self.clamp).to(x.device, x.dtype).view(1,1,C)
        L = _init_like_first(x)
        slope = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        out = torch.zeros_like(x)
        out[:, 0, :] = L[:, 0, :]
        for t in range(1, T):
            pred  = L[:, t-1, :] + slope
            resid = x[:, t, :] - pred
            L[:, t, :] = pred + a.squeeze(0).squeeze(0) * resid
            slope = slope + b.squeeze(0).squeeze(0) * (L[:, t, :] - L[:, t-1, :] - slope)
            out[:, t, :] = L[:, t, :]
        return out

# ============================================================
# 5) EWRLS Level (online RLS with forgetting)
# ============================================================

class EWRLSLevel(nn.Module):
    """
    Online RLS of a constant level with forgetting λ (learnable optional).
    """
    def __init__(self, channels, init_lambda=0.98, learnable=True, clamp=(1e-4, 1-1e-4), init_P=1.0):
        super().__init__()
        self.learnable = learnable
        self.clamp = clamp
        self.init_P = float(init_P)
        if learnable:
            self.logit_lambda = nn.Parameter(torch.logit(torch.tensor(init_lambda)).repeat(channels))
        else:
            self.register_buffer("lambda_fixed", torch.tensor(init_lambda))

    def forward(self, x):
        B,T,C = x.shape
        if self.learnable:
            lam = torch.sigmoid(self.logit_lambda).clamp(*self.clamp).to(x.device, x.dtype).view(1,1,C)
        else:
            lam = _as_device_dtype(x, self.lambda_fixed).clamp(*self.clamp).view(1,1,1).expand(1,1,C)
        theta = x[:, 0, :].clone()                              # [B,C]
        P = torch.full((B, C), self.init_P, device=x.device, dtype=x.dtype)
        y = _init_like_first(x)
        for t in range(1, T):
            e = x[:, t, :] - theta
            denom = lam.squeeze(0).squeeze(0) + P
            K = P / denom
            theta = theta + K * e
            P = (P - K * P) / lam.squeeze(0).squeeze(0)
            y[:, t, :] = theta
        return y

# ============================================================
# 6) Exponentially Weighted Median (robust)
# ============================================================

class EWMedian(nn.Module):
    """
    Robust EW-median via smooth quantile descent (τ=0.5).
    """
    def __init__(self, channels, step=0.05, tau_temp=0.01, learnable_step=False):
        super().__init__()
        self.tau = 0.5
        self.learnable_step = learnable_step
        if learnable_step:
            self.logit_step = nn.Parameter(torch.logit(torch.tensor(step)))
        else:
            self.register_buffer("step_fixed", torch.tensor(step))
        self.register_buffer("tau_temp", torch.tensor(tau_temp))

    def forward(self, x):
        B,T,C = x.shape
        eta = (torch.sigmoid(self.logit_step) if self.learnable_step else _as_device_dtype(x, self.step_fixed)).view(1,1,1)
        temp = _as_device_dtype(x, self.tau_temp).view(1,1,1) + 1e-8
        m = _init_like_first(x)
        for t in range(1, T):
            r = m[:, t-1, :] - x[:, t, :]
            s = torch.sigmoid(r / temp)        # ~I[x_t <= m_{t-1}]
            grad = (self.tau - s)
            m[:, t, :] = m[:, t-1, :] + eta * grad
        return m

# ============================================================
# 7) Alpha Cutoff Filter (first-order IIR from cutoff)
# ============================================================

class AlphaCutoffFilter(nn.Module):
    """
    y_t = α y_{t-1} + (1-α) x_t, with α from (f_c, f_s):
      α = 1 - exp(-2π f_c / f_s)
    """
    def __init__(self, channels, fs=1.0, init_fc=0.05, learnable_fc=True, fc_bounds=(1e-4, 0.5)):
        super().__init__()
        self.fs = float(fs)
        self.fc_bounds = fc_bounds
        self.learnable_fc = learnable_fc
        if learnable_fc:
            self.log_fc = nn.Parameter(torch.log(torch.tensor(init_fc)).repeat(channels))
        else:
            self.register_buffer("fc_fixed", torch.tensor(init_fc))

    def forward(self, x):
        B,T,C = x.shape
        if self.learnable_fc:
            fc = torch.exp(self.log_fc).clamp(*self.fc_bounds).to(x.device, x.dtype).view(1,1,C)
        else:
            fc = _as_device_dtype(x, self.fc_fixed).clamp(*self.fc_bounds).view(1,1,1).expand(1,1,C)
        fs = torch.tensor(self.fs, device=x.device, dtype=x.dtype).view(1,1,1)
        alpha = 1 - torch.exp(-2 * torch.pi * fc / fs)  # (0,1)
        y = _init_like_first(x)
        for t in range(1, T):
            y[:, t, :] = alpha * y[:, t-1, :] + (1 - alpha) * x[:, t, :]
        return y

# ============================================================
# 8) One-Euro & Adaptive Gaussian (optional imports)
# ============================================================

class OneEuro(nn.Module):
    def __init__(self, channels=None, min_cutoff=1.0, beta=0.007, dcutoff=1.0, fs=1.0):
        super().__init__()
        self.min_cutoff, self.beta, self.dcutoff, self.fs = float(min_cutoff), float(beta), float(dcutoff), float(fs)
    def _alpha(self, cutoff, fs, dtype, device):
        tau = 1.0 / (2*math.pi*cutoff)
        te  = 1.0 / fs
        return 1.0 / (1.0 + tau/te)
    def forward(self, x):
        B,T,C = x.shape
        x_prev = x[:,0,:]; dx_prev = torch.zeros_like(x_prev)
        y_prev = x_prev
        y = [y_prev.unsqueeze(1)]
        fs = torch.tensor(self.fs, device=x.device, dtype=x.dtype)
        for t in range(1, T):
            dx = (x[:,t,:] - x[:,t-1,:]) * fs
            a_d = self._alpha(self.dcutoff, fs, x.dtype, x.device)
            dx_hat = a_d*dx + (1-a_d)*dx_prev
            cutoff = self.min_cutoff + self.beta*dx_hat.abs()
            a_x = self._alpha(cutoff, fs, x.dtype, x.device)
            y_t = a_x*x[:,t,:] + (1-a_x)*y_prev
            y.append(y_t.unsqueeze(1))
            dx_prev, y_prev = dx_hat, y_t
        return torch.cat(y, dim=1)

# (If you already have AdaptiveGaussianTrendCausal in your codebase, reuse it.)
# Otherwise you can plug the version we built earlier.

# ============================================================
# Factory
# ============================================================

def build_trend_module(name: str, channels: int, **kwargs) -> nn.Module:
    """
    name ∈ {
      'learnable_ema', 'multi_ema', 'debiased_ema', 'alpha_beta',
      'ewrls', 'ew_median', 'alpha_cutoff',
      'window_hann', 'window_kaiser', 'window_lanczos', 'window_hann_poisson',
      'one_euro'
    }
    kwargs: pass-through to constructors (see classes above).
    """
    name = name.lower()
    if name == "learnable_ema":
        return LearnableEMA(channels, **kwargs)
    if name == "multi_ema":
        return MultiEMAMixture(channels, **kwargs)
    if name == "debiased_ema":
        return DebiasedEMA(channels, **kwargs)
    if name == "alpha_beta":
        return AlphaBetaFilter(channels, **kwargs)
    if name == "ewrls":
        return EWRLSLevel(channels, **kwargs)
    if name == "ew_median":
        return EWMedian(channels, **kwargs)
    if name == "alpha_cutoff":
        return AlphaCutoffFilter(channels, **kwargs)
    if name == "window_hann":
        return CausalWindowTrend(kind="hann", **kwargs)
    if name == "window_kaiser":
        return CausalWindowTrend(kind="kaiser", **kwargs)
    if name == "window_lanczos":
        return CausalWindowTrend(kind="lanczos", **kwargs)
    if name == "window_hann_poisson":
        return CausalWindowTrend(kind="hann_poisson", **kwargs)
    if name == "one_euro":
        # channels is ignored but kept for a uniform signature
        return OneEuro(**kwargs)
    raise ValueError(f"Unknown trend module: {name}")
