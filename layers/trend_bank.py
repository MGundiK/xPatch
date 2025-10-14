import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# For torch.i0: use torch.special.i0 if torch.i0 doesn’t exist in your build.
if not hasattr(torch, "i0"):
    torch.i0 = torch.special.i0  # fallback

# --------------------
# Small utilities
# --------------------
def _as_device_dtype(x, v):
    return torch.as_tensor(v, device=x.device, dtype=x.dtype)

def _init_like_first(x):
    y = torch.zeros_like(x)
    y[:, 0, :] = x[:, 0, :]
    return y

def _causal_pad_left(x, k):
    # x: [B,T,C] -> [B,T+pad,C]
    return F.pad(x, (0,0, k-1,0))  # pad only in time dim (left)

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

class CausalFIRWindow(nn.Module):
    """
    Generic causal depthwise FIR with a differentiable window generator g(theta).
    We build kernel w[0..L-1] (causal), normalize sum=1, and apply depthwise conv.
    """
    def __init__(self, channels, L=129, num_kernels=1, learnable_mix=False):
        super().__init__()
        self.C, self.L, self.K = channels, L, num_kernels
        self.learnable_mix = learnable_mix
        if learnable_mix and self.K > 1:
            self.mix_logits = nn.Parameter(torch.zeros(channels, self.K))

    def window(self, theta):  # override
        raise NotImplementedError

    def build_kernel(self, theta, x):
        # theta: dict of parameters (per-channel or per (K,C))
        # return kernel: [C*K, 1, L] suitable for depthwise conv (groups=C*K)
        w = self.window(theta)                                    # [K,C,L] or [1,C,L]
        w = w / (w.sum(dim=-1, keepdim=True).clamp_min(1e-8))     # normalize DC gain
        return w.reshape(self.K*self.C, 1, self.L).to(x.device).to(x.dtype)

    def forward_with_kernel(self, x, K):  # depthwise conv
        # x: [B,T,C] -> [B,T,C] causal FIR (left-pad by L-1)
        B,T,C = x.shape
        xt = x.transpose(1,2)                    # [B,C,T]
        xt = F.pad(xt, (self.L-1, 0))            # causal
        #y = F.conv1d(xt, K, groups=self.K*self.C)  # [B, C*K, T]
        y = F.conv1d(xt, K, groups=self.C)         # ✅ C groups; weight [C*K,1,L]

        if self.K == 1:
            y = y
        elif self.learnable_mix:
            w = F.softmax(self.mix_logits.to(x.device).to(x.dtype), dim=-1)  # [C,K]
            y = (y.view(B, self.C, self.K, T) * w.view(1,self.C,self.K,1)).sum(dim=2)  # [B,C,T]
        else:
            # simple average
            y = y.view(B, self.C, self.K, T).mean(dim=2)
        return y.transpose(1,2)  # [B,T,C]


class KaiserFIR(CausalFIRWindow):
    """
    Learnable-β Kaiser window FIR (causal), optionally multi-kernel.
    """
    def __init__(self, channels, L=129, num_kernels=1, init_beta=6.0, learnable_mix=False):
        super().__init__(channels, L, num_kernels, learnable_mix)
        # β per (K,C)
        self.log_beta = nn.Parameter(torch.log(torch.full((num_kernels, channels), init_beta)))

    def window(self, theta=None):
        # indices 0..L-1 (causal)
        n = torch.arange(self.L, device=self.log_beta.device, dtype=torch.float32)
        # center at (L-1) to make symmetric weights in causal form
        m = (n - (self.L-1)/2) / ((self.L-1)/2 + 1e-8)  # [-1,1]
        beta = torch.exp(self.log_beta)                 # [K,C]
        # I0 approx (Bessel) using torch.i0 if available
        i0_beta = torch.i0(beta)
        arg = beta.unsqueeze(-1) * torch.sqrt(1 - m.pow(2)).clamp_min(0)  # [K,C,L]
        w = torch.i0(arg) / i0_beta.unsqueeze(-1)                          # [K,C,L]
        # zero out future (right) weights to keep causality? We already causal-pad on left
        # The symmetric window is applied as left-only by padding, so it's causal.
        return w

class HannPoissonFIR(CausalFIRWindow):
    """
    Hann window multiplied by decaying Poisson exp(-λ n), learnable λ.
    """
    def __init__(self, channels, L=129, num_kernels=1, init_lambda=0.02, learnable_mix=False):
        super().__init__(channels, L, num_kernels, learnable_mix)
        self.log_lambda = nn.Parameter(torch.log(torch.full((num_kernels, channels), init_lambda)))

    def window(self, theta=None):
        n = torch.arange(self.L, device=self.log_lambda.device, dtype=torch.float32)
        # Hann in [0..L-1] (causalized by left padding)
        hann = 0.5 * (1 - torch.cos(2*torch.pi*(n / (self.L-1)).clamp(0,1)))  # [L]
        lam = torch.exp(self.log_lambda)                                       # [K,C]
        pois = torch.exp(-lam.unsqueeze(-1) * n.view(1,1,self.L))             # [K,C,L]
        w = pois * hann.view(1,1,self.L)
        return w



# ============================================================
# 1) Learnable EMA (α per channel)
# ============================================================

class FastLearnableEMA(nn.Module):
    def __init__(self, channels, init_alpha=0.9, debias=False, clamp=(1e-4, 1-1e-4)):
        super().__init__()
        self.debias = debias
        self.clamp = clamp
        init = torch.logit(torch.tensor(init_alpha).clamp(*clamp))
        self.logit_alpha = nn.Parameter(init.repeat(channels))  # [C]

    def forward(self, x):
        # x: [B,T,C]
        B, T, C = x.shape
        device, dtype = x.device, x.dtype

        a = torch.sigmoid(self.logit_alpha).clamp(*self.clamp).to(device, dtype)  # [C]
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)              # [T,1]

        # a_pow[t,c] = a[c]^t
        a_pow = torch.pow(a.unsqueeze(0), t)                                      # [T,C]
        divisor = a_pow.clamp_min(1e-8)                                           # [T,C]

        # weights[t,c] = (1-a[c]) * a[c]^t  for t>=1, and = a[c]^0 (=1) for t=0
        # build a scale without in-place
        scale = torch.cat([
            torch.ones(1, C, device=device, dtype=dtype),
            (1.0 - a).unsqueeze(0).expand(T-1, C)
        ], dim=0)                                                                 # [T,C]
        weights = a_pow * scale                                                   # [T,C]

        w = weights.view(1, T, C)
        d = divisor.view(1, T, C)

        y = torch.cumsum(x * w, dim=1) / d                                        # [B,T,C]

        if self.debias:
            deb = (1.0 - a_pow).clamp_min(1e-8).view(1, T, C)
            y = y / deb

        return y


# ============================================================
# 2) Multi-α EMA mixture (K time constants + learned mixing)
# ============================================================

class FastMultiEMAMixture(nn.Module):
    """
    K parallel EMAs with per-channel α_k and softmax mixture per channel.
    x: [B,T,C] -> trend: [B,T,C]
    """
    def __init__(self, channels, K=3, init_alphas=(0.8,0.9,0.98), clamp=(1e-4, 1-1e-4)):
        super().__init__()
        self.K = K
        if len(init_alphas) < K:
            import numpy as np
            init_alphas = list(init_alphas) + list(np.linspace(0.7, 0.99, K-len(init_alphas)))
        init = torch.stack([torch.logit(torch.tensor(a).clamp(*clamp)) for a in init_alphas[:K]], dim=0)  # [K]
        self.logit_alpha = nn.Parameter(init.unsqueeze(-1).repeat(1, channels))  # [K,C]
        self.mix_logits  = nn.Parameter(torch.zeros(channels, K))                # [C,K]
        self.clamp = clamp

    def forward(self, x):
        B,T,C = x.shape
        a = torch.sigmoid(self.logit_alpha).clamp(*self.clamp).to(x.device).to(x.dtype)  # [K,C]
        t_idx = torch.arange(T, device=x.device, dtype=x.dtype).view(T,1)                # [T,1]
        a_pow = torch.pow(a.unsqueeze(0), t_idx)                                         # [T,K,C]
        weights = a_pow.clone()
        weights[1:,:,:] = weights[1:,:,:] * (1 - a.unsqueeze(0))                         # [T,K,C]
        divisor = a_pow.clamp_min(1e-8)                                                  # [T,K,C]
        xw = (x.unsqueeze(2) * weights.unsqueeze(0))                                     # [B,T,K,C]
        yk = torch.cumsum(xw, dim=1) / divisor.unsqueeze(0)                              # [B,T,K,C]
        mix = F.softmax(self.mix_logits.to(x.device).to(x.dtype), dim=-1).transpose(0,1) # [K,C]
        trend = (yk * mix.view(1,1,self.K,C)).sum(dim=2)                                 # [B,T,C]
        return trend


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
    2-state causal filter: level + slope, learnable α, β (per channel).
    x: [B,T,C] -> trend(level): [B,T,C]
    """
    def __init__(self, channels, init_alpha=0.5, init_beta=0.1, clamp=(1e-4, 1-1e-4)):
        super().__init__()
        self.logit_a = nn.Parameter(torch.logit(torch.tensor(init_alpha)).repeat(channels))
        self.logit_b = nn.Parameter(torch.logit(torch.tensor(init_beta )).repeat(channels))
        self.clamp   = clamp

    def forward(self, x):
        B,T,C = x.shape
        a = torch.sigmoid(self.logit_a).clamp(*self.clamp).to(x.device).to(x.dtype).view(1,1,C)
        b = torch.sigmoid(self.logit_b).clamp(*self.clamp).to(x.device).to(x.dtype).view(1,1,C)
        L = torch.zeros_like(x); L[:,0,:] = x[:,0,:]  # level
        V = torch.zeros(B,C, device=x.device, dtype=x.dtype)  # slope
        for t in range(1,T):
            pred  = L[:,t-1,:] + V
            resid = x[:,t,:] - pred
            L[:,t,:] = pred + a.squeeze(0).squeeze(0) * resid
            V = V + b.squeeze(0).squeeze(0) * (L[:,t,:] - L[:,t-1,:] - V)
        return L

# Speed tip: compile
try:
    AlphaBetaFilter = torch.compile(AlphaBetaFilter)  # PyTorch 2.3+
except Exception:
    pass


# ============================================================
# 5) EWRLS Level (online RLS with forgetting)
# ============================================================

class FastEWRLSLevel(nn.Module):
    """
    Level-only EW-RLS with forgetting λ (learnable per channel).
    x: [B,T,C] -> trend: [B,T,C]
    """
    def __init__(self, channels, init_lambda=0.98, clamp=(1e-4, 1-1e-4)):
        super().__init__()
        self.logit_lambda = nn.Parameter(torch.logit(torch.tensor(init_lambda)).repeat(channels))
        self.clamp = clamp
        self.init_P = 1.0

    def forward(self, x):
        B,T,C = x.shape
        lam = torch.sigmoid(self.logit_lambda).clamp(*self.clamp).to(x.device).to(x.dtype)  # [C]
        lam = lam.view(1,C)
        theta = x[:,0,:].clone()                         # [B,C]
        P     = torch.full((B,C), self.init_P, device=x.device, dtype=x.dtype)
        out   = torch.zeros_like(x); out[:,0,:] = theta
        for t in range(1,T):
            e  = x[:,t,:] - theta                        # [B,C]
            denom = lam + P                              # [B,C]
            K  = P / denom
            theta = theta + K * e
            P = (P - K * P) / lam
            out[:,t,:] = theta
        return out

try:
    FastEWRLSLevel = torch.compile(FastEWRLSLevel)
except Exception:
    pass


# ============================================================
# 6) Exponentially Weighted Median (robust)
# ============================================================

class HuberEMA(nn.Module):
    """
    Robust EMA: replace residual with Huber loss derivative.
    """
    def __init__(self, channels, init_alpha=0.9, delta=1.0, clamp=(1e-4,1-1e-4)):
        super().__init__()
        self.logit_alpha = nn.Parameter(torch.logit(torch.tensor(init_alpha)).repeat(channels))
        self.delta = nn.Parameter(torch.tensor(delta), requires_grad=False)
        self.clamp = clamp

    def forward(self, x):
        B,T,C = x.shape
        a = torch.sigmoid(self.logit_alpha).clamp(*self.clamp).to(x.device).to(x.dtype).view(1,1,C)
        y = torch.zeros_like(x); y[:,0,:] = x[:,0,:]
        d = self.delta.to(x.device).to(x.dtype)
        for t in range(1,T):
            r = x[:,t,:] - y[:,t-1,:]
            g = torch.where(r.abs() <= d, r, d * r.sign())  # Huber grad
            y[:,t,:] = y[:,t-1,:] + (1 - a) * g
        return y

try:
    HuberEMA = torch.compile(HuberEMA)
except Exception:
    pass


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

# If you didn’t split them yet, you can keep them in one file and import here accordingly.


def build_trend_module(name: str, channels: int, **kw) -> nn.Module:
    """
    name ∈ {
      'fast_ema', 'alpha_beta',
      'kaiser_fir', 'hann_poisson_fir',
      'ewrls_fast', 'huber_ema'
    }
    """
    n = name.lower()

    if n == "fast_ema":
        # kwargs: init_alpha=0.9, debias=False
        return FastLearnableEMA(
            channels=channels,
            init_alpha=kw.get("fastema_init_alpha", 0.9),
            debias=kw.get("fastema_debias", False),
        )

    if n == "alpha_beta":
        # kwargs: init_alpha=0.5, init_beta=0.1
        mod = AlphaBetaFilter(
            channels=channels,
            init_alpha=kw.get("ab_init_alpha", 0.5),
            init_beta=kw.get("ab_init_beta", 0.1),
        )
        try:
            mod = torch.compile(mod)
        except Exception:
            pass
        return mod

    if n == "kaiser_fir":
        # kwargs: L=129, num_kernels=1, init_beta=6.0, learnable_mix=False
        return KaiserFIR(
            channels=channels,
            L=kw.get("kaiser_L", 129),
            num_kernels=kw.get("kaiser_num_kernels", 1),
            init_beta=kw.get("kaiser_init_beta", 6.0),
            learnable_mix=kw.get("kaiser_learnable_mix", False),
        )

    if n == "hann_poisson_fir":
        # kwargs: L=129, num_kernels=1, init_lambda=0.02, learnable_mix=False
        return HannPoissonFIR(
            channels=channels,
            L=kw.get("hannp_L", 129),
            num_kernels=kw.get("hannp_num_kernels", 1),
            init_lambda=kw.get("hannp_init_lambda", 0.02),
            learnable_mix=kw.get("hannp_learnable_mix", False),
        )

    if n == "ewrls_fast":
        # kwargs: init_lambda=0.98
        mod = FastEWRLSLevel(
            channels=channels,
            init_lambda=kw.get("ewrls_init_lambda", 0.98),
        )
        try:
            mod = torch.compile(mod)
        except Exception:
            pass
        return mod

    if n == "huber_ema":
        # kwargs: init_alpha=0.9, delta=1.0
        mod = HuberEMA(
            channels=channels,
            init_alpha=kw.get("huber_init_alpha", 0.9),
            delta=kw.get("huber_delta", 1.0),
        )
        try:
            mod = torch.compile(mod)
        except Exception:
            pass
        return mod

    raise ValueError(f"Unknown trend module: {name}")
