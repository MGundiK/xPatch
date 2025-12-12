"""
Trend Extraction Module Bank

Collection of learnable and fixed trend extraction methods:
- Learnable EMA variants
- FIR window filters (Kaiser, Hann-Poisson)
- Alpha-Beta filter
- EWRLS
- Huber-robust EMA
- One-Euro filter

All FIR-based methods now handle short sequences by dynamically 
clamping kernel size to the input sequence length.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# For torch.i0: use torch.special.i0 if torch.i0 doesn't exist in your build.
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
    return F.pad(x, (0, 0, k - 1, 0))  # pad only in time dim (left)


# ============================================================
# 0) Causal Half-Window FIR (Hann / Kaiser / Lanczos / Hann-Poisson)
# ============================================================

def _build_window(kind: str, L: int, beta: float = 8.0, a: int = 2,
                  dtype=torch.float32, device=None):
    """Build a symmetric window of given kind and length."""
    n = torch.arange(L, device=device, dtype=dtype)
    
    if kind == "hann":
        w = 0.5 - 0.5 * torch.cos(2 * math.pi * (n / (L - 1)))
    elif kind == "kaiser":
        x = (2 * n / (L - 1) - 1.0)
        t = (1 - x ** 2).clamp_min(0)
        
        def I0(z):
            y = z / 2
            s = torch.ones_like(y)
            term = torch.ones_like(y)
            for k in range(1, 8):
                term = term * (y * y) / (k * k)
                s = s + term
            return s
        
        w = I0(beta * torch.sqrt(t)) / I0(torch.tensor(beta, device=device, dtype=dtype))
    elif kind == "lanczos":
        x = (n - (L - 1) / 2) / ((L - 1) / 2) * a
        
        def sinc(t):
            out = torch.ones_like(t)
            nz = t != 0
            out[nz] = torch.sin(math.pi * t[nz]) / (math.pi * t[nz])
            return out
        
        w = sinc(x) * sinc(x / a)
        w = (w - w.min()).clamp_min(0)  # keep it smoothing
    elif kind == "hann_poisson":
        hann = 0.5 - 0.5 * torch.cos(2 * math.pi * (n / (L - 1)))
        tau = max(1.0, L / 6.0)
        pois = torch.exp(-torch.abs(n - (L - 1) / 2) / tau)
        w = hann * pois
    else:
        raise ValueError(f"Unknown window kind: {kind}")
    
    w = w / (w.sum() + 1e-12)
    return w


class CausalFIRWindow(nn.Module):
    """
    Generic causal depthwise FIR with a differentiable window generator.
    
    Dynamically adjusts kernel size based on input sequence length.
    We build kernel w[0..L_eff-1] (causal), normalize sum=1, and apply depthwise conv.
    
    Args:
        channels: Number of input channels
        L: Maximum kernel length (will be clamped to seq_len if needed)
        num_kernels: Number of parallel kernels
        learnable_mix: Learn mixing weights across kernels
    """
    def __init__(self, channels, L=129, num_kernels=1, learnable_mix=False):
        super().__init__()
        self.C = channels
        self.L_max = L  # Maximum kernel length
        self.K = num_kernels
        self.learnable_mix = learnable_mix
        
        if learnable_mix and self.K > 1:
            self.mix_logits = nn.Parameter(torch.zeros(channels, self.K))

    def _effective_L(self, seq_len: int) -> int:
        """Compute effective kernel length clamped to sequence length."""
        return min(self.L_max, seq_len)

    def window(self, L_eff: int):
        """Override in subclasses. Build window of length L_eff."""
        raise NotImplementedError

    def build_kernel(self, x: torch.Tensor):
        """
        Build kernel with dynamic length based on input sequence.
        
        Args:
            x: Input tensor [B, T, C]
            
        Returns:
            Kernel tensor [C*K, 1, L_eff]
        """
        B, T, C = x.shape
        L_eff = self._effective_L(T)
        
        w = self.window(L_eff)  # [K, C, L_eff] or [1, C, L_eff]
        w = w / (w.sum(dim=-1, keepdim=True).clamp_min(1e-8))  # normalize
        return w.reshape(self.K * self.C, 1, L_eff).to(x.device).to(x.dtype), L_eff

    def forward_with_kernel(self, x: torch.Tensor, K: torch.Tensor, L_eff: int):
        """
        Apply causal FIR with the given kernel.
        
        Args:
            x: Input [B, T, C]
            K: Kernel [C*K, 1, L_eff]
            L_eff: Effective kernel length
            
        Returns:
            Filtered output [B, T, C]
        """
        B, T, C = x.shape
        xt = x.transpose(1, 2)  # [B, C, T]
        xt = F.pad(xt, (L_eff - 1, 0))  # causal left-pad
        y = F.conv1d(xt, K, groups=self.C)  # [B, C*K, T]

        if self.K == 1:
            pass  # y is already [B, C, T]
        elif self.learnable_mix:
            w = F.softmax(self.mix_logits.to(x.device).to(x.dtype), dim=-1)  # [C, K]
            y = (y.view(B, self.C, self.K, T) * w.view(1, self.C, self.K, 1)).sum(dim=2)
        else:
            y = y.view(B, self.C, self.K, T).mean(dim=2)
        
        return y.transpose(1, 2)  # [B, T, C]


class KaiserFIR(CausalFIRWindow):
    """
    Learnable-β Kaiser window FIR (causal), optionally multi-kernel.
    
    Dynamically adjusts kernel size for short sequences.

    Args:
        channels: Number of input channels
        L: Maximum kernel length (clamped to seq_len if needed)
        num_kernels: Number of parallel kernels to mix
        init_beta: Initial Kaiser β (larger → narrower main lobe)
        learnable_mix: Learn per-channel softmax mixing
    """
    def __init__(self, channels, L=129, num_kernels=1, init_beta=6.0, learnable_mix=False):
        super().__init__(channels, L, num_kernels, learnable_mix)
        # β per (K, C)
        self.log_beta = nn.Parameter(
            torch.log(torch.full((num_kernels, channels), float(init_beta)))
        )

    def window(self, L_eff: int):
        """Build Kaiser window of effective length L_eff."""
        device = self.log_beta.device
        dtype = torch.float32
        
        n = torch.arange(L_eff, device=device, dtype=dtype)
        # Map to [-1, 1] with center at (L_eff-1)/2
        m = (n - (L_eff - 1) / 2) / ((L_eff - 1) / 2 + 1e-8)  # [L_eff]
        
        beta = torch.exp(self.log_beta)  # [K, C]
        i0_beta = torch.i0(beta)  # [K, C]
        
        # Kaiser: I0(β * sqrt(1 - m²)) / I0(β)
        arg = beta.unsqueeze(-1) * torch.sqrt(torch.clamp(1 - m.pow(2), min=0))  # [K, C, L_eff]
        w = torch.i0(arg) / (i0_beta.unsqueeze(-1) + 1e-12)  # [K, C, L_eff]
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K, L_eff = self.build_kernel(x)  # [C*K, 1, L_eff]
        return self.forward_with_kernel(x, K, L_eff)
    
    def extra_repr(self) -> str:
        beta_mean = torch.exp(self.log_beta).mean().item()
        return f"L_max={self.L_max}, K={self.K}, beta_mean={beta_mean:.2f}"


class HannPoissonFIR(CausalFIRWindow):
    """
    Hann window multiplied by decaying Poisson exp(-λn), learnable λ.
    
    Dynamically adjusts kernel size for short sequences.

    Args:
        channels: Number of input channels
        L: Maximum kernel length (clamped to seq_len if needed)
        num_kernels: Number of parallel kernels
        init_lambda: Initial decay rate λ
        learnable_mix: Learn per-channel softmax mixing
    """
    def __init__(self, channels, L=129, num_kernels=1, init_lambda=0.02, learnable_mix=False):
        super().__init__(channels, L, num_kernels, learnable_mix)
        self.log_lambda = nn.Parameter(
            torch.log(torch.full((num_kernels, channels), float(init_lambda)))
        )

    def window(self, L_eff: int):
        """Build Hann-Poisson window of effective length L_eff."""
        device = self.log_lambda.device
        dtype = torch.float32
        
        n = torch.arange(L_eff, device=device, dtype=dtype)  # [L_eff]
        
        # Hann over [0..L_eff-1]
        hann = 0.5 * (1 - torch.cos(2 * math.pi * (n / (L_eff - 1)).clamp(0, 1)))  # [L_eff]
        
        lam = torch.exp(self.log_lambda)  # [K, C]
        pois = torch.exp(-lam.unsqueeze(-1) * n.view(1, 1, L_eff))  # [K, C, L_eff]
        
        w = pois * hann.view(1, 1, L_eff)  # [K, C, L_eff]
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K, L_eff = self.build_kernel(x)
        return self.forward_with_kernel(x, K, L_eff)
    
    def extra_repr(self) -> str:
        lam_mean = torch.exp(self.log_lambda).mean().item()
        return f"L_max={self.L_max}, K={self.K}, lambda_mean={lam_mean:.4f}"


# ============================================================
# 1) Learnable EMA (α per channel)
# ============================================================

class FastLearnableEMA(nn.Module):
    """
    EMA with learnable per-channel α, stable autograd (no in-place slicing).
    
    Args:
        channels: Number of channels
        init_alpha: Initial alpha value
        debias: Apply bias correction for start-up
        clamp: (min, max) clamp range for alpha
        
    Input/Output: x, trend ∈ [B, T, C]
    """
    def __init__(self, channels, init_alpha=0.9, debias=False, clamp=(1e-4, 1 - 1e-4)):
        super().__init__()
        self.debias = debias
        self.clamp = clamp
        init = torch.logit(torch.tensor(init_alpha).clamp(*clamp))
        self.logit_alpha = nn.Parameter(init.repeat(channels))  # [C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        a = torch.sigmoid(self.logit_alpha).clamp(*self.clamp).to(x.device, x.dtype)
        a = a.view(1, 1, C)  # [1, 1, C]

        # Time scan without in-place writes
        y0 = x[:, 0, :].unsqueeze(1)  # [B, 1, C]
        outs = [y0]

        one_minus_a = 1.0 - a  # [1, 1, C]
        for t in range(1, T):
            prev = outs[-1]  # [B, 1, C]
            xt = x[:, t, :].unsqueeze(1)  # [B, 1, C]
            yt = a * prev + one_minus_a * xt  # [B, 1, C]
            outs.append(yt)

        y = torch.cat(outs, dim=1)  # [B, T, C]

        if self.debias:
            t_idx = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
            a_pow = torch.pow(a, t_idx)
            denom = (1.0 - a_pow).clamp_min(1e-8)
            y = y / denom

        return y


# ============================================================
# 2) Multi-α EMA mixture (K time constants + learned mixing)
# ============================================================

class FastMultiEMAMixture(nn.Module):
    """
    K parallel EMAs with per-channel α_k and softmax mixture per channel.
    
    Numerically stable (no a^{-t}), causal.
    
    Args:
        channels: Number of channels
        K: Number of parallel EMAs
        init_alphas: Initial alpha values for each EMA
        clamp: Clamp range for alphas
        
    Input/Output: x, trend ∈ [B, T, C]
    """
    def __init__(self, channels, K=3, init_alphas=(0.8, 0.9, 0.98), clamp=(1e-4, 1 - 1e-4)):
        super().__init__()
        self.K = K
        if len(init_alphas) < K:
            init_alphas = list(init_alphas) + list(
                torch.linspace(0.7, 0.99, K - len(init_alphas)).tolist()
            )
        init = torch.stack([
            torch.logit(torch.tensor(a).clamp(*clamp)) for a in init_alphas[:K]
        ], dim=0)  # [K]
        self.logit_alpha = nn.Parameter(init.unsqueeze(-1).repeat(1, channels))  # [K, C]
        self.mix_logits = nn.Parameter(torch.zeros(channels, K))  # [C, K]
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute in fp32 for stability
        x32 = x.to(dtype=torch.float32)

        a = torch.sigmoid(self.logit_alpha).clamp(*self.clamp).to(
            device=x.device, dtype=torch.float32
        )  # [K, C]
        a = a.unsqueeze(0)  # [1, K, C]
        one_minus_a = 1.0 - a  # [1, K, C]

        # Init y_k at t=0 as x0
        x0 = x32[:, 0, :]  # [B, C]
        yk_prev = x0.unsqueeze(1).expand(B, self.K, C).contiguous()  # [B, K, C]

        # Allocate output buffer
        yk_all = torch.empty(B, T, self.K, C, device=x.device, dtype=torch.float32)
        yk_all[:, 0, :, :] = yk_prev

        # Stable recurrence over time
        for t in range(1, T):
            xt = x32[:, t, :].unsqueeze(1)  # [B, 1, C]
            yk_prev = a * yk_prev + one_minus_a * xt  # [B, K, C]
            yk_all[:, t, :, :] = yk_prev

        # Soft mix over K per channel
        mix = F.softmax(
            self.mix_logits.to(device=x.device, dtype=torch.float32), dim=-1
        )  # [C, K]
        trend = (yk_all * mix.T.view(1, 1, self.K, C)).sum(dim=2)  # [B, T, C]

        return trend.to(dtype=x.dtype)


# ============================================================
# 3) Debiased EMA (start-up bias correction)
# ============================================================

class DebiasedEMA(nn.Module):
    """
    EMA with bias correction: ŷ_t = y_t / (1 - α^t).
    
    Args:
        channels: Number of channels
        alpha: Smoothing factor
        learnable: Whether alpha is learnable
        clamp: Clamp range for alpha
    """
    def __init__(self, channels, alpha=0.9, learnable=False, clamp=(1e-4, 1 - 1e-4)):
        super().__init__()
        self.learnable = learnable
        self.clamp = clamp
        if learnable:
            init = torch.logit(torch.tensor(alpha).clamp(*clamp))
            self.logit_alpha = nn.Parameter(init.repeat(channels))
        else:
            self.register_buffer("alpha_fixed", torch.tensor(alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if self.learnable:
            a = torch.sigmoid(self.logit_alpha).clamp(*self.clamp).to(
                x.device, x.dtype
            ).view(1, 1, C)
        else:
            a = _as_device_dtype(x, self.alpha_fixed).clamp(*self.clamp).view(1, 1, 1).expand(1, 1, C)
        
        y = _init_like_first(x)
        for t in range(1, T):
            y[:, t, :] = a * y[:, t - 1, :] + (1 - a) * x[:, t, :]
        
        # Bias correction
        t_idx = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(T, 1)  # [T, 1]
        aC = a.view(1, C).expand(T, C)
        denom = (1 - torch.pow(aC, t_idx)).clamp_min(1e-6).unsqueeze(0)  # [1, T, C]
        return y / denom


# ============================================================
# 4) Alpha-Beta Filter (level + slope)
# ============================================================

class AlphaBetaFilter(nn.Module):
    """
    Causal 2-state filter (level + slope) with learnable per-channel α, β.
    
    Input/Output: x, trend ∈ [B, T, C]
    """
    def __init__(self, channels: int, init_alpha: float = 0.5, init_beta: float = 0.1,
                 clamp=(1e-4, 1 - 1e-4)):
        super().__init__()
        a0 = torch.logit(torch.tensor(init_alpha).clamp(*clamp))
        b0 = torch.logit(torch.tensor(init_beta).clamp(*clamp))
        self.logit_a = nn.Parameter(a0.repeat(channels))  # [C]
        self.logit_b = nn.Parameter(b0.repeat(channels))  # [C]
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        a = torch.sigmoid(self.logit_a).clamp(*self.clamp).to(x.device, x.dtype)  # [C]
        b = torch.sigmoid(self.logit_b).clamp(*self.clamp).to(x.device, x.dtype)  # [C]

        # Initialize level L0 = x0, slope V0 = 0
        L_prev = x[:, 0, :]  # [B, C]
        V_prev = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        outs = [L_prev.unsqueeze(1)]  # list of [B, 1, C]

        for t in range(1, T):
            pred = L_prev + V_prev  # [B, C]
            resid = x[:, t, :] - pred  # [B, C]
            L_t = pred + a * resid  # [B, C]
            V_t = V_prev + b * (L_t - L_prev - V_prev)  # [B, C]
            outs.append(L_t.unsqueeze(1))
            L_prev, V_prev = L_t, V_t

        return torch.cat(outs, dim=1)  # [B, T, C]


# ============================================================
# 5) EWRLS Level (online RLS with forgetting)
# ============================================================

class FastEWRLSLevel(nn.Module):
    """
    Level-only exponentially weighted RLS with learnable per-channel λ.
    
    Input/Output: x, trend ∈ [B, T, C]
    """
    def __init__(self, channels: int, init_lambda: float = 0.98,
                 clamp=(1e-4, 1 - 1e-4), init_P: float = 1.0):
        super().__init__()
        l0 = torch.logit(torch.tensor(init_lambda).clamp(*clamp))
        self.logit_lambda = nn.Parameter(l0.repeat(channels))  # [C]
        self.clamp = clamp
        self.init_P = float(init_P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        lam = torch.sigmoid(self.logit_lambda).clamp(*self.clamp).to(x.device, x.dtype)  # [C]

        theta = x[:, 0, :].clone()  # [B, C]
        P = torch.full((B, C), self.init_P, device=x.device, dtype=x.dtype)  # [B, C]
        outs = [theta.unsqueeze(1)]  # [B, 1, C]

        for t in range(1, T):
            e = x[:, t, :] - theta  # innovation [B, C]
            den = lam + P  # [B, C]
            K = P / den  # gain [B, C]
            theta = theta + K * e  # posterior [B, C]
            P = (P - K * P) / lam  # cov update [B, C]
            outs.append(theta.unsqueeze(1))

        return torch.cat(outs, dim=1)  # [B, T, C]


# ============================================================
# 6) Huber-Robust EMA
# ============================================================

class HuberEMA(nn.Module):
    """
    Robust EMA: EMA update driven by Huber pseudo-gradient of residual.
    
    Learnable per-channel α ∈ (0, 1). δ is kept as a (non-trainable) buffer.
    
    Input/Output: x, trend ∈ [B, T, C]
    """
    def __init__(self, channels: int, init_alpha: float = 0.9, delta: float = 1.0,
                 clamp=(1e-4, 1 - 1e-4)):
        super().__init__()
        a0 = torch.logit(torch.tensor(init_alpha).clamp(*clamp))
        self.logit_alpha = nn.Parameter(a0.repeat(channels))  # [C]
        self.clamp = clamp
        self.register_buffer("delta", torch.tensor(float(delta)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        a = torch.sigmoid(self.logit_alpha).clamp(*self.clamp).to(x.device, x.dtype)  # [C]
        d = self.delta.to(x.device, x.dtype)

        y_prev = x[:, 0, :]  # [B, C]
        outs = [y_prev.unsqueeze(1)]

        for t in range(1, T):
            r = x[:, t, :] - y_prev  # residual [B, C]
            # Huber gradient (piecewise linear)
            g = torch.where(r.abs() <= d, r, d * r.sign())
            y_t = y_prev + (1 - a) * g
            outs.append(y_t.unsqueeze(1))
            y_prev = y_t

        return torch.cat(outs, dim=1)  # [B, T, C]


# ============================================================
# 7) Alpha Cutoff Filter (first-order IIR from cutoff frequency)
# ============================================================

class AlphaCutoffFilter(nn.Module):
    """
    y_t = α y_{t-1} + (1-α) x_t, with α from (f_c, f_s):
        α = 1 - exp(-2π f_c / f_s)
    
    Args:
        channels: Number of channels
        fs: Sampling frequency
        init_fc: Initial cutoff frequency
        learnable_fc: Whether cutoff is learnable
        fc_bounds: (min, max) bounds for cutoff frequency
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if self.learnable_fc:
            fc = torch.exp(self.log_fc).clamp(*self.fc_bounds).to(x.device, x.dtype).view(1, 1, C)
        else:
            fc = _as_device_dtype(x, self.fc_fixed).clamp(*self.fc_bounds).view(1, 1, 1).expand(1, 1, C)
        
        fs = torch.tensor(self.fs, device=x.device, dtype=x.dtype).view(1, 1, 1)
        alpha = 1 - torch.exp(-2 * math.pi * fc / fs)  # (0, 1)
        
        y = _init_like_first(x)
        for t in range(1, T):
            y[:, t, :] = alpha * y[:, t - 1, :] + (1 - alpha) * x[:, t, :]
        return y


# ============================================================
# 8) One-Euro Filter
# ============================================================

class OneEuro(nn.Module):
    """
    One Euro Filter: adaptive cutoff based on signal derivative.
    
    cutoff_t = min_cutoff + β * |dx_t|
    
    Args:
        channels: Number of channels (unused, for API consistency)
        min_cutoff: Minimum cutoff frequency
        beta: Speed coefficient (how much derivative affects cutoff)
        dcutoff: Cutoff for derivative smoothing
        fs: Sampling frequency
    """
    def __init__(self, channels=None, min_cutoff=1.0, beta=0.007, dcutoff=1.0, fs=1.0):
        super().__init__()
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.fs = float(fs)

    def _alpha(self, cutoff, fs, dtype, device):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / fs
        return 1.0 / (1.0 + tau / te)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_prev = x[:, 0, :]
        dx_prev = torch.zeros_like(x_prev)
        y_prev = x_prev
        y = [y_prev.unsqueeze(1)]
        fs = torch.tensor(self.fs, device=x.device, dtype=x.dtype)
        
        for t in range(1, T):
            dx = (x[:, t, :] - x[:, t - 1, :]) * fs
            a_d = self._alpha(self.dcutoff, fs, x.dtype, x.device)
            dx_hat = a_d * dx + (1 - a_d) * dx_prev
            cutoff = self.min_cutoff + self.beta * dx_hat.abs()
            a_x = self._alpha(cutoff, fs, x.dtype, x.device)
            y_t = a_x * x[:, t, :] + (1 - a_x) * y_prev
            y.append(y_t.unsqueeze(1))
            dx_prev, y_prev = dx_hat, y_t
        
        return torch.cat(y, dim=1)


# ============================================================
# Factory
# ============================================================

def build_trend_module(name: str, channels: int, **kw) -> nn.Module:
    """
    Build a trend extraction module by name.
    
    Args:
        name: Module name (case insensitive)
        channels: Number of input channels
        **kw: Additional keyword arguments passed to the module
        
    Returns:
        nn.Module instance
    """
    n = name.lower()

    if n == "fast_ema":
        return FastLearnableEMA(
            channels=channels,
            init_alpha=kw.get("fastema_init_alpha", 0.9),
            debias=kw.get("fastema_debias", False),
        )

    if n == "fast_multi_ema":
        init_alphas = kw.get("mema_init_alphas", None)
        if isinstance(init_alphas, str) and init_alphas:
            init_alphas = [float(x) for x in init_alphas.split(",")]
        return FastMultiEMAMixture(
            channels=channels,
            K=kw.get("mema_K", 3),
            init_alphas=init_alphas if init_alphas is not None else (0.8, 0.9, 0.98),
        )

    if n == "alpha_beta":
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
        return KaiserFIR(
            channels=channels,
            L=kw.get("kaiser_L", 129),
            num_kernels=kw.get("kaiser_num_kernels", 1),
            init_beta=kw.get("kaiser_init_beta", 6.0),
            learnable_mix=kw.get("kaiser_learnable_mix", False),
        )

    if n == "hann_poisson_fir":
        return HannPoissonFIR(
            channels=channels,
            L=kw.get("hannp_L", 129),
            num_kernels=kw.get("hannp_num_kernels", 1),
            init_lambda=kw.get("hannp_init_lambda", 0.02),
            learnable_mix=kw.get("hannp_learnable_mix", False),
        )

    if n == "ewrls_fast":
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

    if n == "debiased_ema":
        return DebiasedEMA(
            channels=channels,
            alpha=kw.get("dema_alpha", 0.9),
            learnable=kw.get("dema_learnable", False),
        )

    if n == "alpha_cutoff":
        return AlphaCutoffFilter(
            channels=channels,
            fs=kw.get("ac_fs", 1.0),
            init_fc=kw.get("ac_init_fc", 0.05),
            learnable_fc=kw.get("ac_learnable_fc", True),
        )

    if n == "one_euro":
        return OneEuro(
            channels=channels,
            min_cutoff=kw.get("oe_min_cutoff", 1.0),
            beta=kw.get("oe_beta", 0.007),
            dcutoff=kw.get("oe_dcutoff", 1.0),
            fs=kw.get("oe_fs", 1.0),
        )

    raise ValueError(f"Unknown trend module: {name}")
