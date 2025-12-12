# layers/gaussma.py
"""
Gaussian Moving Average for trend extraction.

Handles edge cases where kernel size would exceed input sequence length
by dynamically clamping the kernel size.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- helpers ---

def sigma_from_alpha(alpha: float) -> float:
    """EMA alpha -> approx Gaussian sigma (variance-matched geometric kernel)."""
    a = float(alpha)
    a = max(min(a, 0.9999), 1e-6)
    return math.sqrt(1.0 - a) / a


def gaussian_sigma_from_period(P: int, mult: float = 0.75) -> float:
    """Convert period to Gaussian sigma."""
    return float(P) * float(mult)


def _gaussian_kernel_1d(
    sigma: torch.Tensor, 
    truncate: float = 4.0,
    max_kernel_size: int = None,
) -> torch.Tensor:
    """
    Build a 1D Gaussian kernel.
    
    Args:
        sigma: Standard deviation (scalar tensor)
        truncate: Number of sigmas to extend the kernel
        max_kernel_size: Maximum allowed kernel size (for short sequences)
        
    Returns:
        Normalized kernel of shape [K] where K is odd
    """
    sigma = sigma.float().clamp_min(1e-3)
    R = int((truncate * sigma).round().clamp_min(1).item())
    
    # Clamp kernel radius if max_kernel_size is specified
    if max_kernel_size is not None:
        max_R = (max_kernel_size - 1) // 2
        R = min(R, max(1, max_R))
    
    x = torch.arange(-R, R + 1, device=sigma.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k  # [K], odd length


def _depthwise_conv1d_reflect(
    x_bct: torch.Tensor, 
    kernel_1d: torch.Tensor,
) -> torch.Tensor:
    """
    Apply depthwise 1D convolution with reflect padding.
    
    Handles edge case where kernel is larger than input by using
    replicate padding instead of reflect when necessary.
    
    Args:
        x_bct: Input tensor [B, C, T]
        kernel_1d: 1D kernel [K]
        
    Returns:
        Filtered tensor [B, C, T]
    """
    B, C, T = x_bct.shape
    K = kernel_1d.numel()
    pad_size = K // 2
    
    # Choose padding mode based on input size
    # Reflect padding requires pad_size < T
    if pad_size < T:
        x_pad = F.pad(x_bct, (pad_size, pad_size), mode="reflect")
    else:
        # Fall back to replicate padding for very short sequences
        x_pad = F.pad(x_bct, (pad_size, pad_size), mode="replicate")
    
    w = kernel_1d.view(1, 1, K).repeat(C, 1, 1)  # [C, 1, K], depthwise
    return F.conv1d(x_pad, w, stride=1, padding=0, groups=C)


def _gaussian_kernel_1d_safe(
    sigma: torch.Tensor,
    truncate: float,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a Gaussian kernel that's guaranteed to fit the sequence length.
    
    Args:
        sigma: Standard deviation
        truncate: Truncation factor
        seq_len: Input sequence length (to clamp kernel size)
        device: Target device
        dtype: Target dtype
        
    Returns:
        Normalized kernel [K] where K <= seq_len
    """
    sigma_val = sigma.float().clamp_min(1e-3)
    
    # Compute desired radius
    R_desired = int((truncate * sigma_val).round().clamp_min(1).item())
    
    # Clamp to ensure kernel fits: K = 2*R + 1 <= seq_len
    # So R <= (seq_len - 1) / 2
    max_R = max(1, (seq_len - 1) // 2)
    R = min(R_desired, max_R)
    
    x = torch.arange(-R, R + 1, device=device, dtype=dtype)
    
    # Use the original sigma for the Gaussian shape, even if truncated
    k = torch.exp(-0.5 * (x / sigma_val.to(device)) ** 2)
    k = k / (k.sum() + 1e-12)
    
    return k


# --- single-pass Gaussian moving average ---

class GaussianMA(nn.Module):
    """
    Centered Gaussian moving average for inputs [B, T, C].
    
    Automatically adjusts kernel size for short sequences to prevent
    padding errors.
    
    Args:
        sigma: Initial standard deviation for Gaussian kernel
        learnable: If True, sigma is a learnable parameter
        truncate: Number of sigmas to extend kernel (default 4.0)
        min_sigma: Minimum sigma value (prevents degenerate kernels)
        max_sigma: Maximum sigma value (prevents excessive smoothing)
    """
    def __init__(
        self, 
        sigma: float, 
        learnable: bool = False, 
        truncate: float = 4.0,
        min_sigma: float = 0.5,
        max_sigma: float = 100.0,
    ):
        super().__init__()
        self.truncate = float(truncate)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        
        # Store sigma in log-space for unconstrained optimization
        if learnable:
            # Use log parameterization for stability
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma))))
        else:
            self.register_buffer("log_sigma", torch.log(torch.tensor(float(sigma))))
        
        self.learnable = learnable
    
    @property
    def sigma(self) -> torch.Tensor:
        """Get current sigma value (clamped to valid range)."""
        return torch.exp(self.log_sigma).clamp(self.min_sigma, self.max_sigma)
    
    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian smoothing.
        
        Args:
            x_btc: Input tensor [B, T, C]
            
        Returns:
            Smoothed tensor [B, T, C]
        """
        B, T, C = x_btc.shape
        
        # Build kernel with size clamped to sequence length
        k = _gaussian_kernel_1d_safe(
            sigma=self.sigma,
            truncate=self.truncate,
            seq_len=T,
            device=x_btc.device,
            dtype=x_btc.dtype,
        )
        
        # Apply depthwise convolution
        y_bct = _depthwise_conv1d_reflect(x_btc.transpose(1, 2), k)
        return y_bct.transpose(1, 2)  # [B, T, C]
    
    def extra_repr(self) -> str:
        return f"sigma={self.sigma.item():.2f}, learnable={self.learnable}, truncate={self.truncate}"


# --- Causal Gaussian MA (for compatibility with causal methods) ---

class GaussianMACausal(nn.Module):
    """
    Causal (one-sided) Gaussian moving average.
    
    Uses only past values (no future leakage).
    Kernel is the right half of a Gaussian: k[n] for n = 0, 1, ..., R
    
    Args:
        sigma: Standard deviation
        learnable: If True, sigma is learnable
        truncate: Truncation factor
    """
    def __init__(
        self,
        sigma: float,
        learnable: bool = False,
        truncate: float = 4.0,
        min_sigma: float = 0.5,
        max_sigma: float = 100.0,
    ):
        super().__init__()
        self.truncate = float(truncate)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        
        if learnable:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma))))
        else:
            self.register_buffer("log_sigma", torch.log(torch.tensor(float(sigma))))
        
        self.learnable = learnable
    
    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma).clamp(self.min_sigma, self.max_sigma)
    
    def _causal_kernel(self, seq_len: int, device, dtype) -> torch.Tensor:
        """Build causal (right-half) Gaussian kernel."""
        sigma_val = self.sigma.to(device)
        
        # Compute radius, clamped to sequence length
        R_desired = int((self.truncate * sigma_val).round().clamp_min(1).item())
        R = min(R_desired, max(1, seq_len - 1))
        
        # Causal kernel: n = 0, 1, ..., R (only past/present)
        n = torch.arange(0, R + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (n / sigma_val) ** 2)
        k = k / (k.sum() + 1e-12)
        
        return k  # [R+1]
    
    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        """
        Apply causal Gaussian smoothing.
        
        Args:
            x_btc: Input tensor [B, T, C]
            
        Returns:
            Smoothed tensor [B, T, C] (causal: uses only past values)
        """
        B, T, C = x_btc.shape
        x_bct = x_btc.transpose(1, 2)  # [B, C, T]
        
        k = self._causal_kernel(T, x_btc.device, x_btc.dtype)
        K = k.numel()
        
        # Causal padding: left-pad only
        pad_left = K - 1
        x_pad = F.pad(x_bct, (pad_left, 0), mode="replicate")
        
        # Depthwise convolution
        w = k.view(1, 1, K).repeat(C, 1, 1)  # [C, 1, K]
        y_bct = F.conv1d(x_pad, w, stride=1, padding=0, groups=C)
        
        return y_bct.transpose(1, 2)  # [B, T, C]


# --- two-pass Gaussian decomposition (daily→weekly etc.) ---

class GaussianMA2Pass(nn.Module):
    """
    Two-pass Gaussian decomposition returning (seasonal_total, trend_final).
    
    Pass 1: trend1 = G(sigma1)*x; seasonal1 = x - trend1
    Pass 2 on x_res = x - seasonal1: trend2 = G(sigma2)*x_res; seasonal2 = x_res - trend2
    Output: seasonal_total = seasonal1 + seasonal2, trend_final = trend2
    
    Args:
        sigma1: First pass sigma (smaller, captures fast variations)
        sigma2: Second pass sigma (larger, captures slow trend)
        P1, mult1: Alternative: compute sigma1 = P1 * mult1
        P2, mult2: Alternative: compute sigma2 = P2 * mult2
        learnable: If True, sigmas are learnable
        truncate: Truncation factor for kernels
    """
    def __init__(
        self,
        sigma1: float = None,
        sigma2: float = None,
        P1: int = None,
        mult1: float = None,
        P2: int = None,
        mult2: float = None,
        learnable: bool = False,
        truncate: float = 4.0,
    ):
        super().__init__()
        
        # Resolve sigma values
        if sigma1 is None:
            if P1 is None or mult1 is None:
                raise ValueError("Provide sigma1 or (P1, mult1).")
            sigma1 = gaussian_sigma_from_period(P1, mult1)
        if sigma2 is None:
            if P2 is None or mult2 is None:
                raise ValueError("Provide sigma2 or (P2, mult2).")
            sigma2 = gaussian_sigma_from_period(P2, mult2)
        
        self.ma1 = GaussianMA(sigma1, learnable=learnable, truncate=truncate)
        self.ma2 = GaussianMA(sigma2, learnable=learnable, truncate=truncate)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Two-pass decomposition.
        
        Args:
            x: Input tensor [B, T, C]
            
        Returns:
            (seasonal_total, trend_final) each [B, T, C]
        """
        trend1 = self.ma1(x)          # [B, T, C]
        seasonal1 = x - trend1
        x_res = x - seasonal1
        trend2 = self.ma2(x_res)
        seasonal2 = x_res - trend2
        return seasonal1 + seasonal2, trend2


# --- Adaptive Gaussian MA (sigma selected based on input statistics) ---

class AdaptiveGaussianMA(nn.Module):
    """
    Gaussian MA with input-adaptive sigma selection.
    
    Uses local signal statistics to choose sigma from a bank of values.
    This is a simpler alternative to AdaptiveGaussianTrendCausal.
    
    Args:
        sigma_bank: Tuple of sigma values to choose from
        stat_window: Window size for computing local statistics
        learnable_mix: If True, learn mixing weights
    """
    def __init__(
        self,
        sigma_bank: tuple = (4, 8, 16, 32),
        stat_window: int = 16,
        truncate: float = 4.0,
        learnable_mix: bool = True,
    ):
        super().__init__()
        self.sigma_bank = list(sigma_bank)
        self.stat_window = stat_window
        self.truncate = truncate
        self.K = len(sigma_bank)
        
        # Learnable mixing based on local variance
        if learnable_mix:
            self.mix_net = nn.Sequential(
                nn.Linear(2, 32),  # Input: [local_mean, local_std]
                nn.GELU(),
                nn.Linear(32, self.K),
            )
        else:
            self.mix_net = None
    
    def _compute_local_stats(self, x: torch.Tensor) -> tuple:
        """Compute causal local mean and std."""
        B, T, C = x.shape
        x_bct = x.transpose(1, 2)  # [B, C, T]
        
        # Causal sum via conv with ones
        ones = torch.ones(1, 1, self.stat_window, device=x.device, dtype=x.dtype)
        ones = ones.repeat(C, 1, 1)  # [C, 1, W]
        
        pad = self.stat_window - 1
        x_pad = F.pad(x_bct, (pad, 0), mode="replicate")
        x2_pad = F.pad(x_bct ** 2, (pad, 0), mode="replicate")
        
        sum_x = F.conv1d(x_pad, ones, groups=C)  # [B, C, T]
        sum_x2 = F.conv1d(x2_pad, ones, groups=C)
        
        mean = sum_x / self.stat_window
        var = (sum_x2 / self.stat_window - mean ** 2).clamp_min(1e-8)
        std = var.sqrt()
        
        return mean.transpose(1, 2), std.transpose(1, 2)  # [B, T, C]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive Gaussian smoothing.
        
        Args:
            x: Input [B, T, C]
            
        Returns:
            Smoothed output [B, T, C]
        """
        B, T, C = x.shape
        
        # Compute all filtered versions
        filtered = []
        for sigma in self.sigma_bank:
            k = _gaussian_kernel_1d_safe(
                sigma=torch.tensor(sigma, device=x.device),
                truncate=self.truncate,
                seq_len=T,
                device=x.device,
                dtype=x.dtype,
            )
            y = _depthwise_conv1d_reflect(x.transpose(1, 2), k).transpose(1, 2)
            filtered.append(y)
        
        Y = torch.stack(filtered, dim=-1)  # [B, T, C, K]
        
        if self.mix_net is not None:
            # Compute mixing weights from local stats
            mean, std = self._compute_local_stats(x)  # [B, T, C]
            features = torch.stack([mean, std], dim=-1)  # [B, T, C, 2]
            logits = self.mix_net(features)  # [B, T, C, K]
            weights = F.softmax(logits, dim=-1)
        else:
            # Uniform mixing
            weights = torch.ones(B, T, C, self.K, device=x.device) / self.K
        
        # Weighted combination
        out = (Y * weights).sum(dim=-1)  # [B, T, C]
        return out
