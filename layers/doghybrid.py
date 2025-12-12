"""
Difference of Gaussians (DoG) Seasonal Extraction and Hybrid EMA-DoG Decomposition.

Handles edge cases where kernel size would exceed input sequence length
by dynamically clamping the kernel size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gauss_kernel1d_safe(
    sigma: float,
    truncate: float,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a 1D Gaussian kernel with size clamped to sequence length.
    
    Args:
        sigma: Standard deviation
        truncate: Number of sigmas to extend kernel
        seq_len: Input sequence length (for clamping)
        dtype: Target dtype
        device: Target device
        
    Returns:
        Normalized kernel [K] where K <= seq_len (and K is odd)
    """
    # Desired radius
    radius_desired = max(1, int(truncate * sigma + 0.5))
    
    # Clamp: K = 2*R + 1 <= seq_len, so R <= (seq_len - 1) / 2
    max_radius = max(1, (seq_len - 1) // 2)
    radius = min(radius_desired, max_radius)
    
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    k = k / (k.sum() + 1e-12)
    return k  # [K]


def _depthwise_conv1d_centered_safe(
    x_btC: torch.Tensor,
    kernel_1d: torch.Tensor,
) -> torch.Tensor:
    """
    Depthwise centered 1D convolution with safe padding.
    
    Falls back to replicate padding if kernel is too large for reflect.
    
    Args:
        x_btC: Input [B, T, C]
        kernel_1d: 1D kernel [K]
        
    Returns:
        Filtered output [B, T, C]
    """
    B, T, C = x_btC.shape
    x_bCt = x_btC.transpose(1, 2)  # [B, C, T]
    
    k = kernel_1d.to(x_bCt.dtype).to(x_bCt.device)  # [K]
    K = k.numel()
    k = k.view(1, 1, -1).repeat(C, 1, 1)  # [C, 1, K] depthwise
    
    pad = (K - 1) // 2
    
    # Choose padding mode: reflect requires pad < T
    if pad < T:
        x_pad = F.pad(x_bCt, (pad, pad), mode='reflect')
    else:
        x_pad = F.pad(x_bCt, (pad, pad), mode='replicate')
    
    y = F.conv1d(x_pad, k, bias=None, stride=1, padding=0, groups=C)
    return y.transpose(1, 2)  # [B, T, C]


class EMA_Trend(nn.Module):
    """
    Fast EMA trend with scalar alpha (fixed, not learnable).
    
    Uses cumulative formula for O(1) per-step computation.
    """
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, C]
        Returns:
            EMA trend [B, T, C]
        """
        B, T, C = x.shape
        device, dtype = x.device, x.dtype
        
        # Compute via cumulative formula (no loop)
        t_idx = torch.arange(T, dtype=torch.float64, device=device)
        w = (1.0 - self.alpha) ** torch.flip(t_idx, dims=(0,))  # [T]
        w = w.to(dtype)
        w_div = w.clone()
        w[1:] *= self.alpha
        w = w.view(1, T, 1)
        w_div = w_div.view(1, T, 1)
        
        num = torch.cumsum(x * w, dim=1)
        ema = (num / (w_div + 1e-12)).to(dtype)
        return ema


class DoG_Seasonal(nn.Module):
    """
    Difference of Gaussians (DoG) for seasonal extraction.
    
    Seasonal(x) = G(σ1)*x - G(σ2)*x
    
    Where σ1 < σ2, so G(σ1) passes higher frequencies and G(σ2) extracts trend.
    The difference captures the seasonal band.
    
    Automatically adjusts kernel size for short sequences.
    
    Args:
        sigma1: Smaller sigma (passes daily + seasonal)
        sigma2: Larger sigma (extracts trend, removes seasonal)
        truncate: Kernel truncation factor
    """
    def __init__(
        self,
        sigma1: float = 4.2,
        sigma2: float = 96.0,
        truncate: float = 4.0,
    ):
        super().__init__()
        assert sigma2 > sigma1, "Require sigma2 > sigma1"
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.truncate = float(truncate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract seasonal component via DoG.
        
        Args:
            x: Input [B, T, C]
            
        Returns:
            Seasonal component [B, T, C]
        """
        B, T, C = x.shape
        
        # Build kernels with dynamic size clamping
        k1 = _gauss_kernel1d_safe(
            self.sigma1, self.truncate, T, x.dtype, x.device
        )
        k2 = _gauss_kernel1d_safe(
            self.sigma2, self.truncate, T, x.dtype, x.device
        )
        
        # Apply filters
        y1 = _depthwise_conv1d_centered_safe(x, k1)  # LP with small sigma
        y2 = _depthwise_conv1d_centered_safe(x, k2)  # LP with large sigma (trend)
        
        # Difference = bandpass (seasonal)
        seasonal = y1 - y2
        return seasonal
    
    def extra_repr(self) -> str:
        return f"sigma1={self.sigma1}, sigma2={self.sigma2}, truncate={self.truncate}"


class DoG_Trend(nn.Module):
    """
    DoG-based trend extraction (returns the smoother component).
    
    Trend(x) = G(σ2)*x (the large-sigma smoothed version)
    
    This is complementary to DoG_Seasonal:
        x ≈ Trend + Seasonal + Residual
        where Trend = G(σ2)*x and Seasonal = G(σ1)*x - G(σ2)*x
    
    Args:
        sigma: Smoothing sigma for trend
        truncate: Kernel truncation factor
    """
    def __init__(self, sigma: float = 96.0, truncate: float = 4.0):
        super().__init__()
        self.sigma = float(sigma)
        self.truncate = float(truncate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract trend via Gaussian smoothing.
        
        Args:
            x: Input [B, T, C]
            
        Returns:
            Trend [B, T, C]
        """
        B, T, C = x.shape
        k = _gauss_kernel1d_safe(
            self.sigma, self.truncate, T, x.dtype, x.device
        )
        return _depthwise_conv1d_centered_safe(x, k)


class HybridEMA_DoG(nn.Module):
    """
    Hybrid decomposition: EMA for trend, DoG for seasonal.
    
    Returns (trend, seasonal) so you can wire directly into xPatch streams.
    
    Args:
        alpha: EMA smoothing factor for trend
        sigma1: DoG small sigma
        sigma2: DoG large sigma  
        truncate: Kernel truncation
    """
    def __init__(
        self,
        alpha: float = 0.3,
        sigma1: float = 4.2,
        sigma2: float = 96.0,
        truncate: float = 4.0,
    ):
        super().__init__()
        self.ema = EMA_Trend(alpha)
        self.dog = DoG_Seasonal(sigma1, sigma2, truncate)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Decompose into trend and seasonal.
        
        Args:
            x: Input [B, T, C]
            
        Returns:
            (trend, seasonal) each [B, T, C]
        """
        trend = self.ema(x)
        seasonal = self.dog(x)  # Computed on raw x
        return trend, seasonal


class MultiScaleDoG(nn.Module):
    """
    Multi-scale DoG decomposition.
    
    Extracts multiple seasonal bands at different scales.
    Useful for capturing both daily and weekly patterns.
    
    Args:
        sigma_pairs: List of (sigma_low, sigma_high) pairs for each band
        truncate: Kernel truncation factor
    """
    def __init__(
        self,
        sigma_pairs: list = [(4, 24), (24, 96)],
        truncate: float = 4.0,
    ):
        super().__init__()
        self.dogs = nn.ModuleList([
            DoG_Seasonal(s1, s2, truncate) for s1, s2 in sigma_pairs
        ])
        self.sigma_pairs = sigma_pairs
    
    def forward(self, x: torch.Tensor) -> list:
        """
        Extract multiple seasonal bands.
        
        Args:
            x: Input [B, T, C]
            
        Returns:
            List of seasonal components, one per scale
        """
        return [dog(x) for dog in self.dogs]
