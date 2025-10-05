class LearnableLP(nn.Module):
    """
    Depthwise Conv1d smoother with per-channel kernels:
      - positivity via softplus
      - normalized to sum=1 (per channel)
    Modes:
      - 'centered': symmetric padding (reflect)
      - 'causal'  : left padding
    """
    def __init__(self, channels: int, kernel_size: int = 21, mode: str = "centered", ema_alpha: float = 0.3):
        super().__init__()
        assert mode in ("centered", "causal")
        self.channels = channels
        self.kernel_size = int(kernel_size)
        self.mode = mode
        # parameters: unconstrained logits -> softplus -> normalize
        self.weight_raw = nn.Parameter(torch.randn(channels, 1, kernel_size) * 0.01)
        self.bias = None  # not used
        # init to EMA-shaped kernel
        with torch.no_grad():
            k = self._init_ema_kernel(ema_alpha)  # [K]
            if mode == "centered":
                # center the EMA shape by reflecting around the center
                mid = (self.kernel_size - 1) // 2
                left = torch.flip(k[:mid+1], dims=(0,))
                right = k[:mid]
                k0 = torch.cat([left, right], dim=0)
                k0 = k0 / k0.sum()
            else:
                # causal: place EMA weights on the rightmost K entries
                k0 = torch.zeros(self.kernel_size)
                L = min(len(k), self.kernel_size)
                k0[-L:] = k[:L]
                k0 = k0 / k0.sum()
            self.weight_raw.copy_(k0.view(1,1,-1).repeat(channels,1,1))

    def _init_ema_kernel(self, alpha: float):
        # build a long causal EMA impulse response and cut to kernel_size
        L = self.kernel_size
        t = torch.arange(L, dtype=torch.float32)
        k = (1.0 - alpha) ** (L - 1 - t)
        k[1:] *= alpha
        k = k / k.sum()
        return k  # causal EMA tail, length K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,C] -> [B,C,T]
        B,T,C = x.shape
        x_bCt = x.transpose(1,2)
        # positivity + normalization
        w = F.softplus(self.weight_raw)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)  # [C,1,K]
        pad = (self.kernel_size - 1)
        if self.mode == "centered":
            padL = (self.kernel_size - 1) // 2
            padR = pad - padL
            x_pad = F.pad(x_bCt, (padL, padR), mode='reflect')
        else:
            # causal: left-pad, no right pad
            x_pad = F.pad(x_bCt, (pad, 0), mode='replicate')
        y = F.conv1d(x_pad, w, bias=None, stride=1, padding=0, groups=C)
        return y.transpose(1,2)  # [B,T,C]
