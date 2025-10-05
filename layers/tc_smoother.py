class TCSmoother(nn.Module):
    """
    Trend(x) via a light TCN:
      - N blocks of depthwise Conv1d (kernel=7) with dilations [1,2,4,1,...]
      - GLU gating + SE-like channel scaling
      - Residual with small beta to bias toward smoothing
      - Optional final LP blend with short averaging kernel
    """
    def __init__(self, channels: int, hidden_mult: float = 1.0, n_blocks: int = 2, kernel: int = 7, beta: float = 0.3, final_avg: int = 0):
        super().__init__()
        H = max(8, int(channels * hidden_mult))
        dilations = [1,2,4,1][:max(1,n_blocks)]
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(self._block(channels, kernel, dilation=d))
        self.beta = beta
        self.final_avg = final_avg

    def _block(self, C: int, k: int, dilation: int):
        pad = dilation * (k - 1) // 2  # "same" centered
        dw = nn.Conv1d(C, C, kernel_size=k, padding=pad, dilation=dilation, groups=C)
        pw_g = nn.Conv1d(C, 2*C, kernel_size=1)  # GLU gate
        se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(C, max(8, C//8), 1), nn.GELU(),
            nn.Conv1d(max(8, C//8), C, 1), nn.Sigmoid()
        )
        # init dw as near-averaging
        with torch.no_grad():
            w = torch.zeros_like(dw.weight)
            mid = (k - 1) // 2
            w[:, :, mid] = 1.0
            dw.weight.copy_(w)
            if dw.bias is not None: dw.bias.zero_()
        return nn.ModuleDict(dict(dw=dw, pwg=pw_g, se=se))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,C]
        B,T,C = x.shape
        y = x.transpose(1,2)  # [B,C,T]
        for blk in self.blocks:
            z = blk['dw'](y)                            # [B,C,T]
            g = blk['pwg'](z)                           # [B,2C,T]
            a, b = g.chunk(2, dim=1)
            h = a * torch.sigmoid(b)                    # GLU
            s = blk['se'](h)                            # [B,C,1]
            y = y + self.beta * (h * s)                 # residual smooth update
        y = y.transpose(1,2)
        if self.final_avg and self.final_avg > 1:
            k = self.final_avg
            pad = (k - 1) // 2
            w = torch.ones(1,1,k, device=y.device, dtype=y.dtype) / k
            y_bCt = y.transpose(1,2)
            y = F.conv1d(F.pad(y_bCt, (pad,pad), mode='reflect'), w.repeat(C,1,1), groups=C).transpose(1,2)
        return y  # trend [B,T,C]
