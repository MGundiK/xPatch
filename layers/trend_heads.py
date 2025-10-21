# layers/trend_heads.py
import math, torch
from torch import nn
import torch.nn.functional as F

class TrendHeadBase(nn.Module):
    def __init__(self, seq_len:int, pred_len:int):
        super().__init__(); self.seq_len=seq_len; self.pred_len=pred_len
    def forward(self, t: torch.Tensor) -> torch.Tensor: raise NotImplementedError
    def regularization(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(self.parameters()).device)

class BaselineMLPTrendHead(TrendHeadBase):
    def __init__(self, seq_len:int, pred_len:int):
        super().__init__(seq_len, pred_len)
        self.fc5 = nn.Linear(seq_len, pred_len*4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2); self.ln1 = nn.LayerNorm(pred_len*2)
        self.fc6 = nn.Linear(pred_len*2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2); self.ln2 = nn.LayerNorm(pred_len//2)
        self.fc7 = nn.Linear(pred_len//2, pred_len)
    def forward(self, t):
        x = self.fc5(t); x = self.avgpool1(x.unsqueeze(1)).squeeze(1); x = self.ln1(x)
        x = self.fc6(x); x = self.avgpool2(x.unsqueeze(1)).squeeze(1); x = self.ln2(x)
        return self.fc7(x)

class FIRTrendHead(TrendHeadBase):
    def __init__(self, seq_len:int, pred_len:int, k_list=(32,64), d_list=(1,4),
                 channels=16, gelu=True, aa_pool=True, smooth_l2=1e-5):
        super().__init__(seq_len, pred_len)
        assert len(k_list)==len(d_list)
        self.gelu, self.aa_pool, self.smooth_l2 = gelu, aa_pool, smooth_l2
        layers=[]; in_ch=1; ch=channels; self._convs=[]
        for i,(k,d) in enumerate(zip(k_list,d_list)):
            pad=(k-1)*d
            conv=nn.Conv1d(in_ch if i==0 else ch, ch, k, dilation=d, padding=pad)
            layers.append(conv); self._convs.append(conv)
            if gelu: layers.append(nn.GELU())
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(pred_len) if aa_pool else None
        if not aa_pool: self.to_feat = nn.Linear(seq_len, pred_len)
        self.refine = nn.Conv1d(ch, 1, 1)
    def forward(self, t):
        x = self.conv(t.unsqueeze(1))
        if self.pool: x = self.pool(x)
        else: x = self.to_feat(x.flatten(-2)); return x
        return self.refine(x).squeeze(1)
    def regularization(self):
        if self.smooth_l2<=0: return super().regularization()
        reg=0.0
        for c in self._convs:
            w=c.weight
            if w.shape[-1]>1: reg+=((w[...,1:]-w[...,:-1])**2).mean()
        return reg*self.smooth_l2

class BasisTrendHead(TrendHeadBase):
    def __init__(self, seq_len:int, pred_len:int, poly_degree=2, fourier_k=4,
                 normalize_t=True, l2_curv=1e-5):
        super().__init__(seq_len, pred_len)
        self.l2_curv=l2_curv
        t_pred = torch.linspace(0,1,pred_len) if normalize_t else torch.arange(pred_len).float()
        Phi=[t_pred**d for d in range(poly_degree+1)]
        for k in range(1, fourier_k+1):
            Phi += [torch.sin(2*math.pi*k*t_pred), torch.cos(2*math.pi*k*t_pred)]
        Phi_pred=torch.stack(Phi,0)
        self.nbasis=Phi_pred.shape[0]; self.register_buffer("Phi_pred", Phi_pred, persistent=False)
        self.to_coeff = nn.Linear(seq_len, self.nbasis)
        self.residual = nn.Sequential(nn.LayerNorm(self.nbasis), nn.Linear(self.nbasis, pred_len))
    def forward(self, t):
        c = self.to_coeff(t); y = c @ self.Phi_pred; return y + self.residual(c)
    def regularization(self):
        if self.l2_curv<=0: return super().regularization()
        P=self.pred_len; D2=torch.zeros(P-2,P,device=self.Phi_pred.device)
        r=torch.arange(P-2,device=self.Phi_pred.device); D2[r,r]=1; D2[r,r+1]=-2; D2[r,r+2]=1
        curv=((D2 @ self.Phi_pred.T)**2).mean(); return curv*self.l2_curv
