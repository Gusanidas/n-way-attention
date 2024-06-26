import torch as t
import torch.nn as nn
from nway_attention.cfgs import Config
import torch.nn.functional as F

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class Attention(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        # output projection
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.cfg.d_model, dim=2)
        k = k.view(B, T, self.cfg.n_heads, C // self.cfg.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.cfg.n_heads, C // self.cfg.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.cfg.n_heads, C // self.cfg.n_heads).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


if __name__ == "__main__":
    cfg = Config()
    model = Attention(cfg)
    x = t.randn(12, 1024, 768)
    y = model(x)
    print(y.shape)