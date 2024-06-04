import einops
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
from jaxtyping import Float
from nway_attention.cfgs import Config

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))


    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        # Assuming self.W_Q, self.W_K, self.W_V, and self.W_O are parameter matrices of the model
        q = t.einsum('ndh,bpd->bpnh', self.W_Q, normalized_resid_pre) + self.b_Q
        k = t.einsum('ndh,bpd->bpnh', self.W_K, normalized_resid_pre) + self.b_K
        v = t.einsum('ndh,bpd->bpnh', self.W_V, normalized_resid_pre) + self.b_V

        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.cfg.causal_attn)
        z = t.einsum('bpnh,nhd->bpnd', y, self.W_O) + self.b_O

        out = einops.reduce(z,"b p n d -> b p d", reduction='sum') + self.b_O
        return out
