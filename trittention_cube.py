import einops
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
from utils_misc import Config


class TrittentionCube(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_A = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_B = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_C = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_D = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_E = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))

        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_head, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_head, cfg.d_head)))

        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))

        self.b_A = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_B = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_C = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_D = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_E = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_DE = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))

        nn.init.normal_(self.W_A, std=self.cfg.init_range)
        nn.init.normal_(self.W_B, std=self.cfg.init_range)
        nn.init.normal_(self.W_C, std=self.cfg.init_range)
        nn.init.normal_(self.W_D, std=self.cfg.init_range)
        nn.init.normal_(self.W_E, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.register_buffer("IGNORE", t.tensor(-1e6, dtype=t.float32, device=device))


    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        # Assuming self.W_Q, self.W_K, self.W_V, and self.W_O are parameter matrices of the model
        bs, ts, ds = normalized_resid_pre.shape
        a = t.einsum('ndh,bpd->bpnh', self.W_A, normalized_resid_pre) + self.b_A
        b = t.einsum('ndh,bpd->bpnh', self.W_B, normalized_resid_pre) + self.b_B
        c = t.einsum('ndh,bpd->bpnh', self.W_C, normalized_resid_pre) + self.b_C

        d = t.einsum('ndh,bpd->bpnh', self.W_D, normalized_resid_pre) + self.b_D
        e = t.einsum('ndh,bpd->bpnh', self.W_E, normalized_resid_pre) + self.b_E
        v = einops.einsum(d,e,self.W_V,"b p1 n h1, b p2 n h2, n h1 h2 h3 -> b p1 p2 n h3")

        #attn_score = t.einsum('b p1 n h1, b p2 n h2, b p3 n h3, n h1 h2 h3 -> b n p1 p2 p3', a, b, c, self.W_K)
        step1 = t.einsum('brnk, nijk -> brij', c, self.W_K)
        step2 = t.einsum('brij, bqnj -> briq', step1, b)
        attn_score = t.einsum('briq, bpni -> bnpqr', step2, a)

        attn_score = self.apply_causal_mask(attn_score)

        attn_score = einops.rearrange(attn_score, "b n p1 p2 p3 -> b n p3 (p1 p2)")/self.cfg.d_head
        extra = -1000*t.ones((bs,self.cfg.n_heads,ts, 1), device=attn_score.device)
        attn_score = t.cat((attn_score, extra), dim=-1)
        attn_score = attn_score.softmax(dim=-1)
        v = einops.rearrange(v, "b p s n h -> b n h (p s)")
        extra = t.zeros((bs, self.cfg.n_heads, self.cfg.d_head, 1), device=v.device)
        v = t.cat((v,extra), dim=-1)
        z = t.einsum('bnql, bnhl -> bqnh', attn_score, v)
        zint =  t.einsum('bpnh,nhd->bpnd', z, self.W_O)
        out = einops.reduce(zint,"b p n d -> b p d", reduction='sum') + self.b_O
        return out#, z


    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        b, nn, tt, s, q = attn_scores.shape

        t_indices = t.arange(tt).unsqueeze(0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Shape: (1, t, 1, 1)
        s_indices = t.arange(s).unsqueeze(0).unsqueeze(0).unsqueeze(1).unsqueeze(-1)  # Shape: (1, 1, s, 1)
        q_indices = t.arange(q).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)   # Shape: (1, 1, 1, q)

        if self.cfg.order_attn:
            mask = (t_indices >= s_indices) | (t_indices >= q_indices) | (t_indices >= s_indices)
        else:
            mask = (t_indices >= q_indices) | (s_indices >= q_indices) | (t_indices == s_indices)
        mask = mask.to(attn_scores.device)

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores