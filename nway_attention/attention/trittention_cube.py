import einops
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float

from nway_attention.utils_misc import IdentityModule
from nway_attention.cfgs import Config


import torch as t
import torch.nn as nn

class TrittentionCube(nn.Module):
    IGNORE: Float[Tensor, ""]
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        # Initialize all weights with normal distribution
        self.init_range = cfg.init_range  # Assuming this is the standard deviation you want to use

        self.kkqvv = nn.Linear(cfg.d_model, cfg.d_head*cfg.n_heads*5)
        nn.init.normal_(self.kkqvv.weight, std=self.init_range)
        nn.init.zeros_(self.kkqvv.bias)

        self.W_Kq = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_head, cfg.d_head)))
        nn.init.normal_(self.W_Kq, std=self.init_range)

        self.W_Vq = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_head, cfg.d_head)))
        nn.init.normal_(self.W_Vq, std=self.init_range)

        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))

        self.Out = nn.Linear(cfg.d_head*cfg.n_heads, cfg.d_model)
        nn.init.normal_(self.Out.weight, std=self.init_range)
        nn.init.zeros_(self.Out.bias)

        self.Mask = IdentityModule()
        self.AttentionScore = IdentityModule()
        self.HeadOutputs = IdentityModule()

        self.register_buffer("IGNORE", t.tensor(-1e6, dtype=t.float32))
        self.register_buffer('precomputed_mask', self.create_causal_mask(cfg.n_ctx))

    def create_causal_mask(self, max_seq_len):
        
        t_indices = t.arange(max_seq_len).unsqueeze(0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Shape: (1,1, t, 1, 1)
        s_indices = t.arange(max_seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(1).unsqueeze(-1)  # Shape: (1,1, 1, s, 1)
        q_indices = t.arange(max_seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)   # Shape: (1,1, 1, 1, q)
        mask = (t_indices > q_indices) | (s_indices > q_indices)
        return mask


    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        # Assuming self.W_Q, self.W_K, self.W_V, and self.W_O are parameter matrices of the model
        bs, ts, ds = normalized_resid_pre.shape


        k1, k2, q, v1, v2 = self.kkqvv(normalized_resid_pre).chunk(5, dim=-1)
        k1, k2, q, v1, v2 = map(lambda t: einops.rearrange(t, 'b p (h d) -> b h p d', h=self.cfg.n_heads), (k1, k2, q, v1, v2))
        v = einops.einsum(v1,v2,self.W_Vq,"b n p1 h1, b n p2 h2, n h1 h2 h3 -> b n p1 p2 h3") #+ self.b_V
        v = einops.rearrange(v, "b n p s h -> b n h (p s)")
        
        #step1 = t.einsum('brnk, nijk -> bnrij', q, self.W_Kq)
        #step2 = t.einsum('bnrij, bqnj -> bnriq', step1, k2)
        #attn_score = t.einsum('bnriq, bpni -> bnpqr', step2, k1)

        step1 = einops.einsum(k1,k2, self.W_Kq, 'b n p k, b n t j, n k j i -> b n p t i')#, k1, k2, self.W_Kq)
        attn_score = einops.einsum(step1, q, 'b n p t i, b n q i -> b n p t q')#, step1, q)
        
        # attn_score = einops.einsum(k1, k2, q, self.W_Kq, "b n s i, b n t j, b n q k, n i j k -> b n s t q")
        #attn_score = t.einsum("bsni, btnj, bqnk, nijk -> bnstq", k1,k2,q, self.W_Kq)
        if self.cfg.causal_attn:
            attn_score = self.apply_causal_mask(attn_score)
        attn_score = einops.rearrange(attn_score, "b n s t q -> b n q (s t)")/self.cfg.d_head
        
        attn_score = attn_score.softmax(dim=-1)
        attn_score = self.AttentionScore(attn_score)
        z = t.einsum('bnql, bnhl -> bqnh', attn_score, v)
        z = self.HeadOutputs(z)
        out = self.Out(z.reshape(bs,ts,-1))
        return out


    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads a_pos b_pos c_pos"]
    ) -> Float[Tensor, "batch n_heads a_pos b_pos c_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        b, nn, tt, s, q = attn_scores.shape
        mask = self.precomputed_mask[:,:, :tt, :s, :q].to(attn_scores.device)
        mask = self.Mask(mask)
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


if __name__ == "__main__":
    cfg = Config()
    trittention = TrittentionCube(cfg)
    x = t.rand((2, 48, cfg.d_model))
    out = trittention(x)
    print(out.shape)
    print(out)
    print("Successfully ran a forward pass of TrittentionCube")