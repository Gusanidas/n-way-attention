import einops
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float

from nway_attention.utils_misc import IdentityModule
from nway_attention.cfgs import Config


class Trittention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_K1 = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K2 = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V12 = nn.Parameter(t.empty((cfg.n_heads, 2*cfg.d_model, cfg.d_head)))

        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))

        self.b_K1 = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K2 = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V12 = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        
        self.Mask = IdentityModule()
        self.AttentionScore = IdentityModule()
        self.HeadOutputs = IdentityModule()

        nn.init.normal_(self.W_K1, std=self.cfg.init_range)
        nn.init.normal_(self.W_K2, std=self.cfg.init_range)
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_V12, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.register_buffer("IGNORE", t.tensor(-1e6, dtype=t.float32, device=device))


    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        # Assuming self.W_Q, self.W_K, self.W_V, and self.W_O are parameter matrices of the model
        bs, ts, ds = normalized_resid_pre.shape
        k1 = t.einsum('ndh,bpd->bpnh', self.W_K1, normalized_resid_pre) + self.b_K1
        k2 = t.einsum('ndh,bpd->bpnh', self.W_K2, normalized_resid_pre) + self.b_K2
        q = t.einsum('ndh,bpd->bpnh', self.W_Q, normalized_resid_pre) + self.b_Q

        v1 = normalized_resid_pre.unsqueeze(2).expand(-1,-1,ts,-1)
        v2 = normalized_resid_pre.unsqueeze(1).expand(-1,ts,-1,-1)
        v12 = t.cat((v1,v2), dim=-1)
        v = t.einsum('ndh,bpsd->bpsnh', self.W_V12, v12)
        v += self.b_V12
        v = einops.rearrange(v, "b p s n h -> b n h (p s)")
        
        attn_score = t.einsum("bsnh, btnh, bqnh -> bnstq", k1,k2,q)
        
        if self.cfg.causal_attn:
            attn_score = self.apply_causal_mask(attn_score)
        attn_score = einops.rearrange(attn_score, "b n s t q -> b n q (s t)")/self.cfg.d_head
        
        # if self.cfg.causal_attn:  # Let the first token focus on a dummy value
        #     dummy_logit = -1000*t.ones((bs,self.cfg.n_heads,ts, 1), device=attn_score.device)
        #     attn_score = t.cat((attn_score, dummy_logit), dim=-1)
            
        #     dummy_value = t.zeros((bs, self.cfg.n_heads, self.cfg.d_head, 1), device=v.device)
        #     v = t.cat((v, dummy_value), dim=-1)
    
        attn_score = attn_score.softmax(dim=-1)
        attn_score = self.AttentionScore(attn_score)
        z = t.einsum('bnql, bnhl -> bqnh', attn_score, v)
        zint =  self.HeadOutputs(t.einsum('bpnh, nhd->bpnd', z, self.W_O))
        out = einops.reduce(zint,"b p n d -> b p d", reduction='sum') + self.b_O
        return out#, z


    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads a_pos b_pos c_pos"]
    ) -> Float[Tensor, "batch n_heads a_pos b_pos c_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        b, nn, tt, s, q = attn_scores.shape

        t_indices = t.arange(tt).unsqueeze(0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Shape: (1, t, 1, 1)
        s_indices = t.arange(s).unsqueeze(0).unsqueeze(0).unsqueeze(1).unsqueeze(-1)  # Shape: (1, 1, s, 1)
        q_indices = t.arange(q).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)   # Shape: (1, 1, 1, q)

        if self.cfg.attn_eq:
            mask = (t_indices > q_indices) | (s_indices > q_indices)
            if self.cfg.order_attn:
                mask = mask | (t_indices >= s_indices)
            else:
                mask = mask | (t_indices == s_indices)
        else:
            if self.cfg.order_attn:
                mask = (t_indices >= q_indices) | (s_indices >= q_indices) | (t_indices >= s_indices)
            else:
                mask = (t_indices >= q_indices) | (s_indices >= q_indices) | (t_indices == s_indices)
        mask = mask.to(attn_scores.device)
        mask = self.Mask(mask)

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores
