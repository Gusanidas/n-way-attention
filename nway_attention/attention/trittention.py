import einops
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float

from nway_attention.utils_misc import IdentityModule
from nway_attention.cfgs import Config


class Trittention(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.kkq = nn.Linear(cfg.d_model, cfg.d_head*cfg.n_heads*3)
        self.V12 = nn.Linear(cfg.d_model*2, cfg.d_head*cfg.n_heads)

        self.Out = nn.Linear(cfg.d_head*cfg.n_heads, cfg.d_model)
        self.Mask = IdentityModule()
        self.AttentionScore = IdentityModule()
        self.HeadOutputs = IdentityModule()
        self.register_buffer("IGNORE", t.tensor(-1e6, dtype=t.float32))
        self.register_buffer('precomputed_mask', self.create_causal_mask(cfg.n_ctx))

    def create_causal_mask(self, max_seq_len):
        t_indices = t.arange(max_seq_len).unsqueeze(0).unsqueeze(1).unsqueeze(-1)
        s_indices = t.arange(max_seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(1)
        q_indices = t.arange(max_seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask = (t_indices > q_indices) | (s_indices > q_indices)
        return mask

    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        # Assuming self.W_Q, self.W_K, self.W_V, and self.W_O are parameter matrices of the model
        bs, ts, ds = normalized_resid_pre.shape


        k1, k2, q = self.kkq(normalized_resid_pre).chunk(3, dim=-1)
        k1, k2, q = map(lambda t: einops.rearrange(t, 'b p (h d) -> b p h d', h=self.cfg.n_heads), (k1, k2, q))
        v12 = t.cat((normalized_resid_pre.unsqueeze(2).expand(-1,-1,ts,-1),
                     normalized_resid_pre.unsqueeze(1).expand(-1,ts,-1,-1)), dim=-1)
        v = self.V12(v12)
        v = einops.rearrange(v, "b p s (n h) -> b n h (p s)", n=self.cfg.n_heads)
        attn_score = t.einsum("bsnh, btnh, bqnh -> bnstq", k1,k2,q)
        
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
        mask = self.precomputed_mask[:, :tt, :s, :q].to(attn_scores.device)
        mask = self.Mask(mask)
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores
