import einops
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float

from nway_attention.utils_misc import softmax
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

        #print(f"shpae of de {de.shape}, shape of DE {self.W_V12.shape}, bias = {self.b_V12.shape}")

        v = t.einsum('ndh,bpsd->bpsnh', self.W_V12, v12)
        v += self.b_V12

        attn_score = t.einsum("bsnh, btnh, bqnh -> bnstq", k1,k2,q)
        attn_score = self.apply_causal_mask(attn_score)


        attn_score = einops.rearrange(attn_score, "b n s t q -> b n q (s t)")/self.cfg.d_head
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

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

    def slow_tri(self, normalized_resid_pre):
        '''
        Slower implementation to sanity check.
        '''
        bs, ts, ds = normalized_resid_pre.shape
        out = t.zeros((bs, ts, self.cfg.n_heads, self.cfg.d_head))

        for bi in range(bs):
            for i in range(self.cfg.n_heads):
                for t1 in range(ts):
                    c = normalized_resid_pre[bi,t1,:] @ self.W_Q[i,:,:] + self.b_Q[i,:]
                    scores, vectors = [], []
                    for t2 in range(ts):
                        for t3 in range(ts):
                            b = normalized_resid_pre[bi,t2,:] @ self.W_K2[i,:,:] + self.b_K2[i,:]
                            a = normalized_resid_pre[bi,t3,:] @ self.W_K1[i,:,:] + self.b_K1[i,:]
                            score = self.slow_tri_dot(a,b,c)
                            cv = t.cat((normalized_resid_pre[bi, t3,:], normalized_resid_pre[bi, t2,:]), dim=-1)
                            v = cv @ self.W_V12[i,:,:] + self.b_V12[i,:]
                            if t2 == t3 or t2>=t1 or t3>=t1:
                                pass
                            else:
                                scores.append(score)
                                vectors.append(v)

                    scores = [s.cpu()/self.cfg.d_head for s in scores]
                    scores = softmax(scores)
                    for j in range(len(scores)):
                        out[bi, t1, i, :] += scores[j]*vectors[j]
        return out


    def slow_tri_dot(self, a, b, c):
        assert a.shape == b.shape == c.shape

        h = a.shape[0]
        sum = 0
        for hi in range(h):
            sum += a[hi]*b[hi]*c[hi]
        return sum.detach()
