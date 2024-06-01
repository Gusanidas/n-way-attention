import einops
from einops import rearrange, reduce, repeat, pack, unpack
import math
import torch.nn.functional as F
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
from utils_misc import apply_rotary_emb, pad_to_multiple, look_around
from examples.cfgs import Config
from local_trittention import LocalTrittention
from attention import Attention
from torch.utils.checkpoint import checkpoint



class MixedAttention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config, freqs_cis: t.Tensor = None):
        super().__init__()
        self.cfg = cfg
        self.freqs_cis = freqs_cis
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.pad_value = getattr(cfg, 'pad_value', 0)
        self.autopad = getattr(cfg, 'autopad', True)
        self.use_reentrant = getattr(cfg, 'use_reentrant', False)
        self.causal_mask = self.get_causal_mask(self.cfg.n_ctx).to(self.device)
        self.ts_cm = self.cfg.n_ctx
        self.checkpoint = getattr(cfg, 'checkpoint',False)

        self.qkv = nn.Linear(cfg.d_model, 3*cfg.d_head*cfg.n_heads)
        self.abcde = nn.Linear(cfg.d_model, 5*cfg.dt_head*cfg.nt_heads) if cfg.nt_heads > 0 else None
        self.out_p = nn.Linear(cfg.d_head*cfg.n_heads + cfg.dt_head*cfg.nt_heads, cfg.d_model)

        self.IGNORE = t.tensor(-1e6, dtype=t.float32, device=self.device)

    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        b, ts, ds = normalized_resid_pre.shape
        #recompute the mask
        if 2*self.cfg.window_size != self.causal_mask.shape[-1] or ts != self.ts_cm:
            self.causal_mask = self.get_causal_mask(ts).to(self.device)
        q, k, v = self.qkv(normalized_resid_pre).chunk(3, dim=-1)
        q, k ,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.cfg.n_heads), (q, k, v))
        q, k = apply_rotary_emb(q, freqs_cis = self.freqs_cis), apply_rotary_emb(k, freqs_cis = self.freqs_cis)
        z_attn = t.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.cfg.dropout if self.training else 0, is_causal=True)
        z_attn = rearrange(z_attn, 'b h n d -> b n (h d)')

        if self.abcde is not None:
            abcde = self.abcde(normalized_resid_pre)
            abcde = rearrange(abcde, 'b n (x h d) -> x b h n d', h=self.cfg.nt_heads, x=5)
            a, b, c, d, e = abcde[0], abcde[1], abcde[2], abcde[3], abcde[4]
            a, b, c = apply_rotary_emb(a, freqs_cis = self.freqs_cis), apply_rotary_emb(b, freqs_cis = self.freqs_cis), apply_rotary_emb(c, freqs_cis = self.freqs_cis)
            if self.checkpoint:
                z_tri = checkpoint(self.compute_tri_z, a, b, c, d, e, use_reentrant=self.use_reentrant)
            else:
                z_tri = self.compute_tri_z(a, b, c, d, e)
            z_attn = t.cat((z_attn, z_tri), dim=-1)
        
        out = self.out_p(z_attn)
        return out

    
    def compute_tri_z(self, a,b,c,d,e):
        (a, packed_shape), (b, _), (c, _), (d, _), (e, _) = map(lambda t: pack([t], '* n d'), (a, b, c, d, e))
        if self.autopad:
            orig_seq_len = a.shape[1]
            (needed_pad, a), (_, b), (_, c), (_, d), (_, e) = map(lambda t: pad_to_multiple(t, self.cfg.window_size, dim = -2), (a, b, c, d, e))
        a, b, c, d, e = map(lambda t: rearrange(t, 'b (n w) d -> b n w d', w = self.cfg.window_size), (a, b, c, d, e))  
        look_around_kwargs = dict(backward=self.cfg.look_backward, forward=0, pad_value=self.pad_value)
        attn_pattern = t.einsum('b h n d, b h m d, b h l d -> b h n m l', c, look_around(a, **look_around_kwargs), look_around(b, **look_around_kwargs))
        attn_pattern.masked_fill_(self.causal_mask, self.IGNORE)
        attn_pattern[attn_pattern ==0] = self.IGNORE
        attn_pattern_shape = attn_pattern.shape
        attn_pattern = rearrange(attn_pattern, 'b h n m l -> b h n (m l)')
        attn_pattern = attn_pattern / self.cfg.dt_head
        attn_score = F.softmax(attn_pattern, dim = -1)
        attn_score = t.reshape(attn_score, attn_pattern_shape)
        z = t.einsum('b h n m l, b h m d -> b h n d', attn_score, look_around(d, **look_around_kwargs))
        z += t.einsum('b h n m l, b h l d -> b h n d', attn_score, look_around(e, **look_around_kwargs))
        z = rearrange(z, '(b h) w l d ->b (w l) (h d)', h=self.cfg.nt_heads)
        if self.autopad:
            z = z[:, :orig_seq_len, ...]
        return z

    def get_causal_mask(self, ts):
        self.ts_cm = ts
        seq = t.arange(ts, device=self.device)
        windows = ts // self.cfg.window_size 
        b_t = rearrange(seq, '(w n) -> 1 w n', w=windows, n=self.cfg.window_size)
        look_around_kwargs = dict(backward=self.cfg.look_backward, forward=0, pad_value=self.pad_value)

        bb_t = look_around(b_t, **look_around_kwargs)
        ba_t = look_around(b_t, **look_around_kwargs)
        causal_mask = ((rearrange(b_t, '... i -> ... i 1 1') < rearrange(bb_t, '... k -> ... 1 1 k')) |
                       (rearrange(bb_t, '... k -> ... 1 1 k') <= rearrange(ba_t, '... j -> ... 1 j 1')))
        return causal_mask
    
if __name__ == '__main__':
    from examples.cfgs import Config
    from utils_misc import precompute_freqs_cis
    cfg = Config()
    model = MixedAttention(cfg, freqs_cis=precompute_freqs_cis(cfg.d_head, cfg.n_ctx))
    x = t.randn(12, cfg.n_ctx, cfg.d_model)
    y = model(x)
    print(y.shape)
    cfg = Config(dt_head=64, nt_heads=0)
    model = MixedAttention(cfg, freqs_cis=precompute_freqs_cis(cfg.d_head, cfg.n_ctx))
    x = t.randn(12, cfg.n_ctx, cfg.d_model)
    y = model(x)
    print(y.shape)
