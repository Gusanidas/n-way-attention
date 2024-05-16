import einops
import time
from einops import rearrange, reduce, repeat, pack, unpack
import math
import torch.nn.functional as F
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
from utils_misc import softmax, look_around, pad_to_multiple
from cfgs import Config
from torch.utils.checkpoint import checkpoint



class LocalTrittention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config, freqs_cis: t.Tensor = None):
        super().__init__()
        self.cfg = cfg

        self.abcde = nn.Linear(cfg.d_model, 5*cfg.d_head*cfg.n_heads)
        self.W_O = nn.Linear(cfg.d_head*cfg.n_heads, cfg.d_model)
        self.autopad = getattr(cfg, 'autopad', True)
        self.window_size = getattr(cfg, 'window_size', 16)
        self.look_backward = getattr(cfg, 'look_backward', 1)
        self.pad_value = getattr(cfg, 'pad_value', 0)
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.causal_mask = self.get_causal_mask(self.cfg.n_ctx).to(self.device)
        self.freqs_cis = freqs_cis

        self.register_buffer("IGNORE", t.tensor(-1e6, dtype=t.float32, device=self.device))


    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        bs, ts, ds = normalized_resid_pre.shape
        abcde = self.abcde(normalized_resid_pre)

        abcde = rearrange(abcde, 'b n (x h d) -> x b h n d', h=self.cfg.n_heads, x=5)
        a, b, c, d, e = abcde[0], abcde[1], abcde[2], abcde[3], abcde[4]
        (a, packed_shape), (b, _), (c, _), (d, _), (e, _) = map(lambda t: pack([t], '* n d'), (a, b, c, d, e))

        if self.autopad:
            orig_seq_len = a.shape[1]
            (needed_pad, a), (_, b), (_, c), (_, d), (_, e) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (a, b, c, d, e))
        
        a, b, c, d, e = map(lambda t: rearrange(t, 'b (n w) d -> b n w d', w = self.window_size), (a, b, c, d, e))

        def compute_z(a,b,c,d,e):
            look_around_kwargs = dict(backward=self.cfg.look_backward, forward=0, pad_value=self.pad_value)

            attn_pattern = t.einsum('b h n d, b h m d, b h l d -> b h n m l', c, look_around(a, **look_around_kwargs), look_around(b, **look_around_kwargs))
            attn_pattern.masked_fill_(self.causal_mask, self.IGNORE)
            attn_pattern[attn_pattern ==0] = self.IGNORE
            attn_pattern_shape = attn_pattern.shape
            attn_pattern = rearrange(attn_pattern, 'b h n m l -> b h n (m l)')
            attn_pattern = attn_pattern / self.cfg.d_head
            attn_score = F.softmax(attn_pattern, dim = -1)
            attn_score = t.reshape(attn_score, attn_pattern_shape)

            z = t.einsum('b h n m l, b h m d -> b h n d', attn_score, look_around(d, **look_around_kwargs))
            z += t.einsum('b h n m l, b h l d -> b h n d', attn_score, look_around(e, **look_around_kwargs))
            return z
        z = checkpoint(compute_z, a, b, c, d, e, use_reentrant=False)
        z = rearrange(z, '(b h) w l d ->b (w l) (h d)', h=self.cfg.n_heads, b = bs)
        if self.autopad:
            z = z[:, :orig_seq_len, ...]

        out = self.W_O(z)
        return out

    def get_causal_mask(self, ts):
        seq = t.arange(ts, device=self.device)
        windows = ts // self.window_size 
        b_t = rearrange(seq, '(w n) -> 1 w n', w=windows, n=self.window_size)
        look_around_kwargs = dict(backward=self.cfg.look_backward, forward=0, pad_value=self.pad_value)

        bb_t = look_around(b_t, **look_around_kwargs)
        ba_t = look_around(b_t, **look_around_kwargs)
        causal_mask = ((rearrange(b_t, '... i -> ... i 1 1') < rearrange(bb_t, '... k -> ... 1 1 k')) |
                       (rearrange(bb_t, '... k -> ... 1 1 k') <= rearrange(ba_t, '... j -> ... 1 j 1')))
        return causal_mask

    def forward_2(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        bs, ts, ds = normalized_resid_pre.shape
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

        abcde = self.abcde(normalized_resid_pre)

        a, b, c, d, e = abcde.chunk(5, dim=-1)
        a, b, c, d, e = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.cfg.n_heads), (a, b, c, d, e))

        (a, packed_shape), (b, _), (c, _), (d, _), (e, _) = map(lambda t: pack([t], '* n d'), (a, b, c, d, e))

        if self.autopad:
            orig_seq_len = a.shape[1]
            (needed_pad, a), (_, b), (_, c), (_, d), (_, e) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (a, b, c, d, e))
        
        a, b, c, d, e = map(lambda t: rearrange(t, 'b (n w) d -> b n w d', w = self.window_size), (a, b, c, d, e))

        windows = ts // self.window_size 

        seq = t.arange(ts, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = self.window_size)

        look_around_kwargs = dict(
            backward =  self.look_backward,
            forward = 0,
            pad_value = self.pad_value
        )

        a, b, d, e = map(lambda t: look_around(t, **look_around_kwargs), (a, b, d, e))

        bc_t = b_t
        bb_t = look_around(b_t, **look_around_kwargs)
        ba_t = look_around(b_t, **look_around_kwargs)

        bc_t = rearrange(bc_t, '... i -> ... i 1 1')
        ba_t = rearrange(ba_t, '... j -> ... 1 j 1')
        bb_t = rearrange(bb_t, '... k -> ... 1 1 k')
        causal_mask = (bc_t < bb_t) | (bb_t <= ba_t)
        causal_mask = causal_mask.to(device)

        attn_pattern = t.einsum('b h n d, b h m d, b h l d -> b h n m l', c, a, b)
        attn_pattern[attn_pattern ==0] = self.IGNORE
        attn_pattern.masked_fill_(causal_mask, self.IGNORE)
        attn_pattern_shape = attn_pattern.shape
        attn_pattern = rearrange(attn_pattern, 'b h n m l -> b h n (m l)')
        attn_pattern = attn_pattern / self.cfg.d_head
        attn_score = F.softmax(attn_pattern, dim = -1)
        attn_score = t.reshape(attn_score, attn_pattern_shape)

        #z = t.einsum('b h n m l, b h m d, b h l d -> b h n d', attn_score, d, e)
        z = t.einsum('b h n m l, b h m d -> b h n d', attn_score, d) + t.einsum('b h n m l, b h l d -> b h n d', attn_score, e)
        z = rearrange(z, 'b h l d -> b (h l) d')
        z = rearrange(z, '(b h) t d -> b h t d', h = self.cfg.n_heads, b = bs)
        zout = rearrange(z, 'b h t d -> b t (h d)')
        if self.autopad:
            zout = zout[:, :orig_seq_len, ...]

        out = self.W_O(zout)
        #z = rearrange(z, "b h t d -> b t h d")
        return out


    def slow_tri(self, normalized_resid_pre, slow=False):
        '''
        Slower implementation to sanity check.
        '''
        t0 = time.time()
        bs, ts, ds = normalized_resid_pre.shape
        out = t.zeros((bs, ts, self.cfg.n_heads, self.cfg.d_head))

        a,b,c,d,e = self.abcde(normalized_resid_pre).chunk(5, dim=-1)
        A,B,C,D,E = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = self.cfg.n_heads), (a, b, c, d, e))
        window_size = self.window_size
        prev_window = 0

        for batch in range(bs):
            for head in range(self.cfg.n_heads):
                for t1 in range(ts):
                    if t1%40 == 0:
                        print(f"batch = {batch}, head = {head}, t1 = {t1}, t = {time.time()-t0}, prev_window = {prev_window}")
                    c = C[batch, t1, head, :]
                    scores, vectors = [], []
                    prev_window = max(window_size*(t1 // window_size- self.look_backward), 0)
                    for t2 in range(prev_window, t1+1):
                        for t3 in range(prev_window, t2):
                            b = B[batch, t2, head, :]
                            a = A[batch, t3, head, :]
                            if slow:
                                score = self.slow_tri_dot(a,b,c)
                            else:
                                score = (a*b*c).sum().detach()
                            v = D[batch, t3, head, :] + E[batch, t2, head, :]
                            if t2 == t3 or t2>t1 or t3>=t1:
                                pass
                            else:
                                scores.append(score)
                                vectors.append(v)

                    scores = [s.cpu()/self.cfg.d_head for s in scores]
                    scores = softmax(scores)
                    for j in range(len(scores)):
                        out[batch, t1, head, :] += scores[j]*vectors[j]
            
        return out


    def slow_tri_dot(self, a, b, c):
        assert a.shape == b.shape == c.shape

        h = a.shape[0]
        sum = 0
        for hi in range(h):
            sum += a[hi]*b[hi]*c[hi]
        return sum.detach()


if __name__ == "__main__":
    cfg = Config(
        d_model = 256,
        n_heads = 8,
        d_head = 32,
        window_size = 16,
        look_backward = 1,
        pad_value = 0
    )
    model = LocalTrittention(cfg)
    model.to(model.device)
    model.eval()
    x = t.randn(8, 1024, cfg.d_model)
    out2, *xx = model.forward_2(x)
    out, *xx = model.forward(x)
    print("++++===++++++++=====+++++=====+++++")
    print("++++===++++++++=====+++++=====+++++")
    print(t.allclose(out, out2, atol=1e-3))
    # mean square difference
    print(f"Mean square difference = {(out - out2).abs().mean()}")
    print("++++===++++++++=====+++++=====+++++")
    print("++++===++++++++=====+++++=====+++++")
    # compate execution time:
    import time
    t0 = time.time()
    for _ in range(10):
        out, *xx = model.forward(x)
    out, *xx = model.forward(x)
    print(f"Time for forward = {time.time()-t0}")
    t0 = time.time()
    for _ in range(10):
        out2, *xx = model.forward_2(x)
    out2, *xx = model.forward_2(x)
    print(f"Time for forward_2 = {time.time()-t0}")
