import einops
import time
from einops import rearrange, reduce, repeat, pack, unpack
import math
import torch.nn.functional as F
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
from utils_misc import softmax
from cfgs import Config
from torch.utils.checkpoint import checkpoint


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t_s = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + t_s), ...] for ind in range(forward + backward + 1)]
    return t.cat(tensors, dim = dim)

class LocalTrittention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.abcde = nn.Linear(cfg.d_model, 5*cfg.d_head*cfg.n_heads)
        self.W_O = nn.Linear(cfg.d_head*cfg.n_heads, cfg.d_model)
        self.autopad = getattr(cfg, 'autopad', True)
        self.window_size = getattr(cfg, 'window_size', 16)
        self.look_backward = getattr(cfg, 'look_backward', 1)
        self.pad_value = getattr(cfg, 'pad_value', 0)

        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.register_buffer("IGNORE", t.tensor(-1e6, dtype=t.float32, device=self.device))


    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        bs, ts, ds = normalized_resid_pre.shape
        device = self.device
        abcde = self.abcde(normalized_resid_pre)

        abcde = rearrange(abcde, 'b n (x h d) -> x b h n d', h=self.cfg.n_heads, x=5)
        a, b, c, d, e = abcde[0], abcde[1], abcde[2], abcde[3], abcde[4]
        (a, packed_shape), (b, _), (c, _), (d, _), (e, _) = map(lambda t: pack([t], '* n d'), (a, b, c, d, e))

        if self.autopad:
            orig_seq_len = a.shape[1]
            (needed_pad, a), (_, b), (_, c), (_, d), (_, e) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (a, b, c, d, e))
        
        a, b, c, d, e = map(lambda t: rearrange(t, 'b (n w) d -> b n w d', w = self.window_size), (a, b, c, d, e))

        windows = ts // self.window_size 

        def compute_z(a,b,c,d,e):
            seq = t.arange(ts, device=device)
            b_t = rearrange(seq, '(w n) -> 1 w n', w=windows, n=self.window_size)
            look_around_kwargs = dict(backward=self.look_backward, forward=0, pad_value=self.pad_value)

            bb_t = look_around(b_t, **look_around_kwargs)
            ba_t = look_around(b_t, **look_around_kwargs)
            causal_mask = ((rearrange(b_t, '... i -> ... i 1 1') < rearrange(bb_t, '... k -> ... 1 1 k')) |
                           (rearrange(bb_t, '... k -> ... 1 1 k') <= rearrange(ba_t, '... j -> ... 1 j 1')))

            attn_pattern = t.einsum('b h n d, b h m d, b h l d -> b h n m l', c, look_around(a, **look_around_kwargs), look_around(b, **look_around_kwargs))
            attn_pattern.masked_fill_(causal_mask.to(device), self.IGNORE)
            attn_pattern[attn_pattern ==0] = self.IGNORE
            attn_pattern_shape = attn_pattern.shape
            attn_pattern = rearrange(attn_pattern, 'b h n m l -> b h n (m l)')
            attn_pattern = attn_pattern / self.cfg.d_head
            attn_score = F.softmax(attn_pattern, dim = -1)
            attn_score = t.reshape(attn_score, attn_pattern_shape)

            z = t.einsum('b h n m l, b h m d -> b h n d', attn_score, look_around(d, **look_around_kwargs))
            z += t.einsum('b h n m l, b h l d -> b h n d', attn_score, look_around(e, **look_around_kwargs))
            return z
        z = checkpoint(compute_z, a, b, c, d, e, use_reentrant=True)
        z = rearrange(z, '(b h) w l d ->b (w l) (h d)', h=self.cfg.n_heads, b = bs)
        if self.autopad:
            z = z[:, :orig_seq_len, ...]

        out = self.W_O(z)
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
    out2, z2, *xx = model.forward_2(x)
    out, z, *xx = model.forward(x)
    z = rearrange(z, 'b t (h d) -> b t h d', h=cfg.n_heads)
    print("++++===++++++++=====+++++=====+++++")
    print("++++===++++++++=====+++++=====+++++")
    print(t.allclose(z, z2, atol=1e-3))
    # mean square difference
    print(f"Mean square difference = {(z - z2).abs().mean()}")
    print("++++===++++++++=====+++++=====+++++")
    print("++++===++++++++=====+++++=====+++++")
    # compate execution time:
    import time
    t0 = time.time()
    for _ in range(10):
        out, z, *xx = model.forward(x)
    out, z, *xx = model.forward(x)
    print(f"Time for forward = {time.time()-t0}")
    t0 = time.time()
    for _ in range(10):
        out2, z2, *xx = model.forward_2(x)
    out2, z2, *xx = model.forward_2(x)
    print(f"Time for forward_2 = {time.time()-t0}")
