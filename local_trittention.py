import einops
from einops import rearrange, reduce, repeat, pack, unpack
import math
import torch.nn.functional as F
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float
from utils_misc import softmax
from cfgs import Config



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
        self.autopad = True
        self.window_size = 16
        self.look_backward = 1
        self.pad_value = 0

        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.register_buffer("IGNORE", t.tensor(-1e6, dtype=t.float32, device=self.device))


    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        bs, ts, ds = normalized_resid_pre.shape
        device = self.device
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

        attn_pattern = t.einsum('b h n d, b h m d, b h l d -> b h n m l', c, a, b)
        attn_pattern.masked_fill_(causal_mask.to(device), self.IGNORE)
        attn_pattern[attn_pattern ==0] = self.IGNORE
        attn_pattern_shape = attn_pattern.shape
        attn_pattern = rearrange(attn_pattern, 'b h n m l -> b h n (m l)')
        attn_pattern = attn_pattern / self.cfg.d_head
        attn_score = F.softmax(attn_pattern, dim = -1)
        attn_score = t.reshape(attn_score, attn_pattern_shape)

        z = t.einsum('b h n m l, b h m d -> b h n d', attn_score, d) + t.einsum('b h n m l, b h l d -> b h n d', attn_score, e)
        z = rearrange(z, 'b h l d -> b (h l) d')
        z = rearrange(z, '(b h) t d -> b h t d', h = self.cfg.n_heads, b = bs)
        z = rearrange(z, 'b h t d -> b t (h d)')
        if self.autopad:
            z = z[:, :orig_seq_len, ...]

        out = self.W_O(z)
        return out, z
        

    def slow_tri(self, normalized_resid_pre):
        '''
        Slower implementation to sanity check.
        '''
        bs, ts, ds = normalized_resid_pre.shape
        out = t.zeros((bs, ts, self.cfg.n_heads, self.cfg.d_head))

        a,b,c,d,e = self.abcde(normalized_resid_pre).chunk(5, dim=-1)
        A,B,C,D,E = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = self.cfg.n_heads), (a, b, c, d, e))
        window_size = self.window_size

        for batch in range(bs):
            for head in range(self.cfg.n_heads):
                for t1 in range(ts):
                    c = C[batch, t1, head, :]
                    scores, vectors = [], []
                    prev_window = max(t1 // window_size- self.look_backward*window_size, 0)
                    for t2 in range(prev_window, t1+1):
                        for t3 in range(prev_window, t2):
                            b = B[batch, t2, head, :]
                            a = A[batch, t3, head, :]
                            score = self.slow_tri_dot(a,b,c)
                            v = D[batch, t2, head, :] + E[batch, t3, head, :]
                            if t2 == t3 or t2>=t1 or t3>=t1:
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