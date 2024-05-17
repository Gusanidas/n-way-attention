import einops
import torch.nn.functional as F
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
from jaxtyping import Float, Int
from cfgs import Config
import math

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"],
    tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return log_probs_for_tokens

def get_log_probs_last(
    logits: Float[Tensor, "batch posn d_vocab"],
    tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    log_probs_for_tokens = log_probs[:, -2].gather(dim=-1, index=tokens[:, -1].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens

def pad_cat(tensors):
    max_width = max([x.shape[1] for x in tensors])
    padded_tensors = [nn.functional.pad(tensor, (max_width - tensor.shape[1], 0), 'constant', tensor[0,0]) for tensor in tensors]
    return t.cat(padded_tensors, dim=0)

def softmax(lst):
    if len(lst)<1: return lst
    exp_lst = np.exp(lst - np.max(lst))
    return exp_lst / exp_lst.sum()

def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> t.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(t.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    tt = t.arange(end, device=freqs.device)
    freqs = t.outer(tt, freqs).float()
    freqs_cis = t.polar(t.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: t.Tensor, freqs_cis: t.Tensor) -> t.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    b, _,ts, d = x.shape
    freqs_cis = freqs_cis[:ts, :]
    x_ = t.view_as_complex(
        t.stack(t.chunk(x.float(), 2, dim=-1),
                    dim=-1))
    x_out = t.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = t.cat(t.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1)
    return x_out

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    ts = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + ts), ...] for ind in range(forward + backward + 1)]
    return t.cat(tensors, dim = dim)