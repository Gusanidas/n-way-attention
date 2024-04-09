import einops
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
from jaxtyping import Float, Int
from cfgs import Config

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