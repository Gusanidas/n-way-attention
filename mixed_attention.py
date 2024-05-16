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
from local_trittention import LocalTrittention
from attention import Attention



class MixedAttention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.local_trittention = LocalTrittention(cfg) 
        self.attention = Attention(cfg)

    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        local = self.local_trittention(normalized_resid_pre)
        global_ = self.attention(normalized_resid_pre)
        return local + global_
