import torch as t
import torch.nn as nn

from nway_attention.cfgs import Config
from .attention import Attention
from .trittention import Trittention
from .trittention_cube import TrittentionCube

class MixedAttention(nn.Module):

    def __init__(self, cfg: Config, cube: bool = False):
        super().__init__()
        self.cfg = cfg
        self.attention = Attention(cfg)
        tritt_cfg = Config.from_dict(cfg.to_dict())
        tritt_cfg.d_head = cfg.dt_head
        tritt_cfg.n_heads = cfg.nt_heads
        if cfg.nt_heads <1:
            self.trittention = nn.Identity()
        elif cube:
            self.trittention = TrittentionCube(tritt_cfg)
        else:
            self.trittention = Trittention(tritt_cfg)

    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        return self.attention(normalized_resid_pre) + self.trittention(normalized_resid_pre)