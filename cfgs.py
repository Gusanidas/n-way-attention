
from dataclasses import dataclass
from enum import Enum


class MLP_TYPE(Enum):
    NONE = 0 
    LAST = 1 
    ALL = 2  

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.05
    mlp_type: MLP_TYPE = MLP_TYPE.ALL