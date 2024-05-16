
from dataclasses import dataclass, asdict, fields


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
    mlp_type: str = "all"
    with_ln: bool = True
    order_attn: bool = True
    attn_eq: bool = False
    window_size: int = 16
    look_backward: int = 1
    pad_value: int = 0

    def to_dict(self) -> dict:
        """Converts the dataclass instance to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = {field.name for field in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)