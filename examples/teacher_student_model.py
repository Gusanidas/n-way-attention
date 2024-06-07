import torch.nn as nn

from nway_attention.cfgs import Config
from nway_attention.attention.trittention_cube import TrittentionCube
from nway_attention.attention.trittention import Trittention
from nway_attention.attention.attention import Attention
from nway_attention.modules.transformer_models import TransformerBlock, Transformer
from nway_attention.train import train_student_with_teacher


model_cfg = Config(
    d_model=256,
    debug=True,
    layer_norm_eps=1e-5,
    d_vocab=8192,
    init_range=0.02,
    n_ctx=42,
    d_head=32,           
    d_mlp=288,          
    n_heads=8,         
    n_layers=3,       
    mlp_type="all",
    order_attn = True,
    attn_eq=True,
    with_ln=True,
    causal_attn=True,
)

def compare():
    stb1 = nn.Sequential(*[TransformerBlock(model_cfg, has_mlp=False) for _ in range(4)])
    stri = nn.Sequential(*[TransformerBlock(model_cfg, has_mlp=False) for _ in range(2)])
    train_student_with_teacher(stri, stb1, 256, num_epochs=702)
    stb1 = nn.Sequential(*[TransformerBlock(model_cfg, has_mlp=False) for _ in range(4)])
    stri = nn.Sequential(*[TransformerBlock(model_cfg, has_mlp=False) for _ in range(2)])
    train_student_with_teacher(stb1, stri, 256)

if __name__ == '__main__':
    compare()
