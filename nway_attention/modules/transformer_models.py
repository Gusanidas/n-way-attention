import einops
import torch as t
from torch import Tensor
import torch.nn.functional as F
from typing import Type
import torch.nn as nn
from jaxtyping import Float, Int
from huggingface_hub import PyTorchModelHubMixin

from nway_attention.attention.attention import Attention
from nway_attention.attention.trittention import Trittention
from nway_attention.attention.trittention_cube import TrittentionCube
from nway_attention.cfgs import Config
from nway_attention.utils_misc import precompute_freqs_cis
from nway_attention.attention.mixed_attention import MixedAttention

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return einops.einsum(
            normalized_resid_final, self.W_U,
            "batch posn d_model, d_model d_vocab -> batch posn d_vocab",
        ) + self.b_U

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config, has_mlp: bool = True):
        super().__init__()
        self.cfg = cfg
        self.attn = self._get_attention(cfg)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.ln1 = nn.LayerNorm(cfg.d_model) if cfg.with_ln else nn.Identity()
        self.has_mlp = has_mlp
        if self.has_mlp:
            self.linear1 = nn.Linear(cfg.d_model, cfg.d_mlp)
            self.dropout = nn.Dropout(cfg.dropout)
            self.linear2 = nn.Linear(cfg.d_mlp, cfg.d_model)
            self.gelu = nn.GELU()
            if cfg.is_gated:
                self.linearg = nn.Linear(cfg.d_model, cfg.d_mlp)
                self.linear2 = nn.Linear(cfg.d_mlp, cfg.d_model)

            self.ln2 = nn.LayerNorm(cfg.d_model) if cfg.with_ln else nn.Identity()
            self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(
        self, resid: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        resid = self.dropout1(self.attn(self.ln1(resid))) + resid
        if self.has_mlp:
            if self.cfg.is_gated:
                mid_mlp = self.gelu(self.linearg(resid), approximate="tanh")*self.linear1(resid)
            else:
                mid_mlp = self.gelu(self.linear1(resid))
                
            resid = self.dropout2(self.linear2(mid_mlp)) + resid
        
        return resid
    
    def _get_attention(self, cfg, freqs_cis=None):
        if cfg.attn_type.lower() == 'attention':
            return Attention(cfg)
        elif cfg.attn_type.lower() == 'trittention':
            return Trittention(cfg)
        elif cfg.attn_type.lower() == 'trittentioncube':
            return TrittentionCube(cfg)
        elif cfg.attn_type.lower() == 'mixedattention':
            return MixedAttention(cfg)
        else:
            raise ValueError(f"Attention type {cfg.attn_type} not recognized.")

class Transformer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg = Config.from_dict(config)
        self.embed = Embed(self.cfg)
        self.pos_embed = PosEmbed(self.cfg)
        self.blocks = self._get_blocks(TransformerBlock)
        self.ln_final = nn.LayerNorm(self.cfg.d_model) if self.cfg.with_ln else nn.Identity()
        self.unembed = Unembed(self.cfg)

    def _get_blocks(self, block: Type[TransformerBlock], **kwargs) -> nn.ModuleList:
        if self.cfg.mlp_type.lower() == "none":
            blocks = nn.ModuleList([block(self.cfg, has_mlp=False, **kwargs) for _ in range(self.cfg.n_layers)])
        elif self.cfg.mlp_type.lower() == "last":
            blocks = nn.ModuleList([block(self.cfg, has_mlp=False, **kwargs) for _ in range(self.cfg.n_layers - 1)] +
                                   [block(self.cfg, has_mlp=True, **kwargs)])
        else:  # Assuming ALL or any other case defaults to this
            blocks = nn.ModuleList([block(self.cfg, has_mlp=True, **kwargs) for _ in range(self.cfg.n_layers)])
        return blocks

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits

class TriformerMixedBlock(TransformerBlock):
    def __init__(self, cfg: Config, has_mlp: bool = True, freqs_cis: t.Tensor = None):
        super().__init__(cfg, has_mlp=has_mlp)
        self.attn = MixedAttention(cfg, freqs_cis=freqs_cis)

class TriformerMixed(Transformer):
    def __init__(self, config: dict):
        super().__init__(config)
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        freqs_cis = precompute_freqs_cis(config.get('d_head',64), config.get('n_ctx', 1024)).to(device)
        self.blocks = self._get_blocks(TriformerMixedBlock, freqs_cis=freqs_cis)

if __name__ == "__main__":
    model_cfg = Config(
        d_model = 128,
        debug = True,
        layer_norm_eps = 1e-5,
        d_vocab = 101,
        init_range = 0.02,
        n_ctx = 48,
        d_head = 32,
        dt_head = 32,
        d_mlp = 512,
        n_heads = 4,
        nt_heads = 2,
        n_layers = 4,
        mlp_type="all",
        with_ln=True,
        order_attn=True,
        attn_eq=True,
        attn_type = 'mixedattention',
    )

    model = Transformer(model_cfg.to_dict())
    x = t.randint(0, 101, (32, 48))
    print(f"Input shape: {x.shape}")
    y = model(x)
    print(y.shape)
