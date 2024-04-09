import einops
import torch as t
from torch import Tensor
from typing import Type
import torch.nn as nn
from jaxtyping import Float, Int
from trittention import Trittention
from trittention_cube import TrittentionCube
from attention import Attention
from cfgs import Config, MLP_TYPE

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
        self.attn = Attention(cfg)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.has_mlp = has_mlp
        if self.has_mlp:
            self.linear1 = nn.Linear(cfg.d_model, cfg.d_mlp)
            self.dropout = nn.Dropout(cfg.dropout)
            self.linear2 = nn.Linear(cfg.d_mlp, cfg.d_model)
            self.gelu = nn.GELU()

            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(
        self, resid: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        resid = self.dropout1(self.attn(self.ln1(resid))) + resid
        if self.has_mlp:
            resid = self.dropout2(self.linear2(self.gelu(self.linear1(resid)))) + resid
        
        return resid

class TriformerBlock(TransformerBlock):
    def __init__(self, cfg: Config, has_mlp: bool = True):
        super().__init__(cfg, has_mlp=has_mlp)
        self.attn = Trittention(cfg)

class TriformerCubeBlock(TransformerBlock):
    def __init__(self, cfg: Config, has_mlp: bool = True):
        super().__init__(cfg, has_mlp=has_mlp)
        self.attn = TrittentionCube(cfg)


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = self._get_blocks(TransformerBlock)
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = Unembed(cfg)

    def _get_blocks(self, block: Type[TransformerBlock]) -> nn.ModuleList:
        if self.cfg.mlp_type == MLP_TYPE.NONE:
            blocks = nn.ModuleList([block(self.cfg, has_mlp=False) for _ in range(self.cfg.n_layers)])
        elif self.cfg.mlp_type == MLP_TYPE.LAST:
            blocks = nn.ModuleList([block(self.cfg, has_mlp=False) for _ in range(self.cfg.n_layers - 1)] +
                                   [block(self.cfg, has_mlp=True)])
        else:  # Assuming MLP_TYPE.ALL or any other case defaults to this
            blocks = nn.ModuleList([block(self.cfg, has_mlp=True) for _ in range(self.cfg.n_layers)])
        return blocks

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits

class Triformer(Transformer):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.blocks = self._get_blocks(TriformerBlock)

class TriformerCube(Transformer):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.blocks = self._get_blocks(TriformerCubeBlock)
