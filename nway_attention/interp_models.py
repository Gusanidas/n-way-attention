import einops
import torch as t
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float

from nway_attention.utils_misc import Config
from nway_attention.transformer_models import TransformerBlock, Transformer
from nway_attention.transformer_models import TriformerCubeBlock as TriformerCubeBlockOG
from nway_attention.transformer_models import TrittentionCube as TrittentionCubeOG
from nway_attention.transformer_models import TriformerCube as TriformerCubeOG


class MH_Linear(nn.Module):
    def __init__(self, cfg):
        super(MH_Linear, self).__init__()
        self.cfg = cfg
        self.W = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W, std=self.cfg.init_range)  # Normal initialization with specified std
        self.b = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))

    def forward(self, normalized_resid_pre):
        a = t.einsum('ndh,bpd->bpnh', self.W, normalized_resid_pre) + self.b
        return a

class V_Layer(nn.Module):
    def __init__(self, cfg):
        super(V_Layer, self).__init__()
        self.cfg = cfg
        self.W = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_head, cfg.d_head)))
        nn.init.normal_(self.W, std=self.cfg.init_range)  # Normal initialization with specified std

    def forward(self, d, e):
        v = einops.einsum(d, e, self.W, "b p1 n h1, b p2 n h2, n h1 h2 h3 -> b p1 p2 n h3")
        return v

class TrittentionCScore(nn.Module):
    def __init__(self, cfg):
        super(TrittentionCScore, self).__init__()
        self.cfg = cfg
        self.W = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_head, cfg.d_head)))
        nn.init.normal_(self.W, std=self.cfg.init_range)  # Normal initialization with specified std

    def forward(self, c, b, a):
        step1 = t.einsum('brnk, nijk -> brij', c, self.W)
        step2 = t.einsum('brij, bqnj -> briq', step1, b)
        attn_score = t.einsum('briq, bpni -> bnpqr', step2, a)
        return attn_score

class CausalMask(nn.Module):
    def __init__(self, cfg, IGNORE=-1e6):
        super(CausalMask, self).__init__()
        self.cfg = cfg
        self.IGNORE = IGNORE

    def forward(self, attn_scores):
        b, nn, tt, s, q = attn_scores.shape

        t_indices = t.arange(tt).unsqueeze(0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Shape: (1, t, 1, 1)
        s_indices = t.arange(s).unsqueeze(0).unsqueeze(0).unsqueeze(1).unsqueeze(-1)  # Shape: (1, 1, s, 1)
        q_indices = t.arange(q).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)   # Shape: (1, 1, 1, q)

        if self.cfg.attn_eq:
            mask = (t_indices > q_indices) | (s_indices > q_indices)
            if self.cfg.order_attn:
                mask = mask | (t_indices >= s_indices)
            else:
                mask = mask | (t_indices == s_indices)
        else:
            if self.cfg.order_attn:
                mask = (t_indices >= q_indices) | (s_indices >= q_indices) | (t_indices >= s_indices)
            else:
                mask = (t_indices >= q_indices) | (s_indices >= q_indices) | (t_indices == s_indices)
        mask = mask.to(attn_scores.device)

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

class TrittentionCPattern(nn.Module):
    def __init__(self, cfg):
        super(TrittentionCPattern, self).__init__()
        self.cfg = cfg

    def forward(self, attn_score):
        bs, _, ts, _, _ = attn_score.shape
        attn_score = einops.rearrange(attn_score, "b n p1 p2 p3 -> b n p3 (p1 p2)")/self.cfg.d_head
        extra = -1000*t.ones((bs,self.cfg.n_heads,ts, 1), device=attn_score.device)
        attn_score = t.cat((attn_score, extra), dim=-1)
        attn_score = attn_score.softmax(dim=-1)
        return attn_score

class Z_Layer(nn.Module):
    def __init__(self, cfg):
        super(Z_Layer, self).__init__()
        self.cfg = cfg

    def forward(self, attn_score, v):
        bs, _, ts, _ = attn_score.shape
        v = einops.rearrange(v, "b p s n h -> b n h (p s)")
        extra = t.zeros((bs, self.cfg.n_heads, self.cfg.d_head, 1), device=v.device)
        v = t.cat((v,extra), dim=-1)
        z = t.einsum('bnql, bnhl -> bqnh', attn_score, v)
        return z

class Result_Layer(nn.Module):
    def __init__(self, cfg):
        super(Result_Layer, self).__init__()
        self.cfg = cfg
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)  # Normal initialization with specified std

    def forward(self, z):
        zint =  t.einsum('bpnh,nhd->bpnd', z, self.W_O)
        return zint

class Out_Layer(nn.Module):
    def __init__(self, cfg):
        super(Out_Layer, self).__init__()
        self.cfg = cfg
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))

    def forward(self, zint):
        out = einops.reduce(zint,"b p n d -> b p d", reduction='sum') + self.b_O
        return out

class TrittentionCube(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.A = MH_Linear(cfg)
        self.B = MH_Linear(cfg)
        self.C = MH_Linear(cfg)
        self.D = MH_Linear(cfg)
        self.E = MH_Linear(cfg)

        self.V = V_Layer(cfg)

        self.CScore = TrittentionCScore(cfg)
        self.CMask = CausalMask(cfg)
        self.CPattern = TrittentionCPattern(cfg)
        self.Z = Z_Layer(cfg)
        self.Result = Result_Layer(cfg)
        self.Out = Out_Layer(cfg)

        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.register_buffer("IGNORE", t.tensor(-1e6, dtype=t.float32, device=device))

    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        a = self.A(normalized_resid_pre)
        b = self.B(normalized_resid_pre)
        c = self.C(normalized_resid_pre)

        d = self.D(normalized_resid_pre)
        e = self.E(normalized_resid_pre)
        v = self.V(d, e)

        attn_score = self.CScore(c, b, a)
        attn_score = self.CMask(attn_score)
        attn_score = self.CPattern(attn_score)

        z = self.Z(attn_score, v)
        zint = self.Result(z)
        out = self.Out(zint)
        return out 

    def copy_from_og(self, tc):
        self.A.W.data = tc.W_A
        self.A.b.data = tc.b_A
        self.B.W.data = tc.W_B
        self.B.b.data = tc.b_B
        self.C.W.data = tc.W_C
        self.C.b.data = tc.b_C
        self.D.W.data = tc.W_D
        self.D.b.data = tc.b_D
        self.E.W.data = tc.W_E
        self.E.b.data = tc.b_E

        self.V.W.data = tc.W_V

        self.CScore.W.data = tc.W_K

        self.Result.W_O.data = tc.W_O
        self.Out.b_O.data = tc.b_O

        return self
    

class TriformerCubeBlock(TransformerBlock):
    def __init__(self, cfg: Config, has_mlp: bool = True):
        super().__init__(cfg, has_mlp=has_mlp)
        self.attn = TrittentionCube(cfg)


    def copy_from_og(self, source_block):

        # Get named parameters from both blocks
        target_params = dict(self.named_parameters())
        source_params = dict(source_block.named_parameters())

        # Copy parameters, skipping those belonging to the attention sub-module
        for name, source_param in source_params.items():
            if 'attn' not in name:  # Skip parameters from the attention layer
                target_param = target_params.get(name)
                if target_param is not None:
                    target_param.data.copy_(source_param.data)

        self.attn = self.attn.copy_from_og(source_block.attn)


class TriformerCube(Transformer):
    def __init__(self, config: dict):
        super().__init__(config)
        self.blocks = self._get_blocks(TriformerCubeBlock)

    def copy_weights_from(self, source_transformer):

        # Get named parameters from both transformers
        target_params = dict(self.named_parameters())
        source_params = dict(source_transformer.named_parameters())

        # Copy parameters, skipping those in 'blocks'
        for name, source_param in source_params.items():
            if 'blocks' not in name:  # Skip parameters from the blocks
                target_param = target_params.get(name)
                if target_param is not None:
                    target_param.data.copy_(source_param.data)

        for i, block in enumerate(self.blocks):
            block.copy_from_og(source_transformer.blocks[i])


if __name__ == "__main__":

    cfg = Config()
    trittention_cube = TrittentionCube(cfg)
    trittention_cube_og = TrittentionCubeOG(cfg)
    trittention_cube = trittention_cube.copy_from_og(trittention_cube_og)

    b = 10  # Batch size
    tt = 20  # Temporal dimension
    d = 768  # Feature dimension
    
    # Generate a random tensor with the specified dimensions
    input_tensor = t.randn(b, tt, d)
    
    
    # Optionally, if your modules require being in evaluation mode (e.g., contain dropout layers), toggle them
    trittention_cube.eval()
    trittention_cube_og.eval()
    
    # Pass the random tensor through both modules
    output_cube_og = trittention_cube_og(input_tensor)
    output_cube = trittention_cube(input_tensor)
    
    # Print outputs
    print("Output from TrittentionCube:")
    #print(output_cube)
    print("\nOutput from TrittentionCubeOG:")
    #print(output_cube_og)
    
    # Compute and print the difference between the outputs
    difference = t.abs(output_cube - output_cube_og)
    print("\nDifference between outputs:")
    #print(difference)

    triformer_cube_block = TriformerCubeBlock(cfg)
    triformer_cube_block_og = TriformerCubeBlockOG(cfg)

    triformer_cube_block.copy_from_og(triformer_cube_block_og)

    triformer_cube_block.eval()
    triformer_cube_block_og.eval()

    output_cube_block_og = triformer_cube_block_og(input_tensor)
    output_cube_block = triformer_cube_block(input_tensor)
    difference = t.abs(output_cube_block - output_cube_block_og)
    print("\nDifference between outputs:")
    #print(difference)
    input_tensor = t.randint(0, 50200, (3, 5))
    cfg = cfg.to_dict()

    triformer_cube = TriformerCube(cfg)
    triformer_cube_og = TriformerCubeOG(cfg)

    triformer_cube.copy_weights_from(triformer_cube_og)

    triformer_cube.eval()
    triformer_cube_og.eval()

    output_cube_og = triformer_cube_og(input_tensor)
    output_cube = triformer_cube(input_tensor)
    difference = t.abs(output_cube - output_cube_og)
    print("\nDifference between outputs:")
    print(difference)
    print(difference.sum())
    print(f"max = {difference.max()}")
    print(f"min = {difference.min()}")