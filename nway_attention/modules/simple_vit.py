## Copied from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

import torch
from torch import nn

from einops import rearrange, einsum
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class TrittentionCube(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.abcde = nn.Linear(dim, 5 * dim_head * heads)
        self.out_p = nn.Linear(dim_head * heads, dim)
        self.W_K = nn.Parameter(torch.empty((heads, dim_head, dim_head, dim_head)))
        self.W_V = nn.Parameter(torch.empty((heads, dim_head, dim_head, dim_head)))

        self.dim_head = dim_head
        self.heads = heads

    def forward(self, x):
        x = self.norm(x)
        a, b, c, d, e = self.abcde(x).chunk(5, dim=-1)
        a, b, c, d, e = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = self.heads), (a, b, c, d, e))
        step1 = torch.einsum('brnk, nijk -> brij', c, self.W_K)
        step2 = torch.einsum('brij, bqnj -> briq', step1, b)
        attn_score = torch.einsum('briq, bpni -> bnpqr', step2, a)
        attn_score = self.apply_causal_mask(attn_score)
        attn_score = rearrange(attn_score, "b n p1 p2 p3 -> b n p3 (p1 p2)")/self.dim_head
        attn_score = attn_score.softmax(dim=-1)
        v = einsum(d,e,self.W_V,"b p1 n h1, b p2 n h2, n h1 h2 h3 -> b p1 p2 n h3")
        v = rearrange(v, "b p s n h -> b n h (p s)")
        z = torch.einsum('bnql, bnhl -> bqnh', attn_score, v)
        z = rearrange(z, 'b h n d -> b h (n d)')
        return self.out_p(z)

    def apply_causal_mask(
        self, attn_scores):
        b, nn, tt, s, q = attn_scores.shape

        t_indices = torch.arange(tt).unsqueeze(0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Shape: (1, t, 1, 1)
        s_indices = torch.arange(s).unsqueeze(0).unsqueeze(0).unsqueeze(1).unsqueeze(-1)  # Shape: (1, 1, s, 1)
        q_indices = torch.arange(q).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)   # Shape: (1, 1, 1, q)

        mask = t_indices >= s_indices
        mask = mask.to(attn_scores.device)

        attn_scores.masked_fill_(mask, 1e-6)
        return attn_scores

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

class TriformerCube(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TrittentionCube(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class SimpleViTCube(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = TriformerCube(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x) 


if __name__ == '__main__':
    model = SimpleViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 1024
    )

    img = torch.randn(1, 3, 256, 256)
    preds = model(img)
    print(preds.shape) # (1, 1000)

    model = SimpleViTCube(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 1024
    )

    img = torch.randn(1, 3, 256, 256)
    preds = model(img)
    print(preds.shape) # (1, 1000)