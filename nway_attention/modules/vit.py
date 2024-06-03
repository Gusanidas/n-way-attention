## Copied from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn

from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange
from huggingface_hub import PyTorchModelHubMixin

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class TrittentionCube(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.abcde = nn.Linear(dim, 5 * dim_head * heads)
        self.dropout = nn.Dropout(dropout)
        
        self.out_p = nn.Sequential(nn.Linear(dim_head * heads, dim), nn.Dropout(dropout))
        self.W_K = nn.Parameter(torch.empty((heads, dim_head, dim_head, dim_head)))
        self.W_V = nn.Parameter(torch.empty((heads, dim_head, dim_head, dim_head)))

        self.dim_head = dim_head
        self.heads = heads

    def forward(self, x):
        x = self.norm(x)
        a, b, c, d, e = self.abcde(x).chunk(5, dim=-1)
        a, b, c, d, e = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = self.heads), (a, b, c, d, e))
        step1 = torch.einsum('brnk, nijk -> bnrij', c, self.W_K)
        step2 = torch.einsum('bnrij, bqnj -> bnriq', step1, b)
        attn_score = torch.einsum('bnriq, bpni -> bnpqr', step2, a)
        #attn_score = self.apply_causal_mask(attn_score)
        attn_score = rearrange(attn_score, "b n p1 p2 p3 -> b n p3 (p1 p2)")/self.dim_head
        attn_score = attn_score.softmax(dim=-1)
        attn_score = self.dropout(attn_score)
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

class Trittention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.abcde = nn.Linear(dim, 5 * dim_head * heads)
        self.dropout = nn.Dropout(dropout)
        
        self.out_p = nn.Sequential(nn.Linear(dim_head * heads, dim), nn.Dropout(dropout))

        self.dim_head = dim_head
        self.heads = heads

    def forward(self, x):
        b, t, d = x.shape
        x = self.norm(x)
        a, b, c, d, e = self.abcde(x).chunk(5, dim=-1)
        a, b, c, d, e = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = self.heads), (a, b, c, d, e))

        attn_score = torch.einsum("bsnh, btnh, bqnh -> bnstq", a,b,c)
        attn_score = rearrange(attn_score, "b n p1 p2 p3 -> b n p3 (p1 p2)")/self.dim_head
        attn_score = attn_score.softmax(dim=-1)
        attn_score = rearrange(attn_score, "b n p1 (p2 p3) -> b n p1 p2 p3", p2 = t, p3 = t) 
        z = torch.einsum('bnqlr, blnd -> bnqd', attn_score, d) + torch.einsum('bnqlr, brnd -> bnqd', attn_score, e)
        z = self.dropout(z)
        z = rearrange(z, 'b n q d -> b q (n d)')
        return self.out_p(z)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class TriformerCube(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TrittentionCube(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class Triformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Trittention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class ViTC(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TriformerCube(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class MixedAttention(nn.Module):
    def __init__(self, dim, a_heads, c_heads, dim_head, dropout):
        super().__init__()
        self.a_heads = a_heads
        self.c_heads = c_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.attn = Attention(dim, heads = a_heads, dim_head = dim_head, dropout = dropout)
        self.cube_tri = Trittention(dim, heads = c_heads, dim_head = dim_head, dropout = dropout)
    def forward(self, x):
        x = self.attn(x) + self.cube_tri(x)
        return x

class MixedTrans(nn.Module):

    def __init__(self, dim, depth, a_heads, c_heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MixedAttention(dim, a_heads = a_heads, c_heads = c_heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViTMixed(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, a_heads, c_heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = MixedTrans(dim, depth, a_heads, c_heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
class ViTtri(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        cfg = config
        image_size = cfg['image_size']
        patch_size = cfg['patch_size']
        num_classes = cfg['num_classes']
        dim = cfg['dim']
        depth = cfg['depth']
        heads = cfg['heads']
        mlp_dim = cfg['mlp_dim']
        pool = cfg.get('pool', 'cls')
        channels = cfg.get('channels', 3)
        dim_head = cfg.get('dim_head', 64)
        dropout = cfg.get('dropout', 0.0)
        emb_dropout = cfg.get('emb_dropout', 0.0)

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Triformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)


    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



if __name__ == '__main__':
    model = ViTtri(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 1024)

    img = torch.randn(1, 3, 256, 256)
    preds = model(img) # (1, 1000)
    print(preds.shape)