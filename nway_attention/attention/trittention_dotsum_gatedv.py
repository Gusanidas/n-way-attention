import einops
import torch as t
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float

from nway_attention.utils_misc import IdentityModule
from nway_attention.cfgs import Config


class Trittention(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.kkq = nn.Linear(cfg.d_model, cfg.d_head * cfg.n_heads * 3)
        self.V1 = nn.Linear(cfg.d_model, cfg.d_head * cfg.n_heads)
        self.V2 = nn.Linear(cfg.d_model, cfg.d_head * cfg.n_heads)

        self.Out = nn.Linear(cfg.d_head * cfg.n_heads, cfg.d_model)
        self.Mask = IdentityModule()
        self.AttentionScore = IdentityModule()
        self.HeadOutputs = IdentityModule()
        self.register_buffer("IGNORE", t.tensor(-1e6, dtype=t.float32))
        self.register_buffer("precomputed_mask", self.create_causal_mask(cfg.n_ctx))

    def create_causal_mask(self, max_seq_len):

        t_indices = (
            t.arange(max_seq_len).unsqueeze(0).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        )  # Shape: (1,1, t, 1, 1)
        s_indices = (
            t.arange(max_seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(1).unsqueeze(-1)
        )  # Shape: (1,1, 1, s, 1)
        q_indices = (
            t.arange(max_seq_len).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )  # Shape: (1,1, 1, 1, q)
        mask = (t_indices > q_indices) | (s_indices > q_indices)
        return mask

    def forward(self, normalized_resid_pre: t.Tensor) -> t.Tensor:
        # Assuming self.W_Q, self.W_K, self.W_V, and self.W_O are parameter matrices of the model
        bs, ts, ds = normalized_resid_pre.shape

        k1, k2, q = self.kkq(normalized_resid_pre).chunk(3, dim=-1)
        k1, k2, q = map(
            lambda t: einops.rearrange(t, "b p (h d) -> b p h d", h=self.cfg.n_heads),
            (k1, k2, q),
        )
        v1 = self.V1(normalized_resid_pre)
        v1 = einops.rearrange(v1, "b p (n h) -> b p n h", n=self.cfg.n_heads)
        v2 = self.V2(normalized_resid_pre)
        v2 = einops.rearrange(v2, "b p (n h) -> b p n h", n=self.cfg.n_heads)
        attn_score_qk = t.einsum("btnh, bqnh -> bntq", k2, q)
        attn_score_kk = t.einsum("bsnh, btnh -> bnst", k1, k2)
        attn_score = attn_score_qk.unsqueeze(2) + attn_score_kk.unsqueeze(-1)

        if self.cfg.causal_attn:
            attn_score = self.apply_causal_mask(attn_score)
        attn_score = (
            einops.rearrange(attn_score, "b n s t q -> b n q (s t)") / self.cfg.d_head
        )

        attn_score = attn_score.softmax(dim=-1)
        attn_score = self.AttentionScore(attn_score)
        v = F.silu(v1.unsqueeze(2)) * v2.unsqueeze(1)
        v = einops.rearrange(v, "b p t n h -> b n h (p t)")
        z = t.einsum("bnql, bnhl -> bqnh", attn_score, v)
        z = self.HeadOutputs(z)
        out = self.Out(z.reshape(bs, ts, -1))
        return out

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads a_pos b_pos c_pos"]
    ) -> Float[Tensor, "batch n_heads a_pos b_pos c_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        b, nn, tt, s, q = attn_scores.shape
        mask = self.precomputed_mask[:, :, :tt, :s, :q].to(attn_scores.device)
        mask = self.Mask(mask)
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


if __name__ == "__main__":
    # Test the Trittention module

    # Create a test configuration
    cfg = Config(d_model=512, d_head=64, n_heads=8, n_ctx=1024, causal_attn=True)

    # Create the module
    model = Trittention(cfg)

    # Create dummy input data
    batch_size = 2
    seq_len = 10
    dummy_input = t.randn(batch_size, seq_len, cfg.d_model)

    print(f"Testing Trittention module...")
    print(f"Input shape: {dummy_input.shape}")

    # Forward pass
    with t.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {dummy_input.shape}")

    # Verify output shape matches input shape
    assert (
        output.shape == dummy_input.shape
    ), f"Shape mismatch: {output.shape} vs {dummy_input.shape}"

    print("✓ Test passed! Output shape matches input shape.")

    # Test with different sequence lengths
    for test_seq_len in [1, 5, 20]:
        test_input = t.randn(1, test_seq_len, cfg.d_model)
        with t.no_grad():
            test_output = model(test_input)
        assert test_output.shape == test_input.shape
        print(f"✓ Test passed for sequence length {test_seq_len}")

    print("All tests completed successfully!")
