"""
Single global CSP-style soft prompt for the vanilla CLIP+CSP ablation
baseline (see train_vanilla.py).

Design
------
`n_ctx` learnable context vectors are shared across every composition (not
per-attribute/per-object, unlike the original CSP paper) and prepended, in
token-embedding space, to the tokenized text -- CoOp's technique (Zhou et
al., 2022) applied to whatever text is passed in (attribute word, object
word, or full "attr obj" phrase; training only ever uses the full phrase).
Every CLIP weight touched (token_embedding, positional_embedding,
transformer, ln_final, text_projection) is frozen -- `CLIPEncoder` already
sets requires_grad=False on all of them -- so running the forward pass
WITHOUT torch.no_grad() lets gradients reach `ctx` alone.

This is deliberately a different, simpler class from any per-head soft
prompt encoder elsewhere in this repo: there is exactly one shared context
here, applied identically regardless of caller, because the vanilla baseline
has no primitive routing at all.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VanillaSoftPromptEncoder(nn.Module):
    """
    Args:
        clip_model: the frozen CLIP model (`CLIPEncoder.clip_model`). Its
                    weights are read but never written; only `ctx` below is
                    a trainable parameter of this module.
        n_ctx     : number of learnable context tokens prepended to the text.
    """

    def __init__(self, clip_model: nn.Module, n_ctx: int = 8) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.n_ctx = n_ctx

        dtype = clip_model.token_embedding.weight.dtype
        ctx_dim = clip_model.token_embedding.weight.shape[1]

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        self._prompt_prefix = " ".join(["X"] * n_ctx)

        logger.info(
            "VanillaSoftPromptEncoder: n_ctx=%d ctx_dim=%d dtype=%s | %d trainable params",
            n_ctx, ctx_dim, dtype, self.ctx.numel(),
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Args:
            texts: list of B strings (e.g. "red chair").

        Returns:
            (B, text_dim) L2-normalised text embeddings, computed by running
            CLIP's own frozen text transformer over [SOS, ctx (learnable),
            <tokenized text>, EOT, padding].
        """
        import clip

        device = self.ctx.device
        dtype = self.ctx.dtype

        prompts = [f"{self._prompt_prefix} {t}." for t in texts]
        tokenized = clip.tokenize(prompts, truncate=True).to(device)   # (B, 77)

        embedding = self.clip_model.token_embedding(tokenized).type(dtype)  # (B, 77, dim)
        prefix = embedding[:, :1, :]                 # SOS token embedding
        suffix = embedding[:, 1 + self.n_ctx:, :]    # text + EOT + padding
        ctx = self.ctx.unsqueeze(0).expand(len(texts), -1, -1)

        x = torch.cat([prefix, ctx, suffix], dim=1)               # (B, 77, dim)
        x = x + self.clip_model.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)                                     # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)                                     # LND -> NLD
        x = self.clip_model.ln_final(x).type(dtype)

        # EOT is the highest-id token in CLIP's vocabulary, so argmax over
        # the (unmodified) tokenized sequence locates it regardless of how
        # many context tokens were spliced in ahead of the real text.
        eot_idx = tokenized.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=device), eot_idx] @ self.clip_model.text_projection

        return F.normalize(x.float(), dim=-1)


# ----------------------------------------------------------------------
# Smoke test  —  python models/soft_prompt_vanilla.py --smoke
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run a forward/backward shape+grad check against real CLIP weights.")
    args = parser.parse_args()

    if not args.smoke:
        parser.print_help()
        raise SystemExit(0)

    from models.clip_encoder import CLIPEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = CLIPEncoder.load_pretrained(device=device)
    encoder.clip_model.float()

    soft_prompt = VanillaSoftPromptEncoder(encoder.clip_model, n_ctx=8).to(device)
    texts = ["red chair", "ancient building", "wet dog"]

    out = soft_prompt(texts)
    assert out.shape == (3, encoder.text_dim), f"got {tuple(out.shape)}"
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3, device=device), atol=1e-4), norms.tolist()
    print(f"[check 1] output shape={tuple(out.shape)}, unit-norm ✓")

    soft_prompt.zero_grad(set_to_none=True)
    soft_prompt(texts).sum().backward()
    assert soft_prompt.ctx.grad is not None, "ctx received no gradient"
    assert all(p.grad is None for p in encoder.clip_model.parameters()), \
        "gradient leaked into frozen CLIP weights"
    print("[check 2] grad reaches ctx only, CLIP backbone untouched ✓")

    print("\nAll checks passed.")
