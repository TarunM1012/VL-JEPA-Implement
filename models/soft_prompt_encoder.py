"""
CSP-style learnable soft prompts for CLIP text targets (v3-CLIP r2).

Design summary
--------------
v3-CLIP r1 encoded every text target with CLIP's own frozen text tower fed
raw words ("red", "chair", "red chair") — no template at all. This module
replaces that with three independent sets of learnable context vectors, one
per primitive head, prepended to the class-word tokens before the frozen
CLIP text transformer:

    attr target ─► [SOT] [attr_ctx_1 .. attr_ctx_M] [attr words]     [EOT]
    obj  target ─► [SOT] [obj_ctx_1  .. obj_ctx_M ] [obj words]      [EOT]
    comp target ─► [SOT] [comp_ctx_1 .. comp_ctx_M] [attr obj words] [EOT]

Only the M context vectors per head are trainable (CoOp/CSP-style soft
prompting). The class-word token embeddings come from CLIP's own frozen
token_embedding table, and everything downstream — positional embedding,
transformer, ln_final, text_projection — is CLIP's frozen text tower,
untouched. Nothing else about v3-CLIP r1 changes: visual encoder, primitive
heads, loss, and batch sampler are all identical.

Implementation note
--------------------
Token-level prompt injection (rather than string concatenation + tokenize)
is required because the context vectors are continuous parameters, not
discrete tokens. The standard trick (CoOp) is to tokenize a placeholder
prompt where each context slot is filled by a single-BPE-token stand-in
("X", confirmed to tokenize to exactly one id in CLIP's vocab), then splice
the learned vectors in at that position before running the transformer:

    tokens   = tokenize("X X ... X <words>")   # placeholder reserves M slots
    embeds   = token_embedding(tokens)         # (B, 77, ctx_dim)
    prefix   = embeds[:, :1]                   # SOT
    suffix   = embeds[:, 1+M:]                 # <words> + EOT + padding
    spliced  = cat([prefix, learned_ctx, suffix], dim=1)

Sequence length and the EOT token's position are unchanged by the splice
(placeholder and learned context both occupy exactly M slots), so
`tokens.argmax(dim=-1)` — CLIP's own trick for locating EOT, the highest
token id in the sequence — still correctly pools the final transformer
state for each prompt.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_PLACEHOLDER_TOKEN = "X"   # confirmed to tokenize to exactly one CLIP BPE id
_HEADS = ("attr", "obj", "comp")


class SoftPromptTextEncoder(nn.Module):
    """
    Wraps a frozen CLIP model's text tower with three learnable CSP-style
    soft-prompt context vectors (attr / obj / comp), one per primitive head.

    Usage
    -----
        soft_prompts = SoftPromptTextEncoder(clip_encoder.clip_model, n_ctx=8)
        target_embeds = soft_prompts(texts, head="attr")   # (B, text_dim)

    `texts` are plain class words/phrases exactly as before ("red",
    "chair", "red chair") — the learnable prompt replaces any hand-written
    template, so callers no longer prepend anything themselves.
    """

    def __init__(self, clip_model: nn.Module, n_ctx: int = 8) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.n_ctx = n_ctx
        self.ctx_dim = clip_model.token_embedding.weight.shape[1]

        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        def _new_ctx() -> nn.Parameter:
            v = torch.empty(n_ctx, self.ctx_dim)
            nn.init.normal_(v, std=0.02)
            return nn.Parameter(v)

        self.attr_ctx = _new_ctx()
        self.obj_ctx = _new_ctx()
        self.comp_ctx = _new_ctx()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SoftPromptTextEncoder: n_ctx=%d ctx_dim=%d | 3 heads | %d trainable params",
            n_ctx, self.ctx_dim, n_params,
        )

    # ------------------------------------------------------------------
    # Placeholder tokenization
    # ------------------------------------------------------------------

    def _tokenize_with_placeholder(self, texts: list[str], device: torch.device) -> torch.Tensor:
        import clip

        placeholder = " ".join([_PLACEHOLDER_TOKEN] * self.n_ctx)
        prompts = [f"{placeholder} {t}" for t in texts]
        return clip.tokenize(prompts, truncate=True).to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str], ctx: torch.Tensor) -> torch.Tensor:
        device = ctx.device
        dtype = self.clip_model.dtype

        tokens = self._tokenize_with_placeholder(texts, device)   # (B, 77)
        with torch.no_grad():
            token_embeds = self.clip_model.token_embedding(tokens).type(dtype)  # (B, 77, ctx_dim)

        B = token_embeds.shape[0]
        prefix = token_embeds[:, :1, :]                            # SOT
        suffix = token_embeds[:, 1 + self.n_ctx:, :]                # words + EOT + pad
        ctx_expanded = ctx.unsqueeze(0).expand(B, -1, -1).type(dtype)  # (B, n_ctx, ctx_dim)
        x = torch.cat([prefix, ctx_expanded, suffix], dim=1)       # (B, 77, ctx_dim)

        x = x + self.clip_model.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)                                     # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)                                     # LND -> NLD
        x = self.clip_model.ln_final(x).type(dtype)

        eot_idx = tokens.argmax(dim=-1)                            # EOT = highest token id
        pooled = x[torch.arange(B, device=device), eot_idx]
        text_features = pooled @ self.clip_model.text_projection
        return text_features.float()

    def forward(self, texts: list[str], head: str) -> torch.Tensor:
        """
        Args:
            texts: list of B class words/phrases (no template — the
                   learnable prompt replaces it).
            head : one of "attr", "obj", "comp" — selects which head's
                   context vectors to use.

        Returns:
            (B, text_dim) text embeddings, NOT L2-normalised (matches
            CLIPEncoder.get_text_features's convention; callers normalise
            explicitly).
        """
        if head not in _HEADS:
            raise ValueError(f"head must be one of {_HEADS}, got {head!r}")
        ctx = {"attr": self.attr_ctx, "obj": self.obj_ctx, "comp": self.comp_ctx}[head]
        return self._encode(texts, ctx)


# ----------------------------------------------------------------------
# Smoke test  —  python models/soft_prompt_encoder.py --smoke
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run shape + gradient-isolation checks and exit.")
    args = parser.parse_args()

    if not args.smoke:
        parser.print_help()
        raise SystemExit(0)

    clip = __import__("clip")
    from models.clip_encoder import _CLIP_DOWNLOAD_ROOT, _CLIP_MODEL_NAME

    clip_model, _ = clip.load(_CLIP_MODEL_NAME, device="cpu", download_root=_CLIP_DOWNLOAD_ROOT)
    soft_prompts = SoftPromptTextEncoder(clip_model, n_ctx=8)

    texts = ["red", "chair", "red chair"]

    # ---- Check 1: forward shapes for all three heads ----------------------
    for head in _HEADS:
        out = soft_prompts(texts, head=head)
        assert out.shape == (3, clip_model.text_projection.shape[1]), \
            f"{head}: got {tuple(out.shape)}"
        print(f"[check 1] head={head} output = {tuple(out.shape)} ✓")

    # ---- Check 2: per-head context vectors isolate gradients --------------
    soft_prompts.zero_grad(set_to_none=True)
    soft_prompts(texts, head="attr").sum().backward()
    assert soft_prompts.attr_ctx.grad is not None
    assert soft_prompts.obj_ctx.grad is None
    assert soft_prompts.comp_ctx.grad is None
    print("[check 2] attr-head backward isolated to attr_ctx ✓")

    # ---- Check 3: CLIP backbone receives no gradient -----------------------
    assert all(p.grad is None for p in clip_model.parameters())
    print("[check 3] frozen CLIP backbone untouched by backward ✓")

    print("\nAll checks passed.")
