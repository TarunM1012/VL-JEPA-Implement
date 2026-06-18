"""
Primitive predictor heads for VL-JEPA (object / attribute / composition).

Design summary
--------------
This module replaces the single LLaMA-3.2-1B predictor with three small,
from-scratch heads that each specialise in one CZSL primitive:

    AttributeHead  : visual patch tokens ─► (B, 1536)  ≈ Y("red")
    ObjectHead     : visual patch tokens ─► (B, 1536)  ≈ Y("chair")
    CompositionHead: attr_embed + obj_embed ─► (B, 1536) ≈ Y("red chair")

`AttributeHead` and `ObjectHead` are two independent instances of the SAME
architecture (`TransformerHead`): a fully-bidirectional transformer encoder
built from scratch with random initialisation — NOT a retrofit of a causal
language model.  Each is ~14 M params, well inside the 8–15 M target.

`CompositionHead` is deliberately a plain MLP (no attention) so that any
harmonic-mean improvement over the single-predictor baseline is attributable
to the per-primitive batching/routing strategy rather than to a more
expressive fusion mechanism.  Its inputs are the OUTPUT embeddings of the
attribute and object heads (not the raw visual tokens), so it sits
sequentially on top of them.

All heads emit L2-normalised 1536-dim vectors so that a dot product equals
cosine similarity, matching `InfoNCELoss` and the Y-encoder output geometry.

Loss routing
------------
Each head is trained only on its corresponding batch type emitted by
`data/primitive_sampler.py`:

    attr-batch  ─► AttributeHead   vs  Y(attribute word alone)
    obj-batch   ─► ObjectHead      vs  Y(object word alone)
    comp-batch  ─► CompositionHead vs  Y(full "attr obj" phrase)

Crucially, the composition forward path detaches the attribute/object
embeddings (`PrimitiveHeads.forward_composition` runs the two transformer
heads under `torch.no_grad()`), so composition-batch gradients never flow
back into the attribute or object heads.  This guarantees the invariant that
"each head is trained only on its own batch type".
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_VISUAL_DIM = 1024   # X-encoder patch-token dimension (ViT-L / V-JEPA 2)
_SHARED_DIM = 1536   # shared visual–language embedding space (matches Y-encoder)

Pool = Literal["mean", "token"]


# ----------------------------------------------------------------------
# Transformer head (attribute / object)
# ----------------------------------------------------------------------

class TransformerHead(nn.Module):
    """
    A lightweight, fully-bidirectional transformer encoder over visual tokens.

    Pipeline
    --------
        visual_embeds (B, F, P, 1024)
            └─ flatten frames×patches ─► (B, F*P, 1024)
            └─ input_proj             ─► (B, F*P, hidden_dim)
            └─ [optional CLS token prepended]
            └─ TransformerEncoder     ─► (B, S, hidden_dim)   (no causal mask)
            └─ pool (mean | CLS token)─► (B, hidden_dim)
            └─ output_proj            ─► (B, 1536)
            └─ L2-normalise           ─► (B, 1536)

    The encoder is bidirectional by construction: `nn.TransformerEncoder`
    applies no causal mask unless one is passed, and we never pass one.  All
    patch tokens (across all frames) attend to one another freely.

    Parameter budget (defaults: hidden=512, layers=4, heads=8, ffn_mult=4):
        ~13.9 M trainable params — inside the 8–15 M target.

    Args
    ----
        visual_dim : input patch-token dim (1024 for ViT-L).
        hidden_dim : transformer width (paper-task range 384–512).
        num_layers : number of encoder layers (4–6).
        num_heads  : attention heads per layer (6–8); must divide hidden_dim.
        ffn_mult   : feed-forward expansion factor (~4×).
        output_dim : shared embedding dim (1536, matches Y-encoder).
        pool       : "mean" (masked-free mean over tokens) or "token"
                     (a learnable aggregation/CLS token).
        dropout    : encoder dropout.  Defaults to 0.0 so that the
                     embeddings consumed by the composition head are
                     deterministic; the contrastive objective + per-primitive
                     batching already provide regularisation.
    """

    def __init__(
        self,
        visual_dim: int = _VISUAL_DIM,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_mult: int = 4,
        output_dim: int = _SHARED_DIM,
        pool: Pool = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads "
                f"({num_heads})"
            )
        self.pool = pool
        self.hidden_dim = hidden_dim

        # Project ViT patch tokens into the transformer width.
        self.input_proj = nn.Linear(visual_dim, hidden_dim)

        # Optional learnable aggregation token (CLS-style).  Prepended to the
        # sequence; its final state is used as the pooled representation.
        if pool == "token":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

        # norm_first (pre-LN) is the modern default — it trains more stably
        # from random init than post-LN, which matters here because nothing is
        # pretrained.  GELU matches the transformer-encoder convention.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: nested-tensor fast path is incompatible
        # with norm_first and only helps padded batches (we never pad visual
        # tokens); disabling it also silences a spurious warning.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
        )

        # Final projection into the shared 1536-dim space.  No bias keeps the
        # origin meaningful after L2 normalisation (same rationale as the
        # Y-encoder projection head).
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=False)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "TransformerHead: hidden=%d layers=%d heads=%d ffn=%d pool=%s "
            "| %.2f M params",
            hidden_dim, num_layers, num_heads, hidden_dim * ffn_mult,
            pool, n_params / 1e6,
        )

    def forward(self, visual_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_embeds : (B, F, num_patches, visual_dim) patch tokens from
                            the X-encoder.

        Returns:
            (B, output_dim) L2-normalised embeddings.
        """
        B, num_frames, P, D = visual_embeds.shape

        # Flatten frames × patches into one token axis so every patch from
        # every frame participates in attention together.
        x = visual_embeds.reshape(B, num_frames * P, D)   # (B, F*P, visual_dim)
        x = self.input_proj(x)                            # (B, F*P, hidden_dim)

        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)        # (B, 1, hidden_dim)
            x = torch.cat([cls, x], dim=1)                # (B, 1+F*P, hidden_dim)

        # No mask passed → fully bidirectional self-attention.  Visual tokens
        # are never padded, so there is nothing to mask.
        x = self.encoder(x)                               # (B, S, hidden_dim)

        if self.pool == "token":
            pooled = x[:, 0]                              # CLS state → (B, hidden_dim)
        else:
            pooled = x.mean(dim=1)                        # mean over tokens → (B, hidden_dim)

        projected = self.output_proj(pooled)              # (B, output_dim)
        return F.normalize(projected, dim=-1)


# ----------------------------------------------------------------------
# Composition head (MLP fusion)
# ----------------------------------------------------------------------

class CompositionHead(nn.Module):
    """
    Fuses the attribute and object embeddings into a composition embedding.

    Deliberately a plain MLP (no attention) so the composition mechanism is no
    more expressive than a baseline fusion — any improvement should come from
    the routing/batching strategy, not the fusion.

    Args
    ----
        embed_dim : dimension of the attr/obj/output embeddings (1536).
        hidden_dim: width of the MLP hidden layers.
        num_layers: number of linear layers (2 or 3).
        fusion    : "concat" → input is [attr ; obj]  (2*embed_dim)
                    "sum"    → input is attr + obj     (embed_dim)
        dropout   : MLP dropout.
    """

    def __init__(
        self,
        embed_dim: int = _SHARED_DIM,
        hidden_dim: int = _SHARED_DIM,
        num_layers: int = 3,
        fusion: Literal["concat", "sum"] = "concat",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError(f"CompositionHead needs >=2 layers, got {num_layers}")
        self.fusion = fusion

        in_dim = 2 * embed_dim if fusion == "concat" else embed_dim

        # Build [Linear → GELU → (dropout)] × (num_layers-1) → Linear(→embed_dim).
        layers: List[nn.Module] = []
        prev = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, embed_dim))
        self.mlp = nn.Sequential(*layers)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "CompositionHead: fusion=%s layers=%d hidden=%d | %.2f M params",
            fusion, num_layers, hidden_dim, n_params / 1e6,
        )

    def forward(
        self,
        attr_embed: torch.Tensor,   # (B, embed_dim), L2-normalised
        obj_embed: torch.Tensor,    # (B, embed_dim), L2-normalised
    ) -> torch.Tensor:
        """Returns (B, embed_dim) L2-normalised composition embeddings."""
        if self.fusion == "concat":
            x = torch.cat([attr_embed, obj_embed], dim=-1)   # (B, 2*embed_dim)
        else:  # "sum"
            x = attr_embed + obj_embed                       # (B, embed_dim)
        out = self.mlp(x)                                    # (B, embed_dim)
        return F.normalize(out, dim=-1)


# ----------------------------------------------------------------------
# Container
# ----------------------------------------------------------------------

class PrimitiveHeads(nn.Module):
    """
    Bundles the attribute, object, and composition heads behind one module so
    training/eval/checkpointing treat them as a unit.

    Forward entry points (one per batch type):
        forward_attribute(visual)    → (B, 1536)   train on attr-batches
        forward_object(visual)       → (B, 1536)   train on obj-batches
        forward_composition(visual)  → (B, 1536)   train on comp-batches
        compose(attr_embed, obj_embed) → (B, 1536) reuse precomputed embeds
    """

    def __init__(
        self,
        attr_head: TransformerHead,
        obj_head: TransformerHead,
        comp_head: CompositionHead,
    ) -> None:
        super().__init__()
        self.attr_head = attr_head
        self.obj_head = obj_head
        self.comp_head = comp_head
        # Populated by build(); used to round-trip architecture through a
        # checkpoint.  None when heads are constructed directly.
        self.config: Optional[dict] = None

    # ---- Construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_mult: int = 4,
        pool: Pool = "mean",
        head_dropout: float = 0.0,
        comp_fusion: Literal["concat", "sum"] = "concat",
        comp_layers: int = 3,
        comp_hidden: int = _SHARED_DIM,
        device: Optional[torch.device] = None,
    ) -> "PrimitiveHeads":
        """Construct all three heads with shared transformer hyperparameters."""
        attr_head = TransformerHead(
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            ffn_mult=ffn_mult, pool=pool, dropout=head_dropout,
        )
        obj_head = TransformerHead(
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            ffn_mult=ffn_mult, pool=pool, dropout=head_dropout,
        )
        comp_head = CompositionHead(
            hidden_dim=comp_hidden, num_layers=comp_layers, fusion=comp_fusion,
        )
        instance = cls(attr_head, obj_head, comp_head)

        # Record the build hyperparameters so a checkpoint can be rebuilt with
        # the identical architecture at eval time (state_dict alone does not
        # capture structural choices like layer count or pooling).
        instance.config = dict(
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            ffn_mult=ffn_mult, pool=pool, head_dropout=head_dropout,
            comp_fusion=comp_fusion, comp_layers=comp_layers, comp_hidden=comp_hidden,
        )

        total = sum(p.numel() for p in instance.parameters() if p.requires_grad)
        logger.info("PrimitiveHeads: %.2f M trainable params total", total / 1e6)

        if device is not None:
            instance = instance.to(device)
        return instance

    # ---- Optimiser integration --------------------------------------------

    def param_groups(self, base_lr: float) -> List[dict]:
        """
        One param group per head, all at the full base LR.  These heads are
        trained from scratch (unlike the Y-encoder projection, which sits on a
        frozen backbone and uses a reduced LR), so no LR multiplier is applied.
        """
        return [
            {"params": self.attr_head.parameters(), "lr": base_lr},
            {"params": self.obj_head.parameters(),  "lr": base_lr},
            {"params": self.comp_head.parameters(), "lr": base_lr},
        ]

    # ---- Forward entry points ---------------------------------------------

    def forward_attribute(self, visual_embeds: torch.Tensor) -> torch.Tensor:
        return self.attr_head(visual_embeds)

    def forward_object(self, visual_embeds: torch.Tensor) -> torch.Tensor:
        return self.obj_head(visual_embeds)

    def compose(
        self,
        attr_embed: torch.Tensor,
        obj_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse precomputed attr/obj embeddings (used at eval time)."""
        return self.comp_head(attr_embed, obj_embed)

    def forward_composition(self, visual_embeds: torch.Tensor) -> torch.Tensor:
        """
        Composition prediction from raw visual tokens.

        The attribute/object heads are run under `torch.no_grad()` so their
        outputs are detached: composition-batch gradients update ONLY the
        composition head, never the attr/obj heads.  This enforces the
        "each head trains on its own batch type" invariant.
        """
        with torch.no_grad():
            attr_embed = self.attr_head(visual_embeds)
            obj_embed = self.obj_head(visual_embeds)
        return self.comp_head(attr_embed, obj_embed)


# ----------------------------------------------------------------------
# Smoke test  —  python models/primitive_heads.py --smoke
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run shape + param-count checks and exit.")
    args = parser.parse_args()

    if not args.smoke:
        parser.print_help()
        raise SystemExit(0)

    torch.manual_seed(0)

    # (batch=2, frames=2, patches=196, visual_dim=1024)
    B, num_frames, P, D = 2, 2, 196, _VISUAL_DIM
    visual = torch.randn(B, num_frames, P, D)

    heads = PrimitiveHeads.build()

    # ---- Check 1: per-head param budget (8–15 M) -------------------------
    for name, head in [("attr", heads.attr_head), ("obj", heads.obj_head)]:
        n = sum(p.numel() for p in head.parameters()) / 1e6
        assert 8.0 <= n <= 15.0, f"{name} head has {n:.2f} M params (want 8–15 M)"
        print(f"[check 1] {name} head params = {n:.2f} M  (8–15 M) ✓")

    # ---- Check 2: forward shapes ----------------------------------------
    attr_e = heads.forward_attribute(visual)
    obj_e  = heads.forward_object(visual)
    comp_e = heads.forward_composition(visual)
    for name, out in [("attr", attr_e), ("obj", obj_e), ("comp", comp_e)]:
        assert out.shape == (B, _SHARED_DIM), f"{name}: got {tuple(out.shape)}"
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5), \
            f"{name} not L2-normalised: {norms.tolist()}"
        print(f"[check 2] {name} output = {tuple(out.shape)}, unit-norm ✓")

    # ---- Check 3: composition does NOT touch attr/obj head grads ---------
    heads.zero_grad(set_to_none=True)
    heads.forward_composition(visual).sum().backward()
    attr_grad = any(p.grad is not None for p in heads.attr_head.parameters())
    obj_grad  = any(p.grad is not None for p in heads.obj_head.parameters())
    comp_grad = any(p.grad is not None for p in heads.comp_head.parameters())
    assert not attr_grad and not obj_grad, \
        "composition backward leaked gradients into attr/obj heads"
    assert comp_grad, "composition backward produced no gradient in comp head"
    print("[check 3] comp-batch grads isolated to composition head ✓")

    # ---- Check 4: attr/obj backward updates only their own head ----------
    heads.zero_grad(set_to_none=True)
    heads.forward_attribute(visual).sum().backward()
    assert any(p.grad is not None for p in heads.attr_head.parameters())
    assert all(p.grad is None for p in heads.obj_head.parameters())
    print("[check 4] attr-batch grads isolated to attribute head ✓")

    print("\nAll checks passed.")
