"""
Primitive predictor heads for VL-JEPA (object / attribute / composition).

Design summary
--------------
This module replaces the single LLaMA-3.2-1B predictor with three small,
from-scratch heads that each specialise in one CZSL primitive:

    AttributeHead  : visual patch tokens ─► (B, 768)  ≈ Y("red")
    ObjectHead     : visual patch tokens ─► (B, 768)  ≈ Y("chair")
    CompositionHead: attr_embed + obj_embed ─► (B, 768) ≈ Y("red chair")

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

All heads emit L2-normalised 768-dim vectors so that a dot product equals
cosine similarity, matching `InfoNCELoss` and CLIP's text-projection geometry
(ViT-L/14's projected embed dim).

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

_VISUAL_DIM = 1024   # CLIP ViT-L/14 visual transformer width (pre-projection patch tokens)
_SHARED_DIM = 768    # CLIP ViT-L/14 projected text embedding dim (image/text share this space)

Pool = Literal["mean", "token"]


# ----------------------------------------------------------------------
# Transformer head (attribute / object)
# ----------------------------------------------------------------------

class TransformerHead(nn.Module):
    """
    A lightweight, fully-bidirectional transformer encoder over visual tokens.

    Pipeline
    --------
        visual_embeds (B, P, 1024)
            └─ input_proj             ─► (B, P, hidden_dim)
            └─ [optional CLS token prepended]
            └─ TransformerEncoder     ─► (B, S, hidden_dim)   (no causal mask)
            └─ pool (mean | CLS token)─► (B, hidden_dim)
            └─ output_proj            ─► (B, 768)
            └─ L2-normalise           ─► (B, 768)

    The encoder is bidirectional by construction: `nn.TransformerEncoder`
    applies no causal mask unless one is passed, and we never pass one.  All
    patch tokens attend to one another freely.

    Parameter budget (defaults: hidden=512, layers=4, heads=8, ffn_mult=4):
        ~13.9 M trainable params — inside the 8–15 M target.

    Args
    ----
        visual_dim : input patch-token dim (1024 for CLIP ViT-L/14, pre-projection).
        hidden_dim : transformer width (paper-task range 384–512).
        num_layers : number of encoder layers (4–6).
        num_heads  : attention heads per layer (6–8); must divide hidden_dim.
        ffn_mult   : feed-forward expansion factor (~4×).
        output_dim : shared embedding dim (768, matches CLIP text projection).
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

        # Final projection into the shared 768-dim space.  No bias keeps the
        # origin meaningful after L2 normalisation (same rationale as the
        # CLIP text-projection geometry it targets).
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
            visual_embeds : (B, num_patches, visual_dim) patch tokens from
                            the X-encoder.

        Returns:
            (B, output_dim) L2-normalised embeddings.
        """
        x = self.input_proj(visual_embeds)                # (B, P, hidden_dim)

        if self.cls_token is not None:
            cls = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, hidden_dim)
            x = torch.cat([cls, x], dim=1)                # (B, 1+P, hidden_dim)

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
        embed_dim : dimension of the attr/obj/output embeddings (768).
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
        visual_dim: int = _VISUAL_DIM,
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError(f"CompositionHead needs >=2 layers, got {num_layers}")
        self.fusion = fusion

        in_dim = 2 * embed_dim + visual_dim if fusion == "concat" else embed_dim

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
        visual_vec: torch.Tensor,   # (B, visual_dim), L2-normalised mean-pooled patch tokens
    ) -> torch.Tensor:
        """Returns (B, embed_dim) L2-normalised composition embeddings."""
        if self.fusion == "concat":
            x = torch.cat([attr_embed, obj_embed, visual_vec], dim=-1)  # (B, 2*embed_dim+visual_dim)
        else:  # "sum"
            x = attr_embed + obj_embed                       # (B, embed_dim)
        out = self.mlp(x)                                    # (B, embed_dim)
        return F.normalize(out, dim=-1)


# ----------------------------------------------------------------------
# Optimiser param-group splitting (decay vs. no-decay)
# ----------------------------------------------------------------------

def _split_decay_params(module: nn.Module) -> tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Split a module's parameters into (decay, no_decay) groups for AdamW.

    LayerNorm weight/bias and all Linear biases should not be weight-decayed:
    decaying them pulls the value toward zero for no principled reason,
    fighting whatever the task loss is doing to it every step (same
    rationale as not decaying the InfoNCE temperature). Matched by module
    type (nn.LayerNorm) rather than name substring, since
    nn.TransformerEncoderLayer names its norms "norm1"/"norm2", not
    "LayerNorm" — a literal string match would miss them.
    """
    no_decay: List[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            no_decay.extend(m.parameters())
    no_decay_ids = {id(p) for p in no_decay}

    decay: List[nn.Parameter] = []
    for name, p in module.named_parameters():
        if id(p) in no_decay_ids:
            continue
        (no_decay if name.endswith("bias") else decay).append(p)
    return decay, no_decay


# ----------------------------------------------------------------------
# Container
# ----------------------------------------------------------------------

class PrimitiveHeads(nn.Module):
    """
    Bundles the attribute, object, and composition heads behind one module so
    training/eval/checkpointing treat them as a unit.

    Forward entry points (one per batch type):
        forward_attribute(visual)    → (B, 768)   train on attr-batches
        forward_object(visual)       → (B, 768)   train on obj-batches
        forward_composition(visual)  → (B, 768)   train on comp-batches
        compose(attr_embed, obj_embed) → (B, 768) reuse precomputed embeds
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
        comp_visual_dim: int = _VISUAL_DIM,
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
            visual_dim=comp_visual_dim,
        )
        instance = cls(attr_head, obj_head, comp_head)

        # Record the build hyperparameters so a checkpoint can be rebuilt with
        # the identical architecture at eval time (state_dict alone does not
        # capture structural choices like layer count or pooling).
        instance.config = dict(
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            ffn_mult=ffn_mult, pool=pool, head_dropout=head_dropout,
            comp_fusion=comp_fusion, comp_layers=comp_layers, comp_hidden=comp_hidden,
            comp_visual_dim=comp_visual_dim,
        )

        total = sum(p.numel() for p in instance.parameters() if p.requires_grad)
        logger.info("PrimitiveHeads: %.2f M trainable params total", total / 1e6)

        if device is not None:
            instance = instance.to(device)
        return instance

    # ---- Optimiser integration --------------------------------------------

    def param_groups(self, base_lr: float) -> List[dict]:
        """
        Two param groups — decay and no-decay — spanning all three heads.
        LayerNorm weight/bias and all Linear biases are excluded from weight
        decay (see `_split_decay_params`); everything else decays normally
        via the optimiser's global weight_decay. All heads are trained from
        scratch on top of a fully frozen CLIP backbone (no trainable
        projection anywhere else in the model), so no LR multiplier is
        applied.
        """
        decay, no_decay = [], []
        for head in (self.attr_head, self.obj_head, self.comp_head):
            head_decay, head_no_decay = _split_decay_params(head)
            decay += head_decay
            no_decay += head_no_decay
        return [
            {"params": decay, "lr": base_lr},
            {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
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
        visual_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse precomputed attr/obj embeddings and visual_vec (used at eval time)."""
        return self.comp_head(attr_embed, obj_embed, visual_vec)

    def forward_composition(self, visual_embeds: torch.Tensor) -> torch.Tensor:
        """
        Composition prediction from raw visual tokens.

        The attribute/object heads are run under `torch.no_grad()` so their
        outputs are detached: composition-batch gradients update ONLY the
        composition head, never the attr/obj heads.  This enforces the
        "each head trains on its own batch type" invariant.

        visual_vec is mean-pooled over patches from the frozen encoder output;
        it carries no grad so it does not break the gradient isolation.
        """
        with torch.no_grad():
            attr_embed = self.attr_head(visual_embeds)
            obj_embed = self.obj_head(visual_embeds)
        visual_vec = F.normalize(visual_embeds.mean(dim=1), dim=-1)  # (B, 1024)
        return self.comp_head(attr_embed, obj_embed, visual_vec)


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

    # (batch=2, patches=256, visual_dim=1024) — CLIP ViT-L/14 @ 224px has 256 patches
    B, P, D = 2, 256, _VISUAL_DIM
    visual = torch.randn(B, P, D)

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
