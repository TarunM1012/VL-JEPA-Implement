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

Residual prediction (`residual=True`)
-------------------------------------
A head built from scratch on top of frozen CLIP starts out predicting noise,
so it must relearn CLIP's image/text alignment before it can beat zero-shot
CLIP at all.  With `residual=True` a head instead predicts a *correction* on
CLIP's own image embedding:

    pred = normalise( normalise(clip_image_embed) + α · correction(tokens) )

α (`residual_scale`) is a single learnable scalar initialised to zero, so the
prediction starts out exactly equal to the normalised CLIP image embedding:
the model begins training as zero-shot CLIP and moves away from it only as far
as the InfoNCE objective pays for.

The gate is what makes the correction *small*, not just zero at step 0.  A
bare zero-initialised output projection also starts at CLIP, but nothing then
bounds how fast the correction grows — measured on real CLIP features at the
default lr=5e-5 its norm reaches 0.5 after a single AdamW step and ~2.8 (vs. a
unit-norm base) by step 50, which throws CLIP's alignment away again just a few
steps later than before.  AdamW moves each parameter by about the learning rate
per step, so a whole projection matrix shifts the output far more than the
learning rate suggests; routing the correction through one scalar makes the
departure from CLIP gradual instead.

α is not itself the distance from CLIP — the correction's own norm grows
alongside the gate — so train.py logs both α and the mean cosine between the
prediction and the CLIP embedding (`cos_clip`, 1.0 = still exactly zero-shot
CLIP), which is the honest measure.

α is excluded from weight decay for the same reason as the InfoNCE temperature:
decay would pull it toward zero independently of what the loss wants.

The frozen base carries no gradient, so this costs 1 parameter per head and no
extra backbone compute (`CLIPEncoder.get_visual_features` returns the patch
tokens and the image embedding from the same pass).

`residual=False` keeps the original behaviour (prediction is the head output
alone) so that pre-residual checkpoints still evaluate faithfully.

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


def _finalise(
    correction: torch.Tensor,
    image_embed: Optional[torch.Tensor],
    scale: Optional[nn.Parameter],
) -> torch.Tensor:
    """
    Turn a head's raw output into its L2-normalised prediction.

    `scale` is None for a non-residual head: the output *is* the prediction.
    Otherwise the output is a gated correction on CLIP's image embedding. The
    base is normalised first so the gate is relative to a unit vector — CLIP's
    raw image embeddings have norms around 10, which would otherwise swamp the
    correction entirely.
    """
    if scale is None:
        return F.normalize(correction, dim=-1)
    if image_embed is None:
        raise ValueError(
            "residual=True heads need image_embed "
            "(CLIPEncoder.get_visual_features returns it alongside the patch tokens)"
        )
    return F.normalize(F.normalize(image_embed, dim=-1) + scale * correction, dim=-1)


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
            └─ [+ normalised CLIP image embedding, if residual]
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
        residual   : predict a correction on CLIP's image embedding instead of
                     the embedding itself (see module docstring).  Requires
                     `image_embed` to be passed to forward().
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
        residual: bool = False,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads "
                f"({num_heads})"
            )
        self.pool = pool
        self.hidden_dim = hidden_dim

        # Learnable gate on the correction; None (and absent from the state
        # dict) for a non-residual head. Zero init ⇒ the prediction starts as
        # CLIP's own embedding. The transformer body gets no gradient on the
        # very first step (dL/dbody ∝ α = 0); α itself does, so from step 2 on
        # everything trains.
        if residual:
            self.residual_scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("residual_scale", None)

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
            "residual=%s | %.2f M params",
            hidden_dim, num_layers, num_heads, hidden_dim * ffn_mult,
            pool, residual, n_params / 1e6,
        )

    def forward(
        self,
        visual_embeds: torch.Tensor,
        image_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_embeds : (B, num_patches, visual_dim) patch tokens from
                            the X-encoder.
            image_embed   : (B, output_dim) CLIP's own image embedding; the
                            base the correction is added to.  Required when
                            `residual=True`, ignored otherwise.  Need not be
                            normalised — it is normalised here so the
                            correction's scale is relative to a unit vector.

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
        return _finalise(projected, image_embed, self.residual_scale)


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
        residual  : predict a correction on CLIP's image embedding instead of
                    the embedding itself (see module docstring).
    """

    def __init__(
        self,
        embed_dim: int = _SHARED_DIM,
        hidden_dim: int = _SHARED_DIM,
        num_layers: int = 3,
        fusion: Literal["concat", "sum"] = "concat",
        dropout: float = 0.0,
        visual_dim: int = _VISUAL_DIM,
        residual: bool = False,
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError(f"CompositionHead needs >=2 layers, got {num_layers}")
        self.fusion = fusion

        # Same zero-initialised gate as TransformerHead (see module docstring).
        if residual:
            self.residual_scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("residual_scale", None)

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
            "CompositionHead: fusion=%s layers=%d hidden=%d residual=%s | %.2f M params",
            fusion, num_layers, hidden_dim, residual, n_params / 1e6,
        )

    def forward(
        self,
        attr_embed: torch.Tensor,   # (B, embed_dim), L2-normalised
        obj_embed: torch.Tensor,    # (B, embed_dim), L2-normalised
        visual_vec: torch.Tensor,   # (B, visual_dim), L2-normalised mean-pooled patch tokens
        image_embed: Optional[torch.Tensor] = None,  # (B, embed_dim), CLIP image embedding
    ) -> torch.Tensor:
        """Returns (B, embed_dim) L2-normalised composition embeddings."""
        if self.fusion == "concat":
            x = torch.cat([attr_embed, obj_embed, visual_vec], dim=-1)  # (B, 2*embed_dim+visual_dim)
        else:  # "sum"
            x = attr_embed + obj_embed                       # (B, embed_dim)
        out = self.mlp(x)                                    # (B, embed_dim)
        return _finalise(out, image_embed, self.residual_scale)


# ----------------------------------------------------------------------
# Optimiser param-group splitting (decay vs. no-decay)
# ----------------------------------------------------------------------

def _split_decay_params(module: nn.Module) -> tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Split a module's parameters into (decay, no_decay) groups for AdamW.

    LayerNorm weight/bias, all Linear biases, and the residual gate should not
    be weight-decayed: decaying them pulls the value toward zero for no
    principled reason, fighting whatever the task loss is doing to it every
    step (same rationale as not decaying the InfoNCE temperature). For the
    residual gate specifically, decay would be a standing pull back toward
    pure CLIP that the loss has to fight on every step. Norms are matched by
    module type (nn.LayerNorm) rather than name substring, since
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
        skip = name.endswith("bias") or name.endswith("residual_scale")
        (no_decay if skip else decay).append(p)
    return decay, no_decay


# ----------------------------------------------------------------------
# Container
# ----------------------------------------------------------------------

class PrimitiveHeads(nn.Module):
    """
    Bundles the attribute, object, and composition heads behind one module so
    training/eval/checkpointing treat them as a unit.

    Forward entry points (one per batch type).  `image_embed` is CLIP's own
    image embedding for the same images; it is the residual base and is
    required when the heads were built with `residual=True`:
        forward_attribute(visual, image_embed)   → (B, 768)  attr-batches
        forward_object(visual, image_embed)      → (B, 768)  obj-batches
        forward_composition(visual, image_embed) → (B, 768)  comp-batches
        compose(attr_embed, obj_embed, visual_vec, image_embed) → (B, 768)
            reuse precomputed embeds
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
        residual: bool = False,
        device: Optional[torch.device] = None,
    ) -> "PrimitiveHeads":
        """
        Construct all three heads with shared transformer hyperparameters.

        `residual` defaults to False so that checkpoints written before the
        residual change — whose `head_config` has no `residual` key — rebuild
        with their original behaviour when evaluate.py splats the config.
        """
        attr_head = TransformerHead(
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            ffn_mult=ffn_mult, pool=pool, dropout=head_dropout, residual=residual,
        )
        obj_head = TransformerHead(
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            ffn_mult=ffn_mult, pool=pool, dropout=head_dropout, residual=residual,
        )
        comp_head = CompositionHead(
            hidden_dim=comp_hidden, num_layers=comp_layers, fusion=comp_fusion,
            visual_dim=comp_visual_dim, residual=residual,
        )
        instance = cls(attr_head, obj_head, comp_head)

        # Record the build hyperparameters so a checkpoint can be rebuilt with
        # the identical architecture at eval time (state_dict alone does not
        # capture structural choices like layer count or pooling).
        instance.config = dict(
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            ffn_mult=ffn_mult, pool=pool, head_dropout=head_dropout,
            comp_fusion=comp_fusion, comp_layers=comp_layers, comp_hidden=comp_hidden,
            comp_visual_dim=comp_visual_dim, residual=residual,
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

    def residual_scales(self) -> dict:
        """
        Current gate value per head — how far each head has moved from CLIP's
        own embedding. Empty dict when the heads were built non-residual.
        """
        return {
            name: head.residual_scale.item()
            for name, head in (
                ("attr", self.attr_head),
                ("obj", self.obj_head),
                ("comp", self.comp_head),
            )
            if head.residual_scale is not None
        }

    def forward_attribute(
        self,
        visual_embeds: torch.Tensor,
        image_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.attr_head(visual_embeds, image_embed)

    def forward_object(
        self,
        visual_embeds: torch.Tensor,
        image_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.obj_head(visual_embeds, image_embed)

    def compose(
        self,
        attr_embed: torch.Tensor,
        obj_embed: torch.Tensor,
        visual_vec: torch.Tensor,
        image_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse precomputed attr/obj embeddings and visual_vec (used at eval time)."""
        return self.comp_head(attr_embed, obj_embed, visual_vec, image_embed)

    def forward_composition(
        self,
        visual_embeds: torch.Tensor,
        image_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
            attr_embed = self.attr_head(visual_embeds, image_embed)
            obj_embed = self.obj_head(visual_embeds, image_embed)
        visual_vec = F.normalize(visual_embeds.mean(dim=1), dim=-1)  # (B, 1024)
        return self.comp_head(attr_embed, obj_embed, visual_vec, image_embed)


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

    # ---- Check 5: residual heads start exactly at the CLIP embedding ------
    res_heads = PrimitiveHeads.build(residual=True)
    image_embed = torch.randn(B, _SHARED_DIM) * 10.0   # CLIP embeds are not unit-norm
    base = F.normalize(image_embed, dim=-1)
    for name, out in [
        ("attr", res_heads.forward_attribute(visual, image_embed)),
        ("obj",  res_heads.forward_object(visual, image_embed)),
        ("comp", res_heads.forward_composition(visual, image_embed)),
    ]:
        assert torch.allclose(out, base, atol=1e-6), \
            f"{name} residual head does not start at the CLIP image embedding"
    print("[check 5] untrained residual heads == normalised CLIP image embedding ✓")

    # ---- Check 6: the zero-init gate still receives gradient ---------------
    res_heads.zero_grad(set_to_none=True)
    res_heads.forward_attribute(visual, image_embed).sum().backward()
    g = res_heads.attr_head.residual_scale.grad
    assert g is not None and g.abs().max() > 0, \
        "zero-initialised residual gate got no gradient — the head can never leave CLIP"
    print("[check 6] zero-init residual gate receives gradient ✓")

    # ---- Check 6b: the gate is excluded from weight decay ------------------
    decay, no_decay = _split_decay_params(res_heads.attr_head)
    gate_id = id(res_heads.attr_head.residual_scale)
    assert gate_id in {id(p) for p in no_decay}, "residual gate landed in the decay group"
    assert gate_id not in {id(p) for p in decay}
    print("[check 6b] residual gate excluded from weight decay ✓")

    # ---- Check 7: residual heads refuse to run without the base ----------
    try:
        res_heads.forward_attribute(visual)
    except ValueError:
        print("[check 7] residual head without image_embed raises ✓")
    else:
        raise AssertionError("residual head silently ran without image_embed")

    print("\nAll checks passed.")
