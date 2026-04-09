"""
Predictor: VL-JEPA cross-modal predictor.

Design summary
--------------
Backbone    : LLaMA transformer layers 8–15 (0-indexed), extracted from
              meta-llama/Llama-3.2-1B.  This is 8 of the 16 layers — the
              second half.  Each layer is ~60.8 M params (Q/K/V/O attention
              + SwiGLU MLP), giving 8 × 60.8 M ≈ 486 M ≈ 490 M stated in
              the paper.  The first half of LLaMA is discarded; it handles
              low-level token syntax that is not relevant to multi-modal fusion.

Text embed  : LLaMA's embed_tokens table (FROZEN — ~262 M params excluded
              from the 490 M trainable count).  Freezing preserves LLaMA's
              learned token geometry and avoids the cost of training it.

Visual in   : (batch, frames, num_patches, 1024)
              Frames are flattened into a single long sequence before being
              projected linearly into LLaMA's hidden dim (2048).

Text in     : List[str] → tokenise → embed_tokens → (batch, seq_len, 2048)

Attention   : Fully BIDIRECTIONAL (no causal mask).  An additive -inf mask
              is applied to padding key positions so they are ignored.
              The causal triangular mask used in autoregressive LLaMA is
              deliberately omitted — the predictor is an encoder, not a
              next-token generator.

Pooling     : Masked mean over non-PAD tokens → (batch, 2048).
              Visual tokens are never padded; only trailing text pad tokens
              are excluded.

Head        : Linear(2048, 1536, bias=False) → L2-normalise
Output      : (batch, 1536) in the shared visual–language embedding space
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HF_MODEL_ID  = "meta-llama/Llama-3.2-1B"
_LAYER_START  = 8      # inclusive, 0-indexed
_LAYER_END    = 16     # exclusive  →  layers[8:16] = 8 layers ≈ 486 M ≈ 490 M
_VISUAL_DIM   = 1024   # X-encoder patch-token dimension (ViT-L)
_SHARED_DIM   = 1536   # shared visual–language embedding space
_MAX_TEXT_LEN = 512    # truncation cap (matches YEncoder and paper)


class Predictor(nn.Module):
    """
    Cross-modal predictor for VL-JEPA.

    Takes visual patch sequences from the X-encoder and a text query, fuses
    them through bidirectional LLaMA transformer layers, and returns an
    L2-normalised embedding in the same 1536-dim space as the Y-encoder so
    that the prediction loss can be computed as cosine similarity.

    Typical usage
    -------------
    predictor = Predictor.load_pretrained(device=device)
    optimizer = torch.optim.AdamW(predictor.param_groups(base_lr=1e-4))
    pred = predictor(visual_embeds, ["What is happening in the video?"])
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        embed_tokens: nn.Embedding,
        norm: nn.Module,
        rotary_emb: Optional[nn.Module],
        tokenizer,
        hidden_size: int,
    ) -> None:
        super().__init__()

        # --- Text embedding substrate (frozen) ----------------------------
        # embed_tokens maps token IDs → dense vectors.  It is frozen because:
        #   (a) LLaMA's vocabulary representations are already well-calibrated
        #       and align with the weights of the layers we borrow;
        #   (b) training it would add ~262 M params with negligible benefit;
        #   (c) the 490 M trainable count in the paper refers only to the
        #       transformer layers themselves.
        self.embed_tokens = embed_tokens
        for p in self.embed_tokens.parameters():
            p.requires_grad = False

        # --- LLaMA transformer layers (trainable) -------------------------
        self.llama_layers = layers  # nn.ModuleList → registered for state_dict / .to()
        self.norm = norm             # LLaMA's final RMSNorm; small but trained jointly

        # rotary_emb is LLaMA's shared position-encoding module (no learnable
        # params).  We keep it as an nn.Module so .to(device) propagates.
        # It may not exist in older transformers checkpoints — handled below.
        self.rotary_emb = rotary_emb if rotary_emb is not None else None
        if self.rotary_emb is not None:
            self.rotary_emb = rotary_emb  # already set; explicit for clarity

        self.tokenizer   = tokenizer
        self.hidden_size = hidden_size

        # --- Visual projection (trainable) --------------------------------
        # Projects patch tokens from ViT-L dim (1024) to LLaMA hidden dim
        # (2048).  bias=False is conventional for embedding-space projections:
        # the origin remains meaningful after L2 normalisation, and a bias
        # would shift every token by the same vector regardless of content.
        self.visual_proj = nn.Linear(_VISUAL_DIM, hidden_size, bias=False)

        # --- Output projection head (trainable) ---------------------------
        # Maps the pooled LLaMA representation (2048) into the shared
        # 1536-dim space that the Y-encoder also targets.  No bias for the
        # same reason as visual_proj.
        self.projection_head = nn.Linear(hidden_size, _SHARED_DIM, bias=False)

        logger.info(
            "Predictor: %d LLaMA layers | hidden=%d | visual_proj %d→%d | head %d→%d",
            len(self.llama_layers),
            hidden_size,
            _VISUAL_DIM, hidden_size,
            hidden_size, _SHARED_DIM,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load_pretrained(
        cls,
        device: Optional[torch.device] = None,
    ) -> "Predictor":
        """
        Load Llama-3.2-1B from HuggingFace, extract layers 8–15, discard
        the rest, and return a ready-to-use Predictor.

        attn_implementation="eager" is forced so that the standard
        attention path is used.  Flash-Attention and SDPA both expect a
        cache_position tensor that is only supplied by LlamaModel.forward;
        we bypass LlamaModel and call layers directly, so eager is simpler.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Predictor: downloading %s", _HF_MODEL_ID)

        tokenizer = AutoTokenizer.from_pretrained(_HF_MODEL_ID)
        # LLaMA's tokenizer has no dedicated pad token; use EOS as the
        # padding sentinel.  This is safe because we mask padded positions
        # out of both attention and pooling.
        tokenizer.pad_token = tokenizer.eos_token

        llama = AutoModelForCausalLM.from_pretrained(
            _HF_MODEL_ID,
            attn_implementation="eager",
            dtype=torch.float32,
        )

        llama_model = llama.model
        hidden_size = llama.config.hidden_size   # 2048 for Llama-3.2-1B

        # Extract exactly what we need, then drop the full model.
        # Python's reference counting keeps sub-modules alive after del llama.
        layers      = nn.ModuleList(
            list(llama_model.layers[_LAYER_START:_LAYER_END])
        )
        embed_tokens = llama_model.embed_tokens   # vocabulary embedding table
        norm         = llama_model.norm           # final RMSNorm
        rotary_emb   = getattr(llama_model, "rotary_emb", None)

        del llama   # free ~1.24 GB of weights we no longer need

        logger.info(
            "Predictor: extracted layers[%d:%d], hidden_size=%d",
            _LAYER_START, _LAYER_END, hidden_size,
        )

        instance = cls(layers, embed_tokens, norm, rotary_emb, tokenizer, hidden_size)

        if device is not None:
            instance = instance.to(device)

        return instance

    # ------------------------------------------------------------------
    # Optimiser integration
    # ------------------------------------------------------------------

    def param_groups(self, base_lr: float) -> List[dict]:
        """
        Return two optimizer-ready param groups.

        embed_tokens is frozen and excluded entirely.  The LLaMA layers and
        the two projection heads all use the full base_lr — the paper does
        not apply a separate multiplier to the predictor's head.
        """
        backbone_params = (
            list(self.llama_layers.parameters())
            + list(self.norm.parameters())
        )
        head_params = (
            list(self.visual_proj.parameters())
            + list(self.projection_head.parameters())
        )
        return [
            {"params": backbone_params, "lr": base_lr},
            {"params": head_params,     "lr": base_lr},
        ]

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_additive_mask(
        padding_mask: torch.Tensor,   # (B, S) — 1 for real token, 0 for pad
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Convert a binary padding mask to a 4-D additive attention mask.

        Output shape: (B, 1, 1, S)

        Entries are 0.0 for real tokens (attend freely) and -inf for padding
        tokens (suppressed by softmax).  The shape broadcasts over both the
        attention-head dimension and the query-position dimension, so every
        query position applies the same key-padding mask — correct for an
        encoder where no causal constraint exists.

        No upper-triangular causal mask is added.  This is intentional:
        the predictor must attend to all tokens simultaneously to fuse visual
        and text information, which requires bidirectional attention.
        """
        # 0.0 where the token is real, 1.0 where it is padding
        inv_mask = 1.0 - padding_mask.float()
        # Multiply by the most negative finite float for this dtype so that
        # softmax(q·k + mask) ≈ 0 for padding positions
        additive = inv_mask * torch.finfo(dtype).min
        return additive.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, S)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        visual_embeds: torch.Tensor,   # (B, num_frames, num_patches, 1024)
        texts: List[str],              # length-B list of query strings
    ) -> torch.Tensor:
        """
        Args:
            visual_embeds : Patch token sequences from the X-encoder,
                            shape (batch, frames, num_patches, 1024).
            texts         : One query string per sample in the batch.

        Returns:
            (batch, 1536) L2-normalised embeddings in the shared
            visual–language space, ready for cosine loss against Y-encoder
            outputs.
        """
        B, num_frames, P, _ = visual_embeds.shape
        device               = visual_embeds.device

        # ---- Visual branch -----------------------------------------------
        # Flatten frames × patches into one sequence axis.  All patch tokens
        # from all frames participate in attention together, allowing the
        # transformer to pool temporal information freely.
        vis_flat = visual_embeds.view(B, num_frames * P, _VISUAL_DIM)   # (B, num_frames*P, 1024)
        vis_proj = self.visual_proj(vis_flat)                             # (B, num_frames*P, 2048)

        # Visual tokens are never padded — mask is all ones.
        vis_mask = torch.ones(B, num_frames * P, dtype=torch.long, device=device)

        # ---- Text branch -------------------------------------------------
        # Tokenise all queries in one batched call.  padding=True aligns
        # sequences; truncation enforces the 512-token budget.
        encoding   = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=_MAX_TEXT_LEN,
            return_tensors="pt",
        )
        input_ids  = encoding["input_ids"].to(device)               # (B, T)
        text_mask  = encoding["attention_mask"].to(device)          # (B, T)

        # Embedding table is frozen — no gradients needed here.
        with torch.no_grad():
            text_embeds = self.embed_tokens(input_ids)              # (B, T, 2048)

        # ---- Concatenate visual prefix and text query --------------------
        # Visual tokens are prepended (image-before-text) following standard
        # vision-language practice.  The transformer sees the full visual
        # context before the question, which stabilises early cross-modal
        # attention patterns during training.
        hidden         = torch.cat([vis_proj,  text_embeds], dim=1)  # (B, num_frames*P+T, 2048)
        combined_mask  = torch.cat([vis_mask,  text_mask],   dim=1)  # (B, num_frames*P+T)

        # ---- Bidirectional attention mask --------------------------------
        # 4-D additive mask: 0.0 for real tokens, -inf for padding tokens.
        # No causal upper-triangle — the predictor is an encoder.
        attn_mask_4d = self._build_additive_mask(
            combined_mask, dtype=hidden.dtype
        )                                                             # (B, 1, 1, num_frames*P+T)

        # ---- Position IDs ------------------------------------------------
        # Starting positions at 0 regardless of which LLaMA layers were
        # borrowed.  RoPE angles are computed fresh from position index 0,
        # treating the predictor as a standalone encoder with its own
        # position counting.
        seq_len      = hidden.size(1)
        position_ids = (
            torch.arange(seq_len, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )                                                             # (B, S)

        # ---- Pre-compute rotary embeddings -------------------------------
        # Some transformers versions return cos/sin of shape (..., head_dim//2)
        # while the position_embeddings API expects full head_dim.  Expand if needed.
        cos, sin = self.rotary_emb(hidden, position_ids)
        head_dim = self.llama_layers[0].self_attn.head_dim
        if cos.shape[-1] == head_dim // 2:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
        position_embeddings = (cos, sin)

        # ---- Transformer pass --------------------------------------------
        for layer in self.llama_layers:
            kwargs: dict = dict(
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
            out    = layer(hidden, **kwargs)
            hidden = out[0]                                          # (B, S, 2048)

        hidden = self.norm(hidden)                                   # (B, S, 2048)

        # ---- Masked mean pooling ----------------------------------------
        # Average over all non-padding positions (visual tokens are always
        # included; trailing text-pad tokens are excluded by combined_mask).
        # Dividing by token count normalises for variable sequence lengths.
        mask_f = combined_mask.unsqueeze(-1).float()                 # (B, S, 1)
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
        #                                                             (B, 2048)

        # ---- Project and L2-normalise -----------------------------------
        # L2 normalisation makes dot-product = cosine similarity, matching
        # the loss function used in VL-JEPA (and consistent with YEncoder).
        projected = self.projection_head(pooled)                     # (B, 1536)
        return F.normalize(projected, dim=-1)


# ----------------------------------------------------------------------
# Smoke test  —  python models/predictor.py --smoke
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a quick shape-check forward pass and exit.",
    )
    args = parser.parse_args()

    if not args.smoke:
        parser.print_help()
        raise SystemExit(0)

    # (batch=1, frames=2, patches=196, visual_dim=1024)
    # 196 = (224/16)^2 — standard ViT-L/16 patch count at 224 px
    B, num_frames, P, D = 1, 2, 196, _VISUAL_DIM
    visual_embeds = torch.randn(B, num_frames, P, D)
    texts         = ["What is happening in the video?"]

    print(f"Visual input : {tuple(visual_embeds.shape)}")
    print(f"Text input   : {texts}")

    predictor = Predictor.load_pretrained()
    predictor.eval()

    with torch.no_grad():
        out = predictor(visual_embeds, texts)

    print(f"Output shape : {tuple(out.shape)}")

    expected = (B, _SHARED_DIM)
    assert tuple(out.shape) == expected, (
        f"Shape mismatch: got {tuple(out.shape)}, expected {expected}"
    )
    print(f"Shape check passed: {expected}")
