"""
CLIP encoder: frozen visual + text backbone for VL-JEPA target/context embeddings.

Design summary
--------------
Backbone   : CLIP ViT-L/14 (openai/CLIP package, NOT HuggingFace transformers)
Visual out : patch tokens from the final transformer block, BEFORE the CLS
             projection layer — shape (B, num_patches, 1024) for ViT-L/14.
Text out   : CLIP's own projected text embedding — shape (B, embed_dim), where
             embed_dim is read off the loaded checkpoint (768 for ViT-L/14;
             deliberately not hardcoded so this stays correct if the checkpoint
             is swapped for a different CLIP variant).
Both encoders are frozen; forward passes run under torch.no_grad().
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_CLIP_MODEL_NAME = "ViT-L/14"
_CLIP_DOWNLOAD_ROOT = "/lustre06/project/6001346/tarunm10/.cache/clip"
_VISUAL_DIM = 1024   # ViT-L/14 visual transformer width (pre-projection)

# ── Optional CSP-style soft-prompt hook (vanilla ablation baseline) ─────────
# Opt-in only, via the VLJEPA_SOFT_PROMPT_CKPT env var. When unset (true for
# every existing v3-CLIP r1 run), get_text_features() below is byte-identical
# to plain CLIP text encoding. This exists purely so evaluate.py -- which
# calls get_text_features() directly on raw strings with no hook of its own,
# and which this repo's instructions say must not be edited -- can still
# score a trained soft prompt: train_vanilla.py / scripts/eval_vanilla_narval.sh
# set the env var before invoking `python evaluate.py`. See
# models/soft_prompt_vanilla.py for the prompt encoder itself.
_ACTIVE_SOFT_PROMPT: Optional[nn.Module] = None
_SOFT_PROMPT_CKPT_LOADED: Optional[str] = None


def _get_active_soft_prompt(clip_model: nn.Module) -> Optional[nn.Module]:
    global _ACTIVE_SOFT_PROMPT, _SOFT_PROMPT_CKPT_LOADED

    ckpt_path = os.environ.get("VLJEPA_SOFT_PROMPT_CKPT")
    if not ckpt_path:
        return None
    if _ACTIVE_SOFT_PROMPT is not None and _SOFT_PROMPT_CKPT_LOADED == ckpt_path:
        return _ACTIVE_SOFT_PROMPT

    from models.soft_prompt_vanilla import VanillaSoftPromptEncoder

    logger.info("CLIPEncoder: loading soft prompt from VLJEPA_SOFT_PROMPT_CKPT=%s", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_ctx = state["soft_prompt_config"]["n_ctx"]

    soft_prompt = VanillaSoftPromptEncoder(clip_model, n_ctx=n_ctx)
    soft_prompt.load_state_dict(state["soft_prompt"])
    soft_prompt.eval()
    for p in soft_prompt.parameters():
        p.requires_grad = False

    _ACTIVE_SOFT_PROMPT = soft_prompt
    _SOFT_PROMPT_CKPT_LOADED = ckpt_path
    return soft_prompt


class CLIPEncoder(nn.Module):
    """
    Wraps a frozen CLIP ViT-L/14 model to expose two feature extractors:

      get_visual_features(images) -> (B, num_patches, 1024) patch tokens from
          the last transformer block, before CLIP's CLS projection head.
      get_text_features(texts)    -> (B, embed_dim) CLIP's own projected text
          embedding (embed_dim read from the checkpoint; 768 for ViT-L/14).

    Both are frozen (requires_grad=False) and run under torch.no_grad().

    Typical usage
    -------------
    encoder = CLIPEncoder.load_pretrained(device=device)
    visual_tokens = encoder.get_visual_features(images)   # (B, P, 1024)
    text_embeds   = encoder.get_text_features(texts)       # (B, embed_dim)
    """

    def __init__(self, clip_model: nn.Module, preprocess) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.preprocess = preprocess

        self.visual_dim = _VISUAL_DIM
        self.text_dim = clip_model.text_projection.shape[1]

        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        logger.info(
            "CLIPEncoder: %s frozen | visual_dim=%d (pre-projection) | text_dim=%d",
            _CLIP_MODEL_NAME, self.visual_dim, self.text_dim,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load_pretrained(cls, device: torch.device | None = None) -> "CLIPEncoder":
        """Load CLIP ViT-L/14 from the shared Narval cache via the openai/CLIP package."""
        import clip

        logger.info(
            "CLIPEncoder: loading %s from %s", _CLIP_MODEL_NAME, _CLIP_DOWNLOAD_ROOT
        )
        clip_model, preprocess = clip.load(
            _CLIP_MODEL_NAME,
            device=device if device is not None else "cpu",
            download_root=_CLIP_DOWNLOAD_ROOT,
        )
        return cls(clip_model, preprocess)

    # ------------------------------------------------------------------
    # Visual features
    # ------------------------------------------------------------------

    def get_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W), already CLIP-preprocessed.

        Returns:
            (B, num_patches, 1024) — patch tokens from the final transformer
            block of the visual tower, CLS token dropped, before CLIP's
            output projection (ln_post + proj are intentionally skipped so
            the primitive heads see raw pre-projection patch features).
        """
        visual = self.clip_model.visual
        dtype = visual.conv1.weight.dtype

        with torch.no_grad():
            x = images.to(dtype)
            x = visual.conv1(x)                                   # (B, width, grid, grid)
            x = x.reshape(x.shape[0], x.shape[1], -1)              # (B, width, grid**2)
            x = x.permute(0, 2, 1)                                 # (B, grid**2, width)
            cls_token = visual.class_embedding.to(dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=dtype, device=x.device
            )
            x = torch.cat([cls_token, x], dim=1)                   # (B, 1+grid**2, width)
            x = x + visual.positional_embedding.to(dtype)
            x = visual.ln_pre(x)

            x = x.permute(1, 0, 2)                                 # NLD -> LND
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)                                 # LND -> NLD

            patch_tokens = x[:, 1:, :]                             # drop CLS token
            return patch_tokens.float()

    # ------------------------------------------------------------------
    # Text features
    # ------------------------------------------------------------------

    def get_text_features(self, texts: list) -> torch.Tensor:
        """
        Args:
            texts: list of B strings.

        Returns:
            (B, text_dim) — CLIP's own projected text embedding, unless a
            trained soft prompt is active (VLJEPA_SOFT_PROMPT_CKPT env var —
            see the module-level hook above), in which case that prompt's
            output is returned instead.
            Not L2-normalised (matches CLIP's native output; callers that
            need cosine similarity should normalise explicitly).
        """
        import clip

        soft_prompt = _get_active_soft_prompt(self.clip_model)
        if soft_prompt is not None:
            with torch.no_grad():
                return soft_prompt(texts).float()

        device = next(self.clip_model.parameters()).device
        tokens = clip.tokenize(texts, truncate=True).to(device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)

        return text_features.float()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = CLIPEncoder.load_pretrained(device=device)

    images = torch.randn(2, 3, 224, 224, device=device)
    texts = ["a red car", "a green apple"]

    visual_tokens = encoder.get_visual_features(images)
    text_embeds = encoder.get_text_features(texts)

    print(f"Visual tokens shape : {tuple(visual_tokens.shape)}")
    print(f"Text embeds shape   : {tuple(text_embeds.shape)}")

    assert visual_tokens.shape[0] == 2 and visual_tokens.shape[2] == _VISUAL_DIM, (
        f"Visual shape mismatch: {tuple(visual_tokens.shape)}"
    )
    assert tuple(text_embeds.shape) == (2, encoder.text_dim), (
        f"Text shape mismatch: got {tuple(text_embeds.shape)}, expected (2, {encoder.text_dim})"
    )
    print("Shape check passed.")
