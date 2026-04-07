import logging

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_HF_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
_TIMM_FALLBACK = "vit_large_patch16_224"
_EMBED_DIM = 1024


def _load_vjepa2():
    """Try to load V-JEPA 2 from HuggingFace. Returns (model, backend) or raises."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(_HF_MODEL_ID)
    return model, "vjepa2"


def _load_timm(img_size: int = 256):
    model = timm.create_model(
        _TIMM_FALLBACK, pretrained=False, num_classes=0, img_size=img_size
    )
    return model, "timm"


class VisualEncoder(nn.Module):
    """
    Encodes video frames to patch token sequences.

    Each frame is encoded independently.  Output shape is always
    (batch, frames, num_patches, embed_dim).

    Instantiate directly for a lightweight timm backbone (no HF download).
    Use VisualEncoder.load_pretrained() to get V-JEPA 2 weights with
    automatic fallback to timm when HF is unavailable.
    """

    def __init__(self, backbone: nn.Module, backend: str, is_frozen: bool = False):
        super().__init__()
        self.backbone = backbone
        self._backend = backend
        self.embed_dim = _EMBED_DIM
        if is_frozen:
            self.freeze()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_timm(cls, img_size: int = 256, is_frozen: bool = False) -> "VisualEncoder":
        """Lightweight constructor — no HF download, random weights."""
        backbone, backend = _load_timm(img_size)
        logger.info("VisualEncoder: loaded timm %s (img_size=%d)", _TIMM_FALLBACK, img_size)
        return cls(backbone, backend, is_frozen=is_frozen)

    @classmethod
    def load_pretrained(cls, is_frozen: bool = True) -> "VisualEncoder":
        """
        Load V-JEPA 2 weights from HuggingFace.
        Falls back to timm vit_large_patch16_224 if HF is unavailable.
        """
        try:
            backbone, backend = _load_vjepa2()
            logger.info("VisualEncoder: loaded V-JEPA 2 from %s", _HF_MODEL_ID)
        except Exception as exc:
            logger.warning(
                "VisualEncoder: could not load V-JEPA 2 from HF (%s); "
                "falling back to timm %s",
                exc,
                _TIMM_FALLBACK,
            )
            backbone, backend = _load_timm()
        return cls(backbone, backend, is_frozen=is_frozen)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("VisualEncoder: backbone frozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, frames, channels, height, width)
        Returns:
            (batch, frames, num_patches, embed_dim)
        """
        B, F, C, H, W = x.shape

        if self._backend == "vjepa2":
            # V-JEPA 2 accepts (batch, frames, channels, height, width) natively.
            # last_hidden_state: (B, 1 + F*num_patches, embed_dim) — index 0 is CLS.
            tokens = self.backbone(pixel_values=x).last_hidden_state
            patch_tokens = tokens[:, 1:, :]          # strip CLS
            num_patches = patch_tokens.size(1) // F
            return patch_tokens.view(B, F, num_patches, self.embed_dim)

        else:  # timm
            # Flatten frames into batch, encode, then unflatten.
            x_flat = x.view(B * F, C, H, W)
            # forward_features: (B*F, 1+num_patches, embed_dim)
            tokens = self.backbone.forward_features(x_flat)
            patch_tokens = tokens[:, 1:, :]          # strip CLS
            num_patches = patch_tokens.size(1)
            return patch_tokens.view(B, F, num_patches, self.embed_dim)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run with a small tensor")
    args = parser.parse_args()

    if args.smoke:
        B, F, C, H, W = 1, 2, 3, 224, 224
    else:
        B, F, C, H, W = 2, 8, 3, 256, 256

    x = torch.randn(B, F, C, H, W)
    encoder = VisualEncoder.from_timm(img_size=H)

    with torch.no_grad():
        out = encoder(x)

    print(f"Input  shape : {tuple(x.shape)}")
    print(f"Output shape : {tuple(out.shape)}")

    expected = (B, F, (H // 16) * (W // 16), _EMBED_DIM)
    assert tuple(out.shape) == expected, f"Shape mismatch: got {tuple(out.shape)}, expected {expected}"
    print(f"Shape check passed: {expected}")
