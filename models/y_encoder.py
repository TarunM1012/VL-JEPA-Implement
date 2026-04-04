"""
Y-Encoder: text encoder for VL-JEPA target embeddings.

Design summary
--------------
Backbone : EmbeddingGemma-300M  (google/embeddinggemma-300m, frozen)
Head     : nn.Linear(hidden_size -> 1536)  (trainable, LR × 0.05)
Pooling  : attention-mask-weighted mean pool over the token sequence
Output   : L2-normalised (batch, 1536) vectors in the shared embedding space
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

_HF_MODEL_ID = "google/embeddinggemma-300m"
_MAX_LENGTH   = 512    # paper cap; keeps memory bounded and matches the model's context window
_SHARED_DIM   = 1536   # shared visual–language embedding space dimension from the paper
_LR_MULT      = 0.05   # projection head uses 5% of the base LR (per paper)


class YEncoder(nn.Module):
    """
    Encodes text targets into the VL-JEPA shared embedding space.

    Only the linear projection head is trainable; the backbone is frozen so
    the pre-trained sentence representations are preserved while the head
    learns to align with the visual embedding space.

    Typical usage
    -------------
    encoder = YEncoder.load_pretrained(device=device)
    optimizer = torch.optim.AdamW(encoder.param_groups(base_lr=1e-4))
    embeddings = encoder(["a cat on a mat", "the sky is blue"])  # (2, 1536)
    """

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.backbone  = backbone
        self.tokenizer = tokenizer

        # Linear projection: maps backbone hidden dim to shared space.
        # No bias is common in embedding projection heads; it keeps the
        # origin meaningful after L2 normalisation.
        self.projection = nn.Linear(hidden_size, _SHARED_DIM, bias=False)

        # Freeze the backbone immediately on construction so there is no
        # risk of accidentally training it if the caller forgets.
        for p in self.backbone.parameters():
            p.requires_grad = False

        logger.info(
            "YEncoder: backbone frozen | projection head trainable (%d → %d)",
            hidden_size, _SHARED_DIM,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load_pretrained(cls, device: torch.device | None = None) -> "YEncoder":
        """Load EmbeddingGemma-300M weights from HuggingFace Hub."""
        logger.info("YEncoder: downloading %s", _HF_MODEL_ID)

        tokenizer = AutoTokenizer.from_pretrained(_HF_MODEL_ID)
        backbone  = AutoModel.from_pretrained(_HF_MODEL_ID)

        # hidden_size is read from the config so this file stays correct even
        # if the checkpoint is swapped for a different Gemma variant.
        hidden_size = backbone.config.hidden_size
        logger.info("YEncoder: loaded (hidden_size=%d)", hidden_size)

        if device is not None:
            backbone = backbone.to(device)

        return cls(backbone, tokenizer, hidden_size)

    # ------------------------------------------------------------------
    # Optimiser integration
    # ------------------------------------------------------------------

    def param_groups(self, base_lr: float) -> List[dict]:
        """
        Return an optimizer-ready param-group list.

        The backbone is frozen, so it is excluded entirely — there is nothing
        to optimise there.  The projection head uses a reduced LR (× 0.05)
        because it sits on top of a strong pre-trained backbone; a full LR
        would destabilise the early projection alignment.
        """
        return [
            {
                "params": self.projection.parameters(),
                "lr":     base_lr * _LR_MULT,
            }
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the attention-mask-weighted mean over the sequence dimension.

        Using the mask ensures padding tokens do not dilute the representation.
        Shape: (batch, seq_len, hidden) → (batch, hidden)
        """
        mask   = attention_mask.unsqueeze(-1).float()           # (B, L, 1)
        summed = (last_hidden_state * mask).sum(dim=1)          # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)                # (B, 1)
        return summed / counts

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Args:
            texts: list of B strings (answers, captions, or other text targets)

        Returns:
            Tensor of shape (B, 1536) — L2-normalised embeddings in the
            shared visual–language space.
        """
        # Tokenise on the fly so the caller never has to touch tokenizer details.
        # padding=True handles variable-length batches; truncation enforces the
        # 512-token budget from the paper.
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=_MAX_LENGTH,
            return_tensors="pt",
        )

        # Move to the same device as the backbone without hard-coding it.
        device   = next(self.backbone.parameters()).device
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Backbone is frozen → no gradients needed through it.
        with torch.no_grad():
            out = self.backbone(**encoding)

        pooled    = self._mean_pool(out.last_hidden_state, encoding["attention_mask"])
        projected = self.projection(pooled)

        # L2 normalisation makes dot-product equivalent to cosine similarity,
        # which is what the prediction loss in VL-JEPA operates over.
        return F.normalize(projected, dim=-1)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    texts = [
        "A dog is running across a grassy field.",
        "The astronaut floats above the Earth.",
    ]

    encoder = YEncoder.load_pretrained()

    with torch.no_grad():
        out = encoder(texts)

    print(f"Input  : {len(texts)} strings")
    print(f"Output shape : {tuple(out.shape)}")

    expected = (len(texts), _SHARED_DIM)
    assert tuple(out.shape) == expected, (
        f"Shape mismatch: got {tuple(out.shape)}, expected {expected}"
    )
    print(f"Shape check passed: {expected}")
