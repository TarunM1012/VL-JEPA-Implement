"""
End-to-end smoke test for the CLIP-backbone swap (v3-CLIP branch).

Confirms shapes are correct all the way through, on 10 synthetic samples:

    image  ──► CLIPEncoder.get_visual_features ──► attr/obj head ──► (10, 768)
    text   ──► CLIPEncoder.get_text_features   ──►                    (10, 768)
    (pred, target) ──► InfoNCELoss ──► finite scalar

Uses random tensors/strings rather than the real MIT-States dataset so it
runs anywhere the CLIP checkpoint is reachable, without needing the Narval
scratch image tree. Requires the `openai/CLIP` package and the ViT-L/14
checkpoint; both are skipped gracefully if unavailable (e.g. on a local
dev box without the Narval cache mounted).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

clip = pytest.importorskip("clip", reason="openai/CLIP package not installed")

from models.clip_encoder import CLIPEncoder, _CLIP_DOWNLOAD_ROOT, _CLIP_MODEL_NAME
from models.loss import InfoNCELoss
from models.primitive_heads import PrimitiveHeads

N = 10


def _clip_weights_available() -> bool:
    try:
        clip.load(_CLIP_MODEL_NAME, device="cpu", download_root=_CLIP_DOWNLOAD_ROOT)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _clip_weights_available(),
    reason=f"CLIP {_CLIP_MODEL_NAME} checkpoint not reachable at {_CLIP_DOWNLOAD_ROOT}",
)
def test_clip_end_to_end_smoke() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = CLIPEncoder.load_pretrained(device=device)
    heads = PrimitiveHeads.build(device=device)
    loss_fn = InfoNCELoss().to(device)

    images = torch.randn(N, 3, 224, 224, device=device)
    texts = [f"sample text {i}" for i in range(N)]

    # ── Visual path ────────────────────────────────────────────────────────
    patch_tokens = encoder.get_visual_features(images)
    assert patch_tokens.shape[0] == N
    assert patch_tokens.shape[2] == encoder.visual_dim == 1024

    attr_embed = heads.forward_attribute(patch_tokens)
    obj_embed = heads.forward_object(patch_tokens)
    comp_embed = heads.forward_composition(patch_tokens)
    for name, embed in [("attr", attr_embed), ("obj", obj_embed), ("comp", comp_embed)]:
        assert embed.shape == (N, encoder.text_dim), f"{name}: {tuple(embed.shape)}"
        norms = embed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(N, device=device), atol=1e-4), (
            f"{name} not L2-normalised: {norms.tolist()}"
        )

    # ── Text path ──────────────────────────────────────────────────────────
    target_embeds = torch.nn.functional.normalize(encoder.get_text_features(texts), dim=-1)
    assert target_embeds.shape == (N, encoder.text_dim)

    # ── Loss ───────────────────────────────────────────────────────────────
    loss = loss_fn(attr_embed, target_embeds)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


if __name__ == "__main__":
    test_clip_end_to_end_smoke()
    print("CLIP end-to-end smoke test passed.")
