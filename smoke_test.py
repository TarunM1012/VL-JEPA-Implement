"""
Standalone smoke test: full forward pass end to end, CPU only.

Mirrors train.py's model instantiation exactly (same constructor calls,
same arguments) and runs a dummy batch through CLIPEncoder + PrimitiveHeads.
"""

import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from models.clip_encoder import CLIPEncoder
from models.primitive_heads import PrimitiveHeads
from models.soft_prompt_encoder import SoftPromptTextEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_test")


def main() -> None:
    device = torch.device("cpu")
    logger.info("Device: %s", device)

    # ── Models (mirrors train.py) ────────────────────────────────────────────
    logger.info("Loading CLIPEncoder …")
    clip_encoder = CLIPEncoder.load_pretrained(device=device)
    clip_encoder.eval()

    logger.info("Building PrimitiveHeads …")
    primitive_heads = PrimitiveHeads.build(
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        pool="mean",
        device=device,
    )
    primitive_heads.eval()

    logger.info("Building SoftPromptTextEncoder …")
    soft_prompts = SoftPromptTextEncoder(clip_encoder.clip_model, n_ctx=8).to(device)
    soft_prompts.eval()

    # ── Dummy batch ───────────────────────────────────────────────────────────
    images = torch.randn(4, 3, 224, 224, device=device)
    texts = ["red", "green", "wet", "small"]

    # ── Forward pass (mirrors train.py's forward_batch) ──────────────────────
    with torch.no_grad():
        patch_tokens = clip_encoder.get_visual_features(images)
        print(f"Patch tokens shape   : {tuple(patch_tokens.shape)}")

        target_embeds = F.normalize(soft_prompts(texts, head="attr"), dim=-1)
        print(f"Text features shape  : {tuple(target_embeds.shape)}")

        attr_out = primitive_heads.forward_attribute(patch_tokens)
        print(f"Attr head output     : {tuple(attr_out.shape)}")

        obj_out = primitive_heads.forward_object(patch_tokens)
        print(f"Obj head output      : {tuple(obj_out.shape)}")

        comp_out = primitive_heads.forward_composition(patch_tokens)
        print(f"Comp head output     : {tuple(comp_out.shape)}")

    print("all shapes good")


if __name__ == "__main__":
    main()
