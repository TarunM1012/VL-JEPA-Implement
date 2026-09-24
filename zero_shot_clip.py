"""
Zero-shot CLIP baseline for MIT-States CZSL (no training).

Each test image is embedded with CLIP's own image encoder (CLS token ->
ln_post -> proj, i.e. clip_model.encode_image) and compared by cosine
similarity against CLIP text embeddings of "a photo of {attr} {obj}" for every
candidate pair. Calibration and metrics use the same exact curve as
evaluate.py, so the numbers are directly comparable.

This is the floor any trained head must beat: if a trained model scores below
this, training is destroying knowledge CLIP already had.

Usage:
    python zero_shot_clip.py --phase test
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.mit_states import MITStates, _load_pairs
from evaluate import exact_calibration_curve, summarize_curve
from models.clip_encoder import CLIPEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("zero_shot_clip")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["val", "test"], default="test")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--template", default="a photo of {attr} {obj}")
    p.add_argument("--split_root",
                   default="/scratch/tarunm10/datasets/mit-states/compositional-split-natural")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_encoder = CLIPEncoder.load_pretrained(device=device)
    model = clip_encoder.clip_model

    dataset = MITStates(transform=clip_encoder.preprocess, phase=args.phase,
                        splits_root=args.split_root)

    seen_pairs = set(_load_pairs(Path(args.split_root) / "train_pairs.txt"))
    phase_pairs = set(_load_pairs(Path(args.split_root) / f"{args.phase}_pairs.txt"))
    candidates = sorted(seen_pairs | phase_pairs)
    cand2idx = {pair: i for i, pair in enumerate(candidates)}
    seen_mask = torch.tensor([c in seen_pairs for c in candidates], dtype=torch.bool)
    logger.info("%d %s images | %d candidates (%d seen)",
                len(dataset), args.phase, len(candidates), int(seen_mask.sum()))

    # Text bank: one prompt per candidate pair.
    prompts = [args.template.format(attr=a, obj=o) for a, o in candidates]
    text_chunks = []
    for i in range(0, len(prompts), 256):
        text_chunks.append(F.normalize(clip_encoder.get_text_features(prompts[i:i + 256]), dim=-1))
    text_bank = torch.cat(text_chunks)                                  # (C, 768)

    # Image embeddings via CLIP's own image path (CLS token + projection).
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    all_scores, all_gt = [], []
    with torch.no_grad():
        for images, _t, _a, _o, pair_idxs in loader:
            img = model.encode_image(images.to(device).type(model.dtype)).float()
            img = F.normalize(img, dim=-1)                              # (B, 768)
            all_scores.append((img @ text_bank.T).cpu())
            all_gt.extend(pair_idxs.tolist())

    scores = torch.cat(all_scores)                                      # (N, C)
    gt = torch.tensor([cand2idx[dataset.pairs[i]] for i in all_gt], dtype=torch.long)
    gt_is_seen = torch.tensor([dataset.pairs[i] in seen_pairs for i in all_gt], dtype=torch.bool)

    m = summarize_curve(exact_calibration_curve(scores, gt, seen_mask, gt_is_seen))
    print()
    print("=" * 52)
    print(f"  Zero-shot CLIP ViT-L/14 | MIT-States {args.phase}")
    print(f"  template: \"{args.template}\"")
    print("=" * 52)
    print(f"  Best seen   : {m['best_seen']*100:6.2f}%")
    print(f"  Best unseen : {m['best_unseen']*100:6.2f}%")
    print(f"  Best HM     : {m['best_hm']*100:6.2f}%")
    print(f"  AUC         : {m['auc']*100:6.2f}")
    print("=" * 52)


if __name__ == "__main__":
    main()