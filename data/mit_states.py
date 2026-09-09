"""
MIT-States dataset for Compositional Zero-Shot Learning (CZSL) evaluation.

Reference: Isola et al., "Discovering States and Transformations in Image
Collections", CVPR 2015.  The split files (train/val/test_pairs.txt) come from
the compositional-split-natural variant used by most CZSL papers.

Each sample exposes five values so a CZSL evaluator can score all
(attr, obj, pair) axes independently:

    image  : (C, H, W) — preprocessed with the caller-supplied `transform`
              (CLIP's own preprocessing pipeline; see models/clip_encoder.py).
              No frame axis: that was a V-JEPA 2 video-format holdover and
              MIT-States images are static, so CLIP's image encoder consumes
              them directly.
    text   : str, e.g. "ancient building"
    attr_idx : int index into self.attrs
    obj_idx  : int index into self.objs
    pair_idx : int index into self.pairs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal

from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── canonical dataset paths on Narval scratch ────────────────────────────────
_IMAGES_ROOT = Path("/scratch/tarunm10/datasets/release_dataset/images/")
_SPLITS_ROOT = Path("/scratch/tarunm10/datasets/mit-states/compositional-split-natural/")

Phase = Literal["train", "val", "test"]


def _load_pairs(split_file: Path) -> list[tuple[str, str]]:
    """Read 'attribute object\\n' lines and return list of (attr, obj) tuples."""
    pairs = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Each line is exactly two tokens separated by a single space.
            attr, obj = line.split(" ", 1)
            pairs.append((attr, obj))
    return pairs


class MITStates(Dataset):
    """
    MIT-States CZSL dataset.

    Args:
        phase      : "train", "val", or "test"
        transform  : image preprocessing callable, applied to a PIL RGB image
                     and returning a (C, H, W) tensor.  Pass the CLIP model's
                     own `preprocess` (returned by `clip.load(...)`, also
                     exposed as `CLIPEncoder.preprocess`) so image statistics
                     match what the frozen visual tower was trained on.
        images_root: override the default Narval path (useful for local tests)
        splits_root: override the default Narval path (useful for local tests)
    """

    def __init__(
        self,
        transform: Callable,
        phase: Phase = "train",
        images_root: Path | str = _IMAGES_ROOT,
        splits_root: Path | str = _SPLITS_ROOT,
    ) -> None:
        self.phase = phase
        self.images_root = Path(images_root)
        self.splits_root = Path(splits_root)
        self.transform = transform

        # ── load pairs for this split ─────────────────────────────────────
        split_file = self.splits_root / f"{phase}_pairs.txt"
        split_pairs = _load_pairs(split_file)

        # ── build global vocab from *all three* splits ────────────────────
        # Indices must be consistent across train/val/test so that a classifier
        # trained on train-set indices can be evaluated on val/test indices.
        all_files = [
            self.splits_root / "train_pairs.txt",
            self.splits_root / "val_pairs.txt",
            self.splits_root / "test_pairs.txt",
        ]
        all_pairs: list[tuple[str, str]] = []
        for f in all_files:
            all_pairs.extend(_load_pairs(f))

        # Deduplicate while preserving sorted order for reproducibility.
        self.attrs: list[str] = sorted(set(a for a, _ in all_pairs))
        self.objs: list[str] = sorted(set(o for _, o in all_pairs))
        self.pairs: list[tuple[str, str]] = sorted(set(all_pairs))

        self._attr2idx = {a: i for i, a in enumerate(self.attrs)}
        self._obj2idx = {o: i for i, o in enumerate(self.objs)}
        self._pair2idx = {p: i for i, p in enumerate(self.pairs)}

        # ── build sample list, filtering out images that don't exist ──────
        # MIT-States image folders are named "attr obj" (with a space).
        self.samples: list[tuple[Path, str, str]] = []
        missing_dirs: set[str] = set()

        for attr, obj in split_pairs:
            folder = self.images_root / f"{attr} {obj}"
            if not folder.is_dir():
                missing_dirs.add(f"{attr} {obj}")
                continue
            for img_file in sorted(folder.iterdir()):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_file, attr, obj))

        if missing_dirs:
            import warnings
            warnings.warn(
                f"MITStates ({phase}): {len(missing_dirs)} folder(s) not found "
                f"under {self.images_root} — {sorted(missing_dirs)[:5]}{'...' if len(missing_dirs) > 5 else ''}",
                stacklevel=2,
            )

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, attr, obj = self.samples[idx]

        # Load and preprocess image.
        img = Image.open(img_path).convert("RGB")
        image = self.transform(img)  # (C, H, W)

        text = f"{attr} {obj}"
        attr_idx = self._attr2idx[attr]
        obj_idx = self._obj2idx[obj]
        pair_idx = self._pair2idx[(attr, obj)]

        return image, text, attr_idx, obj_idx, pair_idx

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def num_attrs(self) -> int:
        return len(self.attrs)

    @property
    def num_objs(self) -> int:
        return len(self.objs)

    @property
    def num_pairs(self) -> int:
        return len(self.pairs)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from models.clip_encoder import CLIPEncoder

    print("Loading CLIPEncoder (for its preprocess transform) …")
    encoder = CLIPEncoder.load_pretrained()

    print("Loading MITStates train split …")
    ds = MITStates(transform=encoder.preprocess, phase="train")
    print(f"  samples : {len(ds)}")
    print(f"  attrs   : {ds.num_attrs}")
    print(f"  objects : {ds.num_objs}")
    print(f"  pairs   : {ds.num_pairs}")

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    images, texts, attr_idxs, obj_idxs, pair_idxs = next(iter(loader))

    print(f"\nFirst batch:")
    print(f"  images    : {tuple(images.shape)}   dtype={images.dtype}")
    print(f"  texts     : {list(texts)}")
    print(f"  attr_idxs : {attr_idxs.tolist()}")
    print(f"  obj_idxs  : {obj_idxs.tolist()}")
    print(f"  pair_idxs : {pair_idxs.tolist()}")

    # Shape contract: (B, C, H, W) with C=3, H=W=224 (CLIP ViT-L/14 input size).
    B, C, H, W = images.shape
    assert C == 3 and H == 224 and W == 224, (
        f"Unexpected image shape: {tuple(images.shape)}"
    )
    print("\nShape check passed.")
