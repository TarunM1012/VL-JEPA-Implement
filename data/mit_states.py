"""
MIT-States dataset for Compositional Zero-Shot Learning (CZSL) evaluation.

Reference: Isola et al., "Discovering States and Transformations in Image
Collections", CVPR 2015.  The split files (train/val/test_pairs.txt) come from
the compositional-split-natural variant used by most CZSL papers.

Split assignment
----------------
Which IMAGES belong to train/val/test is decided by the official metadata file
`metadata_compositional-split-natural.t7` (Purushwalkam et al., 2019), NOT by
pair-folder membership. The pair lists deliberately include seen pairs in
val/test, so loading whole folders per pair list puts the same seen-pair
images in train AND test (the leak this version fixes). With the metadata, each
image belongs to exactly one split: 30,338 train / 10,420 val / 12,995 test.

Metadata image paths use underscores ("ancient_building/xxx.jpg") while our
image folders use spaces ("ancient building"), so paths are rebuilt from the
entry's `attr` and `obj` fields. Entries with set == "NA" or attr == "NA" are
not part of the benchmark and are skipped.

Each sample exposes five values so a CZSL evaluator can score all
(attr, obj, pair) axes independently:

    image  : (C, H, W) — preprocessed with the caller-supplied `transform`
              (CLIP's own preprocessing pipeline; see models/clip_encoder.py).
    text   : str, e.g. "ancient building"
    attr_idx : int index into self.attrs
    obj_idx  : int index into self.objs
    pair_idx : int index into self.pairs
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Callable, Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ── canonical dataset paths on Narval scratch ────────────────────────────────
_IMAGES_ROOT = Path("/scratch/tarunm10/datasets/release_dataset/images/")
_SPLITS_ROOT = Path("/scratch/tarunm10/datasets/mit-states/compositional-split-natural/")
_METADATA_PATH = Path(
    "/scratch/tarunm10/datasets/mit-states/mit-states/metadata_compositional-split-natural.t7"
)

Phase = Literal["train", "val", "test"]

# Official image counts per split — used as a sanity check at load time.
_EXPECTED_COUNTS = {"train": 30338, "val": 10420, "test": 12995}


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
    MIT-States CZSL dataset (official per-image split).

    Args:
        phase        : "train", "val", or "test"
        transform    : image preprocessing callable, applied to a PIL RGB image
                       and returning a (C, H, W) tensor.  Pass the CLIP model's
                       own `preprocess` so image statistics match what the
                       frozen visual tower was trained on.
        images_root  : override the default Narval image path
        splits_root  : override the default Narval pair-list path
        metadata_path: override the default path to the official
                       metadata_compositional-split-natural.t7 file
    """

    def __init__(
        self,
        transform: Callable,
        phase: Phase = "train",
        images_root: Path | str = _IMAGES_ROOT,
        splits_root: Path | str = _SPLITS_ROOT,
        metadata_path: Path | str = _METADATA_PATH,
    ) -> None:
        self.phase = phase
        self.images_root = Path(images_root)
        self.splits_root = Path(splits_root)
        self.metadata_path = Path(metadata_path)
        self.transform = transform

        # ── pairs for this split (used to validate metadata entries) ──────
        split_pairs = set(_load_pairs(self.splits_root / f"{phase}_pairs.txt"))

        # ── build global vocab from *all three* splits ────────────────────
        # Indices must be consistent across train/val/test so that a classifier
        # trained on train-set indices can be evaluated on val/test indices.
        all_pairs: list[tuple[str, str]] = []
        for name in ("train_pairs.txt", "val_pairs.txt", "test_pairs.txt"):
            all_pairs.extend(_load_pairs(self.splits_root / name))

        self.attrs: list[str] = sorted(set(a for a, _ in all_pairs))
        self.objs: list[str] = sorted(set(o for _, o in all_pairs))
        self.pairs: list[tuple[str, str]] = sorted(set(all_pairs))

        self._attr2idx = {a: i for i, a in enumerate(self.attrs)}
        self._obj2idx = {o: i for i, o in enumerate(self.objs)}
        self._pair2idx = {p: i for i, p in enumerate(self.pairs)}

        # ── build sample list from the official per-image split ───────────
        metadata = torch.load(self.metadata_path, weights_only=False)

        self.samples: list[tuple[Path, str, str]] = []
        folder_cache: dict[str, set[str]] = {}   # list each folder once (fast on Lustre)
        n_missing = 0
        n_bad_pair = 0

        for entry in metadata:
            if entry["set"] != phase or entry["attr"] == "NA":
                continue
            attr, obj = entry["attr"], entry["obj"]
            if (attr, obj) not in split_pairs:
                n_bad_pair += 1
                continue

            folder = f"{attr} {obj}"
            if folder not in folder_cache:
                folder_path = self.images_root / folder
                folder_cache[folder] = (
                    set(os.listdir(folder_path)) if folder_path.is_dir() else set()
                )

            filename = Path(entry["image"]).name
            if filename not in folder_cache[folder]:
                n_missing += 1
                continue

            self.samples.append((self.images_root / folder / filename, attr, obj))

        # ── sanity checks ─────────────────────────────────────────────────
        if n_missing:
            warnings.warn(
                f"MITStates ({phase}): {n_missing} metadata images not found on disk",
                stacklevel=2,
            )
        if n_bad_pair:
            warnings.warn(
                f"MITStates ({phase}): {n_bad_pair} metadata entries have a pair "
                f"not listed in {phase}_pairs.txt (skipped)",
                stacklevel=2,
            )
        expected = _EXPECTED_COUNTS.get(phase)
        if expected is not None and len(self.samples) != expected:
            warnings.warn(
                f"MITStates ({phase}): loaded {len(self.samples)} images, "
                f"expected {expected} for the official split",
                stacklevel=2,
            )

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, attr, obj = self.samples[idx]

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
    # Split check: official counts and zero train/test overlap.
    tr = MITStates(transform=None, phase="train")
    va = MITStates(transform=None, phase="val")
    te = MITStates(transform=None, phase="test")
    print(f"train: {len(tr)}  val: {len(va)}  test: {len(te)}")

    tr_paths = {p for p, _, _ in tr.samples}
    overlap_te = tr_paths & {p for p, _, _ in te.samples}
    overlap_va = tr_paths & {p for p, _, _ in va.samples}
    print(f"train∩test: {len(overlap_te)}  train∩val: {len(overlap_va)}")
    assert not overlap_te and not overlap_va, "images shared across splits"
    print(f"attrs: {tr.num_attrs}  objs: {tr.num_objs}  pairs: {tr.num_pairs}")

    # Shape check with CLIP preprocessing.
    from models.clip_encoder import CLIPEncoder

    encoder = CLIPEncoder.load_pretrained()
    ds = MITStates(transform=encoder.preprocess, phase="train")
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    images, texts, attr_idxs, obj_idxs, pair_idxs = next(iter(loader))
    B, C, H, W = images.shape
    assert C == 3 and H == 224 and W == 224, f"Unexpected image shape: {tuple(images.shape)}"
    print(f"first batch: {list(texts)}  images {tuple(images.shape)}")
    print("\nAll checks passed.")