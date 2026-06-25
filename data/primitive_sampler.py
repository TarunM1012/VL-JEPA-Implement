"""
Per-primitive batch routing for VL-JEPA training on MIT-States.

Why
---
The three predictor heads (`models/primitive_heads.py`) each specialise in one
CZSL primitive and are trained on disjoint batch streams:

    attr-batch  ─► AttributeHead   vs  Y(attribute word alone, e.g. "red")
    obj-batch   ─► ObjectHead      vs  Y(object word alone,    e.g. "chair")
    comp-batch  ─► CompositionHead vs  Y(full phrase,          e.g. "red chair")

This module produces those three batch streams from a single `MITStates`
dataset and interleaves them so each optimiser step trains exactly one head
(the loss is computed independently per head, never summed across heads).

Batching strategy
-----------------
`PrimitiveBatchSampler` guarantees that **every sample in a batch has a
distinct primitive key** for that batch's mode:

    mode="attr"  → distinct attributes  → distinct Y("attr") targets
    mode="obj"   → distinct objects     → distinct Y("obj")  targets
    mode="comp"  → distinct (attr,obj)  → distinct Y("attr obj") targets

Distinct keys mean distinct InfoNCE targets, which removes false negatives
(two different samples that share the same target would otherwise be treated
as a negative pair against each other).  Object-invariance for the attribute
head emerges because many (attribute, different-object) images map to the
*same* attribute-only target across batches, so the head must ignore the
object to minimise the loss — and symmetrically for the object head.

Each batch is sampled as: pick `batch_size` distinct keys at random, then one
random sample index per key.  Sampling is with replacement across batches (an
"epoch" is a fixed step budget, as is standard for contrastive training).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Literal, Tuple

import torch
from torch.utils.data import DataLoader

from data.mit_states import MITStates

logger = logging.getLogger(__name__)

Mode = Literal["attr", "obj", "comp"]
_MODES: Tuple[Mode, ...] = ("attr", "obj", "comp")


def _split_attr_obj(text: str) -> Tuple[str, str]:
    """'ancient building' -> ('ancient', 'building').

    Mirrors the convention used in train.py / evaluate.py: the attribute is
    the first whitespace token, the object is everything after it.
    """
    parts = text.split()
    return parts[0], " ".join(parts[1:])


# ----------------------------------------------------------------------
# Batch sampler
# ----------------------------------------------------------------------

class PrimitiveBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Yields lists of dataset indices, each list a batch with distinct primitive
    keys for the given `mode`.

    Args
    ----
        dataset    : a MITStates instance.
        mode       : "attr" | "obj" | "comp" — which primitive defines a batch.
        batch_size : samples per batch (== distinct keys per batch).
        num_batches: how many batches to yield per epoch.  Defaults to
                     len(dataset) // batch_size so an epoch sees roughly the
                     whole dataset once.
        seed       : base RNG seed; the epoch counter is mixed in so successive
                     epochs differ while remaining reproducible.
    """

    def __init__(
        self,
        dataset: MITStates,
        mode: Mode,
        batch_size: int,
        num_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0

        # Group dataset indices by the primitive key for this mode.
        # dataset.samples is a list of (img_path, attr, obj).
        groups: Dict[object, List[int]] = defaultdict(list)
        for idx, (_path, attr, obj) in enumerate(dataset.samples):
            if mode == "attr":
                key: object = attr
            elif mode == "obj":
                key = obj
            else:  # "comp"
                key = (attr, obj)
            groups[key].append(idx)

        self._keys: List[object] = list(groups.keys())
        self._groups = groups

        if batch_size > len(self._keys):
            raise ValueError(
                f"batch_size ({batch_size}) exceeds the number of distinct "
                f"{mode} keys ({len(self._keys)}); cannot form a batch with "
                f"distinct keys"
            )

        self.num_batches = (
            num_batches if num_batches is not None
            else len(dataset) // batch_size
        )

        logger.info(
            "PrimitiveBatchSampler[%s]: %d distinct keys, %d batches/epoch, "
            "batch_size=%d",
            mode, len(self._keys), self.num_batches, batch_size,
        )

    def set_epoch(self, epoch: int) -> None:
        """Reseed for a new epoch so batches differ run-to-run yet stay reproducible."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch * len(_MODES) + _MODES.index(self.mode))
        for _ in range(self.num_batches):
            chosen_keys = rng.sample(self._keys, self.batch_size)   # distinct
            batch = [rng.choice(self._groups[k]) for k in chosen_keys]
            yield batch


# ----------------------------------------------------------------------
# Collate (mode-aware text extraction)
# ----------------------------------------------------------------------

def _make_collate(mode: Mode):
    """
    Build a collate_fn that stacks clips and emits the text appropriate to
    `mode`.  Each raw sample is the 5-tuple from MITStates.__getitem__:
    (clip, text, attr_idx, obj_idx, pair_idx).
    """

    def collate(samples: List[tuple]):
        clips = torch.stack([s[0] for s in samples], dim=0)   # (B, F, C, H, W)
        texts: List[str] = []
        for s in samples:
            full = s[1]
            attr, obj = _split_attr_obj(full)
            if mode == "attr":
                texts.append(attr)
            elif mode == "obj":
                texts.append(obj)
            else:  # "comp"
                texts.append(full)
        return clips, texts

    return collate


# ----------------------------------------------------------------------
# Routing dataloader
# ----------------------------------------------------------------------

class RoutingDataLoader:
    """
    Interleaves three per-mode DataLoaders round-robin.

    Iterating yields `(clips, texts, batch_type)` where `batch_type` is one of
    "attr" | "obj" | "comp", `clips` is (B, F, C, H, W), and `texts` is the
    list of mode-appropriate target strings (already reduced to attribute-only,
    object-only, or the full phrase).

    One round-robin pass per "round" emits one batch of each type, so over an
    epoch each head receives an equal number of batches.
    """

    def __init__(
        self,
        dataset: MITStates,
        batch_size: int,
        num_batches_per_mode: int | None = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self._samplers: Dict[Mode, PrimitiveBatchSampler] = {}
        self._loaders: Dict[Mode, DataLoader] = {}

        for mode in _MODES:
            sampler = PrimitiveBatchSampler(
                dataset, mode=mode, batch_size=batch_size,
                num_batches=num_batches_per_mode, seed=seed,
            )
            self._samplers[mode] = sampler
            self._loaders[mode] = DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=_make_collate(mode),
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        # All samplers share the same num_batches (derived identically).
        self.num_batches_per_mode = self._samplers["attr"].num_batches

    def set_epoch(self, epoch: int) -> None:
        for sampler in self._samplers.values():
            sampler.set_epoch(epoch)

    def __len__(self) -> int:
        """Total batches per epoch across all three modes."""
        return self.num_batches_per_mode * len(_MODES)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, List[str], Mode]]:
        iters = {mode: iter(loader) for mode, loader in self._loaders.items()}
        for _ in range(self.num_batches_per_mode):
            for mode in _MODES:
                clips, texts = next(iters[mode])
                yield clips, texts, mode


# ----------------------------------------------------------------------
# Smoke test  —  python -m data.primitive_sampler --smoke
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run routing/distinctness checks on a synthetic dataset.")
    args = parser.parse_args()

    if not args.smoke:
        parser.print_help()
        raise SystemExit(0)

    # Build a tiny in-memory stand-in for MITStates so the smoke test needs no
    # dataset on disk.  It mimics the two attributes the sampler relies on:
    #   .samples : list of (path, attr, obj)
    #   __getitem__ : (clip, "attr obj", attr_idx, obj_idx, pair_idx)
    class _FakeMITStates:
        def __init__(self):
            attrs = ["red", "blue", "old", "new", "wet", "dry"]
            objs = ["chair", "table", "car", "cup", "book", "shoe"]
            self.samples = []
            for a in attrs:
                for o in objs:
                    for _ in range(3):   # 3 images per pair
                        self.samples.append((f"{a}_{o}.jpg", a, o))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            _path, attr, obj = self.samples[idx]
            clip = torch.zeros(2, 3, 8, 8)   # tiny fake clip
            return clip, f"{attr} {obj}", 0, 0, 0

    ds = _FakeMITStates()
    bs = 4

    # ---- Check 1: distinct keys per batch, per mode ----------------------
    for mode in _MODES:
        sampler = PrimitiveBatchSampler(ds, mode=mode, batch_size=bs, num_batches=10)
        for batch in sampler:
            assert len(batch) == bs
            if mode == "attr":
                keys = [ds.samples[i][1] for i in batch]
            elif mode == "obj":
                keys = [ds.samples[i][2] for i in batch]
            else:
                keys = [(ds.samples[i][1], ds.samples[i][2]) for i in batch]
            assert len(set(keys)) == bs, f"{mode}: duplicate keys in a batch: {keys}"
        print(f"[check 1] mode={mode}: all batches have {bs} distinct keys ✓")

    # ---- Check 2: round-robin routing emits each type equally ------------
    loader = RoutingDataLoader(ds, batch_size=bs, num_batches_per_mode=5,
                               num_workers=0, pin_memory=False)
    seen = {m: 0 for m in _MODES}
    order = []
    for clips, texts, batch_type in loader:
        seen[batch_type] += 1
        order.append(batch_type)
        assert clips.shape[0] == bs
        assert len(texts) == bs
    assert seen == {"attr": 5, "obj": 5, "comp": 5}, seen
    assert order[:3] == ["attr", "obj", "comp"], order[:3]
    print(f"[check 2] routing emitted {seen}, round-robin order ✓")

    # ---- Check 3: mode-appropriate texts ---------------------------------
    loader = RoutingDataLoader(ds, batch_size=bs, num_batches_per_mode=1,
                               num_workers=0, pin_memory=False)
    for clips, texts, batch_type in loader:
        if batch_type == "attr":
            assert all(" " not in t for t in texts), texts
        elif batch_type == "comp":
            assert all(" " in t for t in texts), texts
    print("[check 3] texts match batch mode (attr/obj single word, comp phrase) ✓")

    print("\nAll checks passed.")
