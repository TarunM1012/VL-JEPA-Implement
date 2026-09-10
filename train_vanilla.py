"""
Vanilla CLIP + CSP-style soft-prompt baseline for MIT-States CZSL.

Ablation goal
-------------
Isolates the contribution of primitive batching + per-head InfoNCE (the core
v3-CLIP architecture in train.py) by removing both and keeping everything
else identical:

    v3-CLIP r1 (train.py)                vanilla (this script)
    ------------------------------       ------------------------------
    3 learned primitive heads            0 learned heads (frozen CLIP proj.)
    PrimitiveBatchSampler (routing)      standard shuffled DataLoader
    3 separate per-head InfoNCE losses   1 InfoNCE loss (image vs full comp.)
    plain "attr obj" text targets        CSP-style learnable soft prompt

Pipeline per batch
-------------------
images (B, C, H, W)   ──► CLIPEncoder.get_visual_features (frozen)      ──► patch tokens (B, P, 1024)
patch tokens          ──► FrozenProjectionHead (frozen CLIP ln_post+proj) ──► image_embed (B, 768)
texts List[str]       ──► VanillaSoftPromptEncoder (learnable ctx)       ──► text_embed  (B, 768)
image_embed + text_embed ──► InfoNCELoss ──► scalar loss (backprop reaches ctx + tau only)

Checkpoint / eval_vanilla.py compatibility
-------------------------------------------
Saves `primitive_heads` state (trivial: only the frozen ln_post/proj
buffers, zero trainable params) with `head_config={"variant": "vanilla"}`,
plus the learned soft prompt under `soft_prompt` / `soft_prompt_config`.
Evaluated with the standalone `eval_vanilla.py` (NOT evaluate.py, which is
wired to the per-head SoftPromptTextEncoder / v3-CLIP r2 checkpoint format
and is intentionally left untouched) — see scripts/eval_vanilla_narval.sh.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.mit_states import MITStates as MITStatesDataset
from models.clip_encoder import CLIPEncoder
from models.loss import InfoNCELoss
from models.primitive_heads import PrimitiveHeads
from models.soft_prompt_vanilla import VanillaSoftPromptEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_vanilla")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train vanilla CLIP + CSP-style soft-prompt baseline on MIT-States CZSL"
    )
    parser.add_argument("--epochs",     type=int,   default=10,
                        help="number of full passes through the training set")
    parser.add_argument("--batch_size", type=int,   default=32,
                        help="samples per gradient step")
    parser.add_argument("--lr",         type=float, default=2e-3,
                        help="soft-prompt learning rate (CoOp-style prompt tuning uses a "
                             "much higher LR than full fine-tuning — only a few thousand "
                             "context-vector params are trained)")
    parser.add_argument("--seed",       type=int,   default=0,
                        help="RNG seed for the standard shuffled DataLoader")
    parser.add_argument("--n_ctx",      type=int,   default=8,
                        help="number of learnable context tokens prepended to the "
                             "composition text (CSP-style soft prompt)")
    parser.add_argument(
        "--data_root",
        default="/scratch/tarunm10/datasets/release_dataset/images",
        help="root containing the 'attr obj' image subdirectories",
    )
    parser.add_argument(
        "--split_root",
        default="/scratch/tarunm10/datasets/mit-states/compositional-split-natural",
        help="directory containing train/val/test_pairs.txt split files",
    )
    return parser.parse_args()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

CKPT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))


def save_checkpoint(
    step: int,
    epoch: int,
    primitive_heads: PrimitiveHeads,
    soft_prompt: VanillaSoftPromptEncoder,
    loss_fn: InfoNCELoss,
    optimizer: torch.optim.Optimizer,
    n_ctx: int,
) -> None:
    """
    Save trainable state (soft prompt + temperature), the frozen head
    buffers (tiny — needed so evaluate.py's PrimitiveHeads.build() +
    load_state_dict() round-trips), and training position. CLIP itself is
    never checkpointed — CLIPEncoder.load_pretrained() deterministically
    reproduces the same frozen backbone at eval/resume time.
    """
    CKPT_DIR.mkdir(exist_ok=True)

    state = {
        "step":               step,
        "epoch":              epoch,
        "primitive_heads":    primitive_heads.state_dict(),
        "head_config":        primitive_heads.config,
        "soft_prompt":        soft_prompt.state_dict(),
        "soft_prompt_config": {"n_ctx": n_ctx},
        "loss_fn":            loss_fn.state_dict(),
        "optimizer":          optimizer.state_dict(),
    }

    ckpt_path = CKPT_DIR / f"step_{step:07d}.pt"
    torch.save(state, ckpt_path)
    logger.info("Checkpoint saved → %s", ckpt_path)

    tmp = CKPT_DIR / "latest.pt.tmp"
    torch.save(state, tmp)
    tmp.rename(CKPT_DIR / "latest.pt")


def load_checkpoint(
    primitive_heads: PrimitiveHeads,
    soft_prompt: VanillaSoftPromptEncoder,
    loss_fn: InfoNCELoss,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    latest = CKPT_DIR / "latest.pt"
    if not latest.exists():
        logger.info("No checkpoint found — starting from scratch")
        return 0, 0

    logger.info("Loading checkpoint from %s", latest)
    state = torch.load(latest, map_location="cpu")

    primitive_heads.load_state_dict(state["primitive_heads"])
    soft_prompt.load_state_dict(state["soft_prompt"])
    loss_fn.load_state_dict(state["loss_fn"])
    optimizer.load_state_dict(state["optimizer"])

    logger.info("Resumed from step=%d  epoch=%d", state["step"], state["epoch"])
    return state["step"], state["epoch"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Models ────────────────────────────────────────────────────────────────
    logger.info("Loading CLIPEncoder …")
    clip_encoder = CLIPEncoder.load_pretrained(device=device)
    clip_encoder.eval()
    # Unlike train.py, this script backprops THROUGH the frozen text
    # transformer into the soft prompt's context vectors. fp32 keeps that
    # gradient path numerically stable without needing loss scaling / AMP —
    # CLIP is fully frozen either way, so this only costs a bit of speed.
    clip_encoder.clip_model.float()

    # ── Datasets and DataLoaders ───────────────────────────────────────────────
    train_dataset = MITStatesDataset(
        transform=clip_encoder.preprocess,
        phase="train",
        images_root=args.data_root,
        splits_root=args.split_root,
    )
    val_dataset = MITStatesDataset(
        transform=clip_encoder.preprocess,
        phase="val",
        images_root=args.data_root,
        splits_root=args.split_root,
    )
    logger.info(
        "Dataset: %d train / %d val | %d attrs  %d objs  %d pairs",
        len(train_dataset), len(val_dataset),
        train_dataset.num_attrs, train_dataset.num_objs, train_dataset.num_pairs,
    )

    # Standard random batching — no PrimitiveBatchSampler. Each sample's text
    # is already the full "attr obj" phrase (MITStates.__getitem__), which is
    # exactly what this baseline's single loss trains against.
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator,
    )
    # Val order must stay fixed across epochs so val_loss is comparable
    # epoch-to-epoch (same gotcha noted in CONTEXT.md for train.py).
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info("Building PrimitiveHeads (variant=vanilla, 0 trainable params) …")
    primitive_heads = PrimitiveHeads.build(variant="vanilla", device=device)
    primitive_heads.load_frozen_projection(clip_encoder)
    primitive_heads.eval()

    logger.info("Building VanillaSoftPromptEncoder (n_ctx=%d) …", args.n_ctx)
    soft_prompt = VanillaSoftPromptEncoder(clip_encoder.clip_model, n_ctx=args.n_ctx).to(device)

    loss_fn = InfoNCELoss().to(device)

    # Only the soft prompt's context vectors and the InfoNCE temperature are
    # trainable — CLIP and the (parameter-free) vanilla head shim are frozen.
    optimizer = torch.optim.AdamW(
        [
            {"params": soft_prompt.parameters(), "lr": args.lr, "weight_decay": 0.0},
            {"params": loss_fn.parameters(),     "lr": args.lr, "weight_decay": 0.0},
        ]
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step, start_epoch = load_checkpoint(primitive_heads, soft_prompt, loss_fn, optimizer)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        logger.info("── Epoch %d / %d ──", epoch + 1, args.epochs)
        soft_prompt.train()

        for images, texts, _attr_idxs, _obj_idxs, _pair_idxs in train_loader:
            images = images.to(device, non_blocking=True)

            # Step 1 — Frozen visual path: patch tokens → CLIP's own
            # projected image embedding (no learned head, no grad).
            with torch.no_grad():
                patch_tokens = clip_encoder.get_visual_features(images)     # (B, P, 1024)
                image_embed  = primitive_heads.forward_composition(patch_tokens)  # (B, 768)

            # Step 2 — Soft-prompted text embedding (differentiable w.r.t. ctx).
            text_embed = soft_prompt(texts)                                 # (B, 768)

            # Step 3 — Single InfoNCE loss, image vs full composition text.
            loss = loss_fn(image_embed, text_embed)

            # Step 4 — Backprop (updates only ctx + tau).
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 10 == 0:
                logger.info(
                    "epoch=%d  step=%d  loss=%.4f  tau=%.4f",
                    epoch + 1, global_step, loss.item(), loss_fn.tau.item(),
                )

            if global_step % 1000 == 0:
                save_checkpoint(
                    step=global_step, epoch=epoch,
                    primitive_heads=primitive_heads, soft_prompt=soft_prompt,
                    loss_fn=loss_fn, optimizer=optimizer, n_ctx=args.n_ctx,
                )

        save_checkpoint(
            step=global_step, epoch=epoch + 1,
            primitive_heads=primitive_heads, soft_prompt=soft_prompt,
            loss_fn=loss_fn, optimizer=optimizer, n_ctx=args.n_ctx,
        )

        # ── Validation ────────────────────────────────────────────────────────
        soft_prompt.eval()
        val_loss_sum, val_steps = 0.0, 0

        with torch.no_grad():
            for images, texts, _attr_idxs, _obj_idxs, _pair_idxs in val_loader:
                images       = images.to(device, non_blocking=True)
                patch_tokens = clip_encoder.get_visual_features(images)
                image_embed  = primitive_heads.forward_composition(patch_tokens)
                text_embed   = soft_prompt(texts)
                val_loss_sum += loss_fn(image_embed, text_embed).item()
                val_steps    += 1

        val_loss = val_loss_sum / max(val_steps, 1)
        logger.info("── epoch %d complete  val_loss=%.4f ──", epoch + 1, val_loss)

    logger.info("Training complete — %d steps  %d epochs", global_step, args.epochs)


if __name__ == "__main__":
    main()
