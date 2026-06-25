"""
Main training loop for VL-JEPA on MIT-States CZSL.

Three-head routing
------------------
Each optimiser step trains exactly ONE primitive head against its own batch
type, routed by `RoutingDataLoader` (see data/primitive_sampler.py):

    attr-batch ─► attr_head(visual)        vs  Y("red")        ─┐
    obj-batch  ─► obj_head(visual)         vs  Y("chair")       ├─ InfoNCE
    comp-batch ─► comp_head(attr_e, obj_e) vs  Y("red chair")  ─┘  (per head)

Pipeline per batch
------------------
clips (B, F, C, H, W)  ──►  VisualEncoder  ──►  patch tokens (B, F, P, 1024)
texts List[str]        ──►  YEncoder       ──►  target embeds (B, 1536)
patch tokens           ──►  one head       ──►  pred embeds   (B, 1536)
pred + target          ──►  InfoNCELoss    ──►  scalar loss (backprop one head)
"""

import argparse
import logging
import sys
from pathlib import Path
import os

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.mit_states import MITStates as MITStatesDataset
from data.primitive_sampler import RoutingDataLoader
from models.loss import InfoNCELoss
from models.primitive_heads import PrimitiveHeads
from models.visual_encoder import VisualEncoder
from models.y_encoder import YEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VL-JEPA on MIT-States CZSL")
    parser.add_argument("--epochs",     type=int,   default=10,
                        help="number of full passes through the training set")
    parser.add_argument("--batch_size", type=int,   default=32,
                        help="samples per gradient step")
    parser.add_argument("--lr",         type=float, default=5e-5,
                        help="base learning rate for AdamW")
    parser.add_argument("--num_frames", type=int,   default=2,
                        help="frames per clip (MIT-States repeats the image)")
    parser.add_argument("--seed",       type=int,   default=0,
                        help="base RNG seed for the primitive batch sampler")
    # ── Primitive-head architecture (attr/obj transformer heads) ──
    parser.add_argument("--head_hidden", type=int, default=512,
                        help="transformer hidden dim per attr/obj head (384–512)")
    parser.add_argument("--head_layers", type=int, default=4,
                        help="transformer encoder layers per attr/obj head (4–6)")
    parser.add_argument("--head_heads",  type=int, default=8,
                        help="attention heads per layer (6–8; must divide head_hidden)")
    parser.add_argument("--head_pool",   choices=["mean", "token"], default="mean",
                        help="patch pooling: mean-pool or learnable aggregation token")
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
    visual_encoder: VisualEncoder,
    y_encoder: YEncoder,
    primitive_heads: PrimitiveHeads,
    loss_fn: InfoNCELoss,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Save all model state dicts, optimizer state, and training position."""
    CKPT_DIR.mkdir(exist_ok=True)

    state = {
        "step":            step,
        "epoch":           epoch,
        "visual_encoder":  visual_encoder.state_dict(),
        "y_encoder":       y_encoder.state_dict(),
        "primitive_heads": primitive_heads.state_dict(),
        "head_config":     primitive_heads.config,
        "loss_fn":         loss_fn.state_dict(),
        "optimizer":       optimizer.state_dict(),
    }

    ckpt_path = CKPT_DIR / f"step_{step:07d}.pt"
    torch.save(state, ckpt_path)
    logger.info("Checkpoint saved → %s", ckpt_path)

    tmp = CKPT_DIR / "latest.pt.tmp"
    torch.save(state, tmp)
    tmp.rename(CKPT_DIR / "latest.pt")


def load_checkpoint(
    visual_encoder: VisualEncoder,
    y_encoder: YEncoder,
    primitive_heads: PrimitiveHeads,
    loss_fn: InfoNCELoss,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    latest = CKPT_DIR / "latest.pt"
    if not latest.exists():
        logger.info("No checkpoint found — starting from scratch")
        return 0, 0

    logger.info("Loading checkpoint from %s", latest)
    state = torch.load(latest, map_location="cpu")

    visual_encoder.load_state_dict(state["visual_encoder"])
    y_encoder.load_state_dict(state["y_encoder"])
    primitive_heads.load_state_dict(state["primitive_heads"])
    loss_fn.load_state_dict(state["loss_fn"])
    optimizer.load_state_dict(state["optimizer"])

    logger.info("Resumed from step=%d  epoch=%d", state["step"], state["epoch"])
    return state["step"], state["epoch"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Datasets and DataLoaders ───────────────────────────────────────────────
    train_dataset = MITStatesDataset(
        phase="train",
        num_frames=args.num_frames,
        images_root=args.data_root,
        splits_root=args.split_root,
    )
    val_dataset = MITStatesDataset(
        phase="val",
        num_frames=args.num_frames,
        images_root=args.data_root,
        splits_root=args.split_root,
    )
    logger.info(
        "Dataset: %d train / %d val | %d attrs  %d objs  %d pairs",
        len(train_dataset), len(val_dataset),
        train_dataset.num_attrs, train_dataset.num_objs, train_dataset.num_pairs,
    )

    # Each optimiser step trains one head against a batch of one primitive
    # type; RoutingDataLoader interleaves the three streams round-robin and
    # guarantees distinct primitive keys (hence distinct targets) per batch.
    train_loader = RoutingDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        seed=args.seed,
    )
    val_loader = RoutingDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        seed=args.seed,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    logger.info("Loading VisualEncoder …")
    visual_encoder = VisualEncoder.load_pretrained(is_frozen=True).to(device)
    visual_encoder.eval()

    logger.info("Loading YEncoder …")
    y_encoder = YEncoder.load_pretrained(device=device).to(device)

    logger.info("Building PrimitiveHeads …")
    primitive_heads = PrimitiveHeads.build(
        hidden_dim=args.head_hidden,
        num_layers=args.head_layers,
        num_heads=args.head_heads,
        pool=args.head_pool,
        device=device,
    )

    loss_fn = InfoNCELoss().to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        primitive_heads.param_groups(base_lr=args.lr)
        + y_encoder.param_groups(base_lr=args.lr)
        + [{"params": loss_fn.parameters(), "lr": args.lr}],
        weight_decay=0.05,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step, start_epoch = load_checkpoint(
        visual_encoder, y_encoder, primitive_heads, loss_fn, optimizer
    )

    # ── Forward routing helper ──────────────────────────────────────────────
    # Returns (pred_embeds, target_embeds) for the head selected by batch_type.
    # `texts` already carries the mode-appropriate strings (attribute-only,
    # object-only, or the full phrase), so the Y-encoder targets are correct
    # for each head without any further splitting here.
    def forward_batch(batch_type, patch_tokens, texts):
        target_embeds = y_encoder(texts)                       # (B, 1536)
        if batch_type == "attr":
            pred = primitive_heads.forward_attribute(patch_tokens)
        elif batch_type == "obj":
            pred = primitive_heads.forward_object(patch_tokens)
        else:  # "comp" — attr/obj heads run detached inside forward_composition
            pred = primitive_heads.forward_composition(patch_tokens)
        return pred, target_embeds

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        logger.info("── Epoch %d / %d ──", epoch + 1, args.epochs)
        primitive_heads.train()
        y_encoder.train()
        train_loader.set_epoch(epoch)

        for clips, texts, batch_type in train_loader:
            clips = clips.to(device, non_blocking=True)

            # Step 1 — Visual encoding (frozen, no grad)
            with torch.no_grad():
                patch_tokens = visual_encoder(clips)   # (B, F, num_patches, 1024)

            # Step 2 — Route to the head for this batch type + encode targets
            pred_embeds, target_embeds = forward_batch(batch_type, patch_tokens, texts)

            # Step 3 — Bidirectional InfoNCE loss, computed per head
            loss = loss_fn(pred_embeds, target_embeds)

            # Step 4 — Backprop (updates only the routed head + Y-encoder head)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 10 == 0:
                logger.info(
                    "epoch=%d  step=%d  type=%-4s  loss=%.4f  tau=%.4f",
                    epoch + 1, global_step, batch_type,
                    loss.item(), loss_fn.tau.item(),
                )

            if global_step % 1000 == 0:
                save_checkpoint(
                    step=global_step,
                    epoch=epoch,
                    visual_encoder=visual_encoder,
                    y_encoder=y_encoder,
                    primitive_heads=primitive_heads,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                )

        save_checkpoint(
            step=global_step,
            epoch=epoch + 1,
            visual_encoder=visual_encoder,
            y_encoder=y_encoder,
            primitive_heads=primitive_heads,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

        # ── Validation ────────────────────────────────────────────────────────
        # Track loss per batch type so each head's progress is visible.
        primitive_heads.eval()
        y_encoder.eval()
        val_loss_sum = {"attr": 0.0, "obj": 0.0, "comp": 0.0}
        val_steps    = {"attr": 0,   "obj": 0,   "comp": 0}
        val_loader.set_epoch(epoch)

        with torch.no_grad():
            for clips, texts, batch_type in val_loader:
                clips        = clips.to(device, non_blocking=True)
                patch_tokens = visual_encoder(clips)
                pred_embeds, target_embeds = forward_batch(batch_type, patch_tokens, texts)
                val_loss_sum[batch_type] += loss_fn(pred_embeds, target_embeds).item()
                val_steps[batch_type]    += 1

        per_type = {
            t: val_loss_sum[t] / max(val_steps[t], 1) for t in val_loss_sum
        }
        val_loss = sum(per_type.values()) / len(per_type)
        logger.info(
            "── epoch %d complete  val_loss=%.4f  (attr=%.4f obj=%.4f comp=%.4f) ──",
            epoch + 1, val_loss, per_type["attr"], per_type["obj"], per_type["comp"],
        )

    logger.info("Training complete — %d steps  %d epochs", global_step, args.epochs)


if __name__ == "__main__":
    main()