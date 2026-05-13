"""
VL-JEPA training loop on MIT-States CZSL.

Pipeline per batch
------------------
clips  (B, F, C, H, W)  ──►  VisualEncoder  ──►  patch tokens  (B, F, P, 1024)
texts  List[str]         ──►  YEncoder       ──►  target embeds  (B, 1536)
patch tokens + texts     ──►  Predictor      ──►  pred embeds    (B, 1536)
pred + target            ──►  InfoNCELoss    ──►  scalar loss
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.mit_states import MITStates as MITStatesDataset
from models.loss import InfoNCELoss
from models.predictor import Predictor
from models.visual_encoder import VisualEncoder
from models.y_encoder import YEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VL-JEPA on MIT-States CZSL")

    parser.add_argument("--epochs",     type=int,   default=10,
                        help="Number of full passes through the training set")
    parser.add_argument("--batch_size", type=int,   default=64,
                        help="Samples per gradient step")
    parser.add_argument("--lr",         type=float, default=5e-5,
                        help="Base learning rate for AdamW")
    parser.add_argument("--num_frames", type=int,   default=2,
                        help="Frames per clip (MIT-States repeats the image)")
    parser.add_argument(
        "--data_root",
        default="/scratch/tarunm10/datasets/release_dataset",
        help="Root directory containing the MIT-States image folders",
    )
    parser.add_argument(
        "--split_root",
        default="/scratch/tarunm10/datasets/mit-states/compositional-split-natural",
        help="Directory containing train/val/test_pairs.txt split files",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CKPT_DIR = Path("checkpoints")


def save_checkpoint(
    step: int,
    epoch: int,
    visual_encoder: VisualEncoder,
    y_encoder: YEncoder,
    predictor: Predictor,
    loss_fn: InfoNCELoss,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Save model + optimizer state; also overwrite checkpoints/latest.pt."""
    CKPT_DIR.mkdir(exist_ok=True)

    state = {
        "step":            step,
        "epoch":           epoch,
        # Visual encoder is frozen — saving its state dict preserves the
        # backbone weights so the checkpoint is fully self-contained.
        "visual_encoder":  visual_encoder.state_dict(),
        "y_encoder":       y_encoder.state_dict(),
        "predictor":       predictor.state_dict(),
        # loss_fn holds the learnable temperature parameter log_inv_tau.
        "loss_fn":         loss_fn.state_dict(),
        "optimizer":       optimizer.state_dict(),
    }

    # Numbered checkpoint so we can inspect training history.
    ckpt_path = CKPT_DIR / f"step_{step:07d}.pt"
    torch.save(state, ckpt_path)
    logger.info("Saved checkpoint → %s", ckpt_path)

    # Overwrite latest.pt atomically via a rename so a crash during write
    # never leaves a corrupt latest.pt.
    latest_tmp = CKPT_DIR / "latest.pt.tmp"
    torch.save(state, latest_tmp)
    latest_tmp.rename(CKPT_DIR / "latest.pt")


def load_checkpoint(
    visual_encoder: VisualEncoder,
    y_encoder: YEncoder,
    predictor: Predictor,
    loss_fn: InfoNCELoss,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    """
    Load checkpoints/latest.pt if it exists.
    Returns (start_step, start_epoch).
    """
    latest = CKPT_DIR / "latest.pt"
    if not latest.exists():
        logger.info("No checkpoint found — starting from scratch")
        return 0, 0

    logger.info("Loading checkpoint from %s", latest)
    state = torch.load(latest, map_location="cpu")

    visual_encoder.load_state_dict(state["visual_encoder"])
    y_encoder.load_state_dict(state["y_encoder"])
    predictor.load_state_dict(state["predictor"])
    loss_fn.load_state_dict(state["loss_fn"])
    optimizer.load_state_dict(state["optimizer"])

    step  = state["step"]
    epoch = state["epoch"]
    logger.info("Resumed from step=%d  epoch=%d", step, epoch)
    return step, epoch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # -----------------------------------------------------------------------
    # Dataset and DataLoader
    # -----------------------------------------------------------------------

    # MIT-States repeats the single image num_frames times along the time axis
    # so the clip shape matches what VisualEncoder expects: (F, C, H, W).
    train_dataset = MITStatesDataset(
        phase="train",
        num_frames=args.num_frames,
        images_root=args.data_root,
        splits_root=args.split_root,
    )
    logger.info(
        "Train set: %d samples | %d attrs × %d objs | %d pairs",
        len(train_dataset),
        train_dataset.num_attrs,
        train_dataset.num_objs,
        train_dataset.num_pairs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,         # shuffle so the model never memorises batch order
        num_workers=4,
        pin_memory=True,      # pin_memory speeds up host→GPU transfer
        drop_last=True,       # InfoNCE needs a full batch to form enough negatives
    )

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------

    # VisualEncoder: frozen V-JEPA 2 (or timm fallback). No trainable params.
    logger.info("Loading VisualEncoder …")
    visual_encoder = VisualEncoder.load_pretrained(is_frozen=True).to(device)
    visual_encoder.eval()   # no BatchNorm / Dropout, but good practice

    # YEncoder: frozen EmbeddingGemma backbone + trainable projection head.
    logger.info("Loading YEncoder …")
    y_encoder = YEncoder.load_pretrained(device=device)

    # Predictor: LLaMA layers 8–15 (trainable) + frozen embed_tokens.
    logger.info("Loading Predictor …")
    predictor = Predictor.load_pretrained(device=device)

    # InfoNCE loss with learnable temperature parameter.
    loss_fn = InfoNCELoss().to(device)

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------

    # Visual encoder is fully frozen — no param groups needed for it.
    #
    # Predictor returns two groups (backbone layers + proj heads), both at base_lr.
    # YEncoder returns one group (projection head only) at base_lr × 0.05 —
    #   the reduced LR preserves the pre-trained backbone alignment.
    # The InfoNCE temperature (log_inv_tau) is trained at base_lr.
    optimizer = torch.optim.AdamW(
        predictor.param_groups(base_lr=args.lr)          # 2 groups
        + y_encoder.param_groups(base_lr=args.lr)        # 1 group
        + [{"params": loss_fn.parameters(), "lr": args.lr}],  # temperature
        weight_decay=0.05,
    )

    # -----------------------------------------------------------------------
    # Resume from checkpoint if one exists
    # -----------------------------------------------------------------------

    global_step, start_epoch = load_checkpoint(
        visual_encoder, y_encoder, predictor, loss_fn, optimizer
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    for epoch in range(start_epoch, args.epochs):
        logger.info("── Epoch %d / %d ──", epoch + 1, args.epochs)

        for batch_idx, (clips, texts, _attr_idxs, _obj_idxs, _pair_idxs) in enumerate(train_loader):
            # clips : (B, F, C, H, W) — float32 image tensors
            # texts : tuple of B strings (attribute + " " + object)
            clips = clips.to(device, non_blocking=True)   # non_blocking works with pin_memory

            # Convert texts tuple (from DataLoader collation) to a plain list
            # so the tokenisers inside YEncoder and Predictor accept it.
            texts = list(texts)

            # ── Forward: visual branch ───────────────────────────────────────
            # VisualEncoder is frozen — no gradients needed here, saving memory.
            with torch.no_grad():
                # patch_tokens: (B, F, num_patches, 1024)
                patch_tokens = visual_encoder(clips)

            # ── Forward: target branch ───────────────────────────────────────
            # YEncoder tokenises and encodes the text labels. The backbone is
            # frozen internally; only the projection head receives gradients.
            # target_embeds: (B, 1536) L2-normalised
            target_embeds = y_encoder(texts)

            # ── Forward: predictor ───────────────────────────────────────────
            # Predictor fuses visual patch tokens with the text query through
            # bidirectional LLaMA layers and outputs a 1536-dim embedding.
            # pred_embeds: (B, 1536) L2-normalised
            pred_embeds = predictor(patch_tokens, texts)

            # ── Loss ─────────────────────────────────────────────────────────
            # Bidirectional InfoNCE: pred must identify its paired target and
            # vice versa. Diagonal of (B×B) similarity matrix = positive pairs.
            loss = loss_fn(pred_embeds, target_embeds)

            # ── Backward + optimizer step ────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # ── Logging ──────────────────────────────────────────────────────
            if global_step % 10 == 0:
                logger.info(
                    "epoch=%d  step=%d  loss=%.4f  tau=%.4f",
                    epoch + 1,
                    global_step,
                    loss.item(),
                    loss_fn.tau.item(),
                )

            # ── Checkpointing ────────────────────────────────────────────────
            if global_step % 1000 == 0:
                save_checkpoint(
                    step=global_step,
                    epoch=epoch,
                    visual_encoder=visual_encoder,
                    y_encoder=y_encoder,
                    predictor=predictor,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                )

        # Save at the end of every epoch so we never lose a full epoch of work.
        save_checkpoint(
            step=global_step,
            epoch=epoch + 1,   # epoch+1 so resuming starts at the next epoch
            visual_encoder=visual_encoder,
            y_encoder=y_encoder,
            predictor=predictor,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

    logger.info("Training complete — %d steps over %d epochs", global_step, args.epochs)


if __name__ == "__main__":
    main()
