"""
Main training loop for VL-JEPA on MIT-States CZSL.

Pipeline per batch
------------------
clips  (B, F, C, H, W)  ──►  VisualEncoder  ──►  patch tokens  (B, F, P, 1024)
texts  List[str]         ──►  YEncoder       ──►  target embeds  (B, 1536)
patch tokens + texts     ──►  Predictor      ──►  pred embeds    (B, 1536)
pred + target            ──►  InfoNCELoss    ──►  scalar loss
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Make the project root importable when the script is run directly from any
# working directory (e.g. python train.py from /scratch/...).
sys.path.insert(0, str(Path(__file__).parent))

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


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VL-JEPA on MIT-States CZSL")
    parser.add_argument("--epochs",     type=int,   default=10,
                        help="number of full passes through the training set")
    parser.add_argument("--batch_size", type=int,   default=64,
                        help="samples per gradient step")
    parser.add_argument("--lr",         type=float, default=5e-5,
                        help="base learning rate for AdamW")
    parser.add_argument("--num_frames", type=int,   default=2,
                        help="frames per clip (MIT-States repeats the image)")
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
    """Save all model state dicts, optimizer state, and training position."""
    CKPT_DIR.mkdir(exist_ok=True)

    state = {
        "step":           step,
        "epoch":          epoch,
        # VisualEncoder is frozen, but saving its weights makes the checkpoint
        # fully self-contained — no HuggingFace download needed to resume.
        "visual_encoder": visual_encoder.state_dict(),
        "y_encoder":      y_encoder.state_dict(),
        "predictor":      predictor.state_dict(),
        # loss_fn holds the learnable temperature scalar (log_inv_tau).
        "loss_fn":        loss_fn.state_dict(),
        "optimizer":      optimizer.state_dict(),
    }

    # Numbered checkpoint preserves history for analysis.
    ckpt_path = CKPT_DIR / f"step_{step:07d}.pt"
    torch.save(state, ckpt_path)
    logger.info("Checkpoint saved → %s", ckpt_path)

    # Atomic overwrite of latest.pt via rename so a crash during the write
    # never produces a corrupt file the resume logic would try to load.
    tmp = CKPT_DIR / "latest.pt.tmp"
    torch.save(state, tmp)
    tmp.rename(CKPT_DIR / "latest.pt")


def load_checkpoint(
    visual_encoder: VisualEncoder,
    y_encoder: YEncoder,
    predictor: Predictor,
    loss_fn: InfoNCELoss,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    """
    Load checkpoints/latest.pt if it exists.
    Returns (global_step, start_epoch) — both 0 when starting from scratch.
    """
    latest = CKPT_DIR / "latest.pt"
    if not latest.exists():
        logger.info("No checkpoint found — starting from scratch")
        return 0, 0

    logger.info("Loading checkpoint from %s", latest)
    # map_location='cpu' so the checkpoint loads regardless of which GPU
    # (or number of GPUs) was used when it was saved.
    state = torch.load(latest, map_location="cpu")

    visual_encoder.load_state_dict(state["visual_encoder"])
    y_encoder.load_state_dict(state["y_encoder"])
    predictor.load_state_dict(state["predictor"])
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
    # MIT-States is a still-image dataset. The class repeats each image
    # num_frames times along the temporal axis so the clip shape (F, C, H, W)
    # matches what VisualEncoder expects.
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,        # shuffle so the model never memorises batch order
        num_workers=4,
        pin_memory=True,     # speeds up host → GPU transfer
        drop_last=True,      # InfoNCE needs a full batch to form enough negatives
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # ── Models ────────────────────────────────────────────────────────────────

    # VisualEncoder: V-JEPA 2 (timm fallback if HF unavailable) — fully frozen.
    # It is a fixed feature extractor with zero trainable parameters.
    logger.info("Loading VisualEncoder …")
    visual_encoder = VisualEncoder.load_pretrained(is_frozen=True).to(device)
    visual_encoder.eval()   # remains in eval mode throughout — no param updates

    # YEncoder: EmbeddingGemma-300M backbone (frozen) + trainable projection head.
    # The backbone's forward is already wrapped in no_grad inside YEncoder.forward().
    logger.info("Loading YEncoder …")
    y_encoder = YEncoder.load_pretrained(device=device)

    # Predictor: LLaMA 3.2-1B layers 8–15 (trainable) + frozen embed_tokens.
    logger.info("Loading Predictor …")
    predictor = Predictor.load_pretrained(device=device)

    # InfoNCE loss with a single learnable temperature scalar (log_inv_tau).
    loss_fn = InfoNCELoss().to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # VisualEncoder has no trainable params — excluded entirely.
    # predictor.param_groups() → 2 groups (LLaMA layers + proj heads) at base_lr.
    # y_encoder.param_groups() → 1 group (projection head) at base_lr × 0.05.
    # The 0.05 multiplier is from the VL-JEPA paper: the head sits on top of a
    # strong frozen backbone and a full LR would destabilise early alignment.
    optimizer = torch.optim.AdamW(
        predictor.param_groups(base_lr=args.lr)              # 2 groups
        + y_encoder.param_groups(base_lr=args.lr)            # 1 group (× 0.05 inside)
        + [{"params": loss_fn.parameters(), "lr": args.lr}], # temperature scalar
        weight_decay=0.05,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step, start_epoch = load_checkpoint(
        visual_encoder, y_encoder, predictor, loss_fn, optimizer
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        logger.info("── Epoch %d / %d ──", epoch + 1, args.epochs)
        predictor.train()
        y_encoder.train()

        for clips, texts, _attr_idxs, _obj_idxs, _pair_idxs in train_loader:
            # clips: (B, num_frames, C, H, W) — float32 image tensors
            # texts: tuple[str, ...] collated by DataLoader; convert to list for tokenisers
            clips = clips.to(device, non_blocking=True)
            texts = list(texts)

            # Step 1 — Encode video clips into visual patch token sequences.
            # VisualEncoder is frozen: skip the computation graph entirely to
            # save GPU memory and avoid building buffers we will never use.
            with torch.no_grad():
                patch_tokens = visual_encoder(clips)   # (B, F, num_patches, 1024)

            # Step 2 — Encode text labels into the shared embedding space.
            # Gradients flow only through YEncoder's trainable projection head;
            # the frozen backbone forward is no_grad inside YEncoder.
            target_embeds = y_encoder(texts)           # (B, 1536), L2-normalised

            # Step 3 — Predict target embeddings by fusing visual patch tokens
            # with the text query through bidirectional LLaMA layers.
            pred_embeds = predictor(patch_tokens, texts)   # (B, 1536), L2-normalised

            # Step 4 — Bidirectional InfoNCE loss.
            # Diagonal of the (B × B) cosine-similarity matrix = positive pairs.
            # Every off-diagonal entry is a negative, so a larger batch size
            # provides harder in-batch negatives and a stronger training signal.
            loss = loss_fn(pred_embeds, target_embeds)

            # Step 5 — Backpropagate and update all trainable parameters.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # Log loss and current temperature every 10 steps.
            if global_step % 10 == 0:
                logger.info(
                    "epoch=%d  step=%d  loss=%.4f  tau=%.4f",
                    epoch + 1, global_step,
                    loss.item(), loss_fn.tau.item(),
                )

            # Save checkpoint every 1000 steps so training can resume mid-epoch.
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

        # Save at end of epoch — epoch+1 so resuming starts at the next epoch.
        save_checkpoint(
            step=global_step,
            epoch=epoch + 1,
            visual_encoder=visual_encoder,
            y_encoder=y_encoder,
            predictor=predictor,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

        # ── Validation ────────────────────────────────────────────────────────
        # Report val loss after each epoch to track generalisation.
        predictor.eval()
        y_encoder.eval()
        val_loss_sum, val_steps = 0.0, 0

        with torch.no_grad():
            for clips, texts, _attr_idxs, _obj_idxs, _pair_idxs in val_loader:
                clips         = clips.to(device, non_blocking=True)
                texts         = list(texts)
                patch_tokens  = visual_encoder(clips)
                target_embeds = y_encoder(texts)
                pred_embeds   = predictor(patch_tokens, texts)
                val_loss_sum += loss_fn(pred_embeds, target_embeds).item()
                val_steps    += 1

        val_loss = val_loss_sum / max(val_steps, 1)
        logger.info("── epoch %d complete  val_loss=%.4f ──", epoch + 1, val_loss)

    logger.info("Training complete — %d steps  %d epochs", global_step, args.epochs)


if __name__ == "__main__":
    main()
