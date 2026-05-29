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
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    parser.add_argument("--batch_size", type=int,   default=32,
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

CKPT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))


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
        "visual_encoder": visual_encoder.state_dict(),
        "y_encoder":      y_encoder.state_dict(),
        "predictor":      predictor.state_dict(),
        "loss_fn":        loss_fn.state_dict(),
        "optimizer":      optimizer.state_dict(),
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
    predictor: Predictor,
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
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
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
    logger.info("Loading VisualEncoder …")
    visual_encoder = VisualEncoder.load_pretrained(is_frozen=True).to(device)
    visual_encoder.eval()

    logger.info("Loading YEncoder …")
    y_encoder = YEncoder.load_pretrained(device=device).to(device)

    logger.info("Loading Predictor …")
    predictor = Predictor.load_pretrained(device=device).to(device)

    loss_fn = InfoNCELoss().to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        predictor.param_groups(base_lr=args.lr)
        + y_encoder.param_groups(base_lr=args.lr)
        + [{"params": loss_fn.parameters(), "lr": args.lr}],
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
            clips = clips.to(device, non_blocking=True)
            texts = list(texts)

            # Step 1 — Visual encoding (frozen, no grad)
            with torch.no_grad():
                patch_tokens = visual_encoder(clips)   # (B, F, num_patches, 1024)

            # Step 2 — Encode attr and obj separately, sum for compositionality
            attrs = [t.split()[0] for t in texts]
            objs  = [" ".join(t.split()[1:]) for t in texts]
            attr_embeds   = y_encoder(attrs)                              # (B, 1536)
            obj_embeds    = y_encoder(objs)                               # (B, 1536)
            target_embeds = F.normalize(attr_embeds + obj_embeds, dim=-1) # (B, 1536)

            # Step 3 — Predict target embeddings from visual tokens
            queries     = ["a photo of"] * len(texts)
            pred_embeds = predictor(patch_tokens, queries)   # (B, 1536)

            # Step 4 — Bidirectional InfoNCE loss
            #loss = loss_fn(pred_embeds, target_embeds)

            #new loss
            # ── Auxiliary compositional loss ──────────────────────────────────────
            # For pairs in the batch that share the same attribute, pull their
            # attr embeddings together. Same for shared objects.
            # This forces the model to learn transferable primitive representations.
            aux_loss = torch.tensor(0.0, device=device)
            num_aux = 0

            for i in range(len(attrs)):
                for j in range(i + 1, len(attrs)):
                    if attrs[i] == attrs[j]:
                        # same attribute — embeddings should be similar
                        aux_loss += 1 - F.cosine_similarity(
                            attr_embeds[i].unsqueeze(0),
                            attr_embeds[j].unsqueeze(0)
                        ).item()
                        num_aux += 1
                    if objs[i] == objs[j]:
                        # same object — embeddings should be similar
                        aux_loss += 1 - F.cosine_similarity(
                            obj_embeds[i].unsqueeze(0),
                            obj_embeds[j].unsqueeze(0)
                        ).item()
                        num_aux += 1

            if num_aux > 0:
                aux_loss = aux_loss / num_aux

            # Combine losses — 0.1 weight keeps aux from dominating
            loss = loss_fn(pred_embeds, target_embeds) + 0.1 * aux_loss


            # Step 5 — Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 10 == 0:
                logger.info(
                    "epoch=%d  step=%d  loss=%.4f  aux=%.4f  tau=%.4f",
                    epoch + 1, global_step,
                    loss.item(), aux_loss.item(), loss_fn.tau.item(),
                )

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
        predictor.eval()
        y_encoder.eval()
        val_loss_sum, val_steps = 0.0, 0

        with torch.no_grad():
            for clips, texts, _attr_idxs, _obj_idxs, _pair_idxs in val_loader:
                clips        = clips.to(device, non_blocking=True)
                texts        = list(texts)
                patch_tokens = visual_encoder(clips)
                attrs        = [t.split()[0] for t in texts]
                objs         = [" ".join(t.split()[1:]) for t in texts]
                attr_embeds  = y_encoder(attrs)
                obj_embeds   = y_encoder(objs)
                target_embeds = F.normalize(attr_embeds + obj_embeds, dim=-1)
                queries      = ["a photo of"] * len(texts)
                pred_embeds  = predictor(patch_tokens, queries)
                val_loss_sum += loss_fn(pred_embeds, target_embeds).item()
                val_steps    += 1

        val_loss = val_loss_sum / max(val_steps, 1)
        logger.info("── epoch %d complete  val_loss=%.4f ──", epoch + 1, val_loss)

    logger.info("Training complete — %d steps  %d epochs", global_step, args.epochs)


if __name__ == "__main__":
    main()