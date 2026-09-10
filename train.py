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
images (B, C, H, W)    ──►  CLIPEncoder.get_visual_features
                        ──►  patch tokens (B, P, 1024)
texts List[str]        ──►  SoftPromptTextEncoder(head=batch_type)  ──►  target embeds (B, 768)
patch tokens           ──►  one head                                ──►  pred embeds   (B, 768)
pred + target          ──►  InfoNCELoss                             ──►  scalar loss (backprop one head)

v3-CLIP r2: text targets are encoded through per-head CSP-style learnable
soft prompts (see models/soft_prompt_encoder.py) instead of raw words fed
straight to frozen CLIP. The soft-prompt context vectors are the only new
trainable parameters; everything else is identical to v3-CLIP r1.
"""

import argparse
import logging
import sys
from pathlib import Path
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from data.mit_states import MITStates as MITStatesDataset
from data.primitive_sampler import RoutingDataLoader
from models.clip_encoder import CLIPEncoder
from models.loss import InfoNCELoss
from models.primitive_heads import PrimitiveHeads
from models.soft_prompt_encoder import SoftPromptTextEncoder

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
    # ── CSP-style soft prompts (text-encoder targets) ──
    parser.add_argument("--prompt_n_ctx", type=int, default=8,
                        help="number of learnable soft-prompt context tokens per head")
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
    soft_prompts: SoftPromptTextEncoder,
    loss_fn: InfoNCELoss,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Save trainable state dicts, optimizer state, and training position.

    CLIP is entirely frozen with no trainable parameters, so its weights are
    not checkpointed — CLIPEncoder.load_pretrained() deterministically
    reproduces the same backbone at eval/resume time, and skipping it avoids
    duplicating a ~900 MB frozen model into every checkpoint file. The
    soft-prompt context vectors ARE checkpointed: they are the only
    trainable parameters inside SoftPromptTextEncoder (its wrapped CLIP
    model is frozen and excluded the same way).
    """
    CKPT_DIR.mkdir(exist_ok=True)

    state = {
        "step":            step,
        "epoch":           epoch,
        "primitive_heads": primitive_heads.state_dict(),
        "head_config":     primitive_heads.config,
        "soft_prompts":    {
            "attr_ctx": soft_prompts.attr_ctx.detach().cpu(),
            "obj_ctx":  soft_prompts.obj_ctx.detach().cpu(),
            "comp_ctx": soft_prompts.comp_ctx.detach().cpu(),
        },
        "prompt_n_ctx":    soft_prompts.n_ctx,
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
    primitive_heads: PrimitiveHeads,
    soft_prompts: SoftPromptTextEncoder,
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
    if "soft_prompts" in state:
        device = soft_prompts.attr_ctx.device
        with torch.no_grad():
            soft_prompts.attr_ctx.copy_(state["soft_prompts"]["attr_ctx"].to(device))
            soft_prompts.obj_ctx.copy_(state["soft_prompts"]["obj_ctx"].to(device))
            soft_prompts.comp_ctx.copy_(state["soft_prompts"]["comp_ctx"].to(device))
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

    # ── Datasets and DataLoaders ───────────────────────────────────────────────
    # Images are preprocessed with CLIP's own transform (encoder.preprocess)
    # so pixel statistics match what the frozen visual tower expects.
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

    logger.info("Building PrimitiveHeads …")
    primitive_heads = PrimitiveHeads.build(
        hidden_dim=args.head_hidden,
        num_layers=args.head_layers,
        num_heads=args.head_heads,
        pool=args.head_pool,
        device=device,
    )

    logger.info("Building SoftPromptTextEncoder …")
    soft_prompts = SoftPromptTextEncoder(
        clip_encoder.clip_model, n_ctx=args.prompt_n_ctx
    ).to(device)

    loss_fn = InfoNCELoss().to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # CLIP is fully frozen (no trainable projection head, unlike the old
    # Y-encoder), so its parameters contribute no param group here. The
    # soft-prompt context vectors are continuous embeddings, not weight
    # matrices, so they are excluded from weight decay for the same reason
    # LayerNorm/bias params and the temperature τ are (see
    # models/primitive_heads.py::_split_decay_params).
    optimizer = torch.optim.AdamW(
        primitive_heads.param_groups(base_lr=args.lr)
        + [{"params": loss_fn.parameters(), "lr": args.lr, "weight_decay": 0.0}]
        + [{"params": soft_prompts.parameters(), "lr": args.lr, "weight_decay": 0.0}],
        weight_decay=0.05,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step, start_epoch = load_checkpoint(
        primitive_heads, soft_prompts, loss_fn, optimizer
    )

    # ── Forward routing helper ──────────────────────────────────────────────
    # Returns (pred_embeds, target_embeds) for the head selected by batch_type.
    # `texts` already carries the mode-appropriate strings (attribute-only,
    # object-only, or the full phrase); SoftPromptTextEncoder prepends that
    # head's learnable context vectors before running CLIP's frozen text
    # tower, so the targets are correct for each head without any further
    # splitting here.
    def forward_batch(batch_type, patch_tokens, texts):
        target_embeds = F.normalize(soft_prompts(texts, head=batch_type), dim=-1)  # (B, 768)
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
        train_loader.set_epoch(epoch)

        for images, texts, batch_type in train_loader:
            images = images.to(device, non_blocking=True)

            # Step 1 — Visual encoding (frozen, no grad)
            with torch.no_grad():
                patch_tokens = clip_encoder.get_visual_features(images)   # (B, num_patches, 1024)

            # Step 2 — Route to the head for this batch type + encode targets
            pred_embeds, target_embeds = forward_batch(batch_type, patch_tokens, texts)

            # Step 3 — Bidirectional InfoNCE loss, computed per head
            loss = loss_fn(pred_embeds, target_embeds)

            # v2r2r3: upweight attr head loss (1.5:1.0:1.0, attr:obj:comp)
            if batch_type == "attr":
                loss = loss * 1.5

            # Step 4 — Backprop (updates only the routed head)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 10 == 0:
                logger.info(
                    "epoch=%d  step=%d  type=%-4s  weighted_loss=%.4f  tau=%.4f",
                    epoch + 1, global_step, batch_type,
                    loss.item(), loss_fn.tau.item(),
                )

            if global_step % 1000 == 0:
                save_checkpoint(
                    step=global_step,
                    epoch=epoch,
                    primitive_heads=primitive_heads,
                    soft_prompts=soft_prompts,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                )

        save_checkpoint(
            step=global_step,
            epoch=epoch + 1,
            primitive_heads=primitive_heads,
            soft_prompts=soft_prompts,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

        # ── Validation ────────────────────────────────────────────────────────
        # Track loss per batch type so each head's progress is visible.
        primitive_heads.eval()
        val_loss_sum = {"attr": 0.0, "obj": 0.0, "comp": 0.0}
        val_steps    = {"attr": 0,   "obj": 0,   "comp": 0}
        val_loader.set_epoch(0)  # fixed arrangement — val loss stays comparable across epochs

        with torch.no_grad():
            for images, texts, batch_type in val_loader:
                images       = images.to(device, non_blocking=True)
                patch_tokens = clip_encoder.get_visual_features(images)
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