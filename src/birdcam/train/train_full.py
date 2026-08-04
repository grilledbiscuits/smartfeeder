"""Phase 6: end-to-end fine-tune. Runs on a CPU laptop or a Kaggle GPU, unmodified.

Everything up to now has trained a linear head on frozen ImageNet features. That
sets a ceiling the head cannot break: per ASSUMPTIONS.md A20, only ~23% of
genuine bird visits clear a confidence threshold, and *Cinnyris chalybeus* --
the most likely visitor at Rondebosch -- is the worst class. Thresholds, priors
and calibration all operate downstream of the features. This is the module that
changes the features.

## Making it finish on a laptop

A full fine-tune of every parameter over 18k images on four CPU cores is a
multi-day job. `freeze_blocks` is the lever that makes it tractable: the early
blocks of an ImageNet backbone encode edges and textures that transfer to birds
unchanged, so freezing them removes most of the backward pass while leaving the
layers that actually specialise.

    freeze_blocks: 0  -> full fine-tune, slowest, highest ceiling
    freeze_blocks: 4  -> train the last 2 blocks + heads (default)
    freeze_blocks: 6  -> heads only, equivalent to the frozen-feature loop

Run `--estimate` first. It measures real throughput on this machine and prints
an honest ETA before you commit hours to it.

## Resumability

Checkpoints are written every epoch and `--resume` picks up mid-run. Kaggle caps
sessions at 12 hours and a laptop gets closed; neither should cost the run.

## Memory

Batch size is deliberately small with gradient accumulation to reach a useful
effective batch. On a machine with ~2-3GB free, a large batch is the difference
between a training run and an OOM kill -- which this project has already
survived four of.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from birdcam.config import Config, load_config

logger = logging.getLogger(__name__)


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_taxon_acc: float
    val_tier_a_recall: float
    val_ece: float
    seconds: float
    lr: float


@dataclass
class RunState:
    """Everything needed to resume. Serialised alongside the weights."""

    epoch: int = 0
    best_val: float = -1.0
    history: list[dict] = field(default_factory=list)


# --- data ---------------------------------------------------------------------


def _build_loaders(cfg: Config, image_size: int, batch_size: int, limit: int | None = None):
    import torch
    from PIL import Image

    from birdcam.data.dataset import load_labelled
    from birdcam.data.manifest import open_manifest
    from birdcam.train.augment import build_eval_transform, build_train_transform

    with open_manifest(cfg.path("manifest_db")) as m:
        items = load_labelled(cfg, m)
    if limit:
        # Deterministic subsample across all classes, for smoke runs.
        rng = np.random.RandomState(0)
        idx = rng.choice(len(items), min(limit, len(items)), replace=False)
        items = [items[i] for i in sorted(idx)]

    train_items = [i for i in items if i.split == "train"]
    val_items = [i for i in items if i.split == "val"]
    if not train_items or not val_items:
        raise RuntimeError("empty train or val split; run birdcam.data.preprocess")

    class _DS(torch.utils.data.Dataset):
        def __init__(self, rows, transform):
            self.rows, self.transform = rows, transform

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            it = self.rows[i]
            with Image.open(it.path) as im:
                x = self.transform(im.convert("RGB"))
            return x, it.taxon_index, torch.from_numpy(it.sex_mask)

    tr_tf = build_train_transform(cfg, image_size)
    ev_tf = build_eval_transform(image_size)

    sampler = None
    shuffle = True
    if cfg.train_cfg["train"]["loss"]["use_class_balanced_sampling"]:
        # Class-balanced ("effective number of samples") weighting. Counteracts
        # a long tail where Cinnyris chalybeus has 20x the images of a rare
        # Tier C species.
        beta = cfg.train_cfg["train"]["loss"]["class_balanced_beta"]
        counts = np.bincount(
            [i.taxon_index for i in train_items], minlength=len(cfg.taxon_classes)
        ).astype(np.float64)
        eff = np.where(counts > 0, (1.0 - np.power(beta, counts)) / (1.0 - beta), 1.0)
        per_class_w = np.where(counts > 0, 1.0 / eff, 0.0)
        weights = [per_class_w[i.taxon_index] for i in train_items]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(train_items), replacement=True
        )
        shuffle = False

    nw = cfg.train_cfg["compute"]["dataloader_num_workers"]
    train_loader = torch.utils.data.DataLoader(
        _DS(train_items, tr_tf), batch_size=batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=nw, drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        _DS(val_items, ev_tf), batch_size=batch_size, shuffle=False,
        num_workers=nw, persistent_workers=nw > 0,
    )
    return train_loader, val_loader, len(train_items), len(val_items)


# --- model --------------------------------------------------------------------


def build_and_freeze(cfg: Config, freeze_blocks: int):
    """Build the two-head model and freeze the first `freeze_blocks` stages.

    Early convolutional stages encode edges and textures that transfer to birds
    unchanged. Freezing them removes most of the backward pass -- which is what
    makes this finish on a CPU -- while leaving the later, more semantic stages
    free to specialise on sunbird plumage.
    """
    from birdcam.models.heads import build_model

    model, name = build_model(cfg, "student")
    backbone = model.backbone

    frozen = trainable = 0
    if freeze_blocks > 0:
        for mod_name in ("conv_stem", "bn1"):
            if hasattr(backbone, mod_name):
                for p in getattr(backbone, mod_name).parameters():
                    p.requires_grad = False
        if hasattr(backbone, "blocks"):
            for i, block in enumerate(backbone.blocks):
                if i < freeze_blocks:
                    for p in block.parameters():
                        p.requires_grad = False

    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    logger.info(
        "%s: %.2fM trainable, %.2fM frozen (freeze_blocks=%d)",
        name, trainable / 1e6, frozen / 1e6, freeze_blocks,
    )
    return model, name, trainable


# --- loss ---------------------------------------------------------------------


def _taxon_loss(cfg: Config, logits, target, class_weights=None):
    import torch.nn.functional as F

    lc = cfg.train_cfg["train"]["loss"]
    if lc["use_focal"]:
        from birdcam.models.heads import focal_loss

        return focal_loss(logits, target, gamma=lc["focal_gamma"], weight=class_weights)
    return F.cross_entropy(
        logits, target, weight=class_weights,
        label_smoothing=cfg.train_cfg["train"]["label_smoothing"],
    )


# --- evaluation ---------------------------------------------------------------


def evaluate(cfg: Config, model, loader, device):
    import torch

    from birdcam.eval.metrics import expected_calibration_error

    model.eval()
    preds, targets, probs = [], [], []
    with torch.inference_mode():
        for x, y, _ in loader:
            out, _ = model(x.to(device))
            p = torch.softmax(out.float(), dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(1))
            targets.append(y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    probs = np.concatenate(probs)

    acc = float((preds == targets).mean())
    tier_a = {
        cfg.taxon_class_index[s.slug]
        for s in cfg.species_by_tier("A")
        if s.slug in cfg.taxon_class_index
    }
    recalls = [
        float((preds[targets == c] == c).mean())
        for c in sorted(tier_a)
        if (targets == c).sum() > 0
    ]
    ece, _ = expected_calibration_error(probs, targets)
    return acc, float(np.mean(recalls)) if recalls else 0.0, float(ece)


# --- checkpointing ------------------------------------------------------------


def _ckpt_path(cfg: Config) -> Path:
    d = cfg.path("checkpoints_dir")
    d.mkdir(parents=True, exist_ok=True)
    return d / "student_last.pt"


def save_checkpoint(cfg, model, optimiser, scheduler, state: RunState, best: bool = False):
    import torch

    payload = {
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict(),
        "state": asdict(state),
    }
    torch.save(payload, _ckpt_path(cfg))
    if best:
        torch.save(payload, _ckpt_path(cfg).with_name("student_best.pt"))


def load_checkpoint(cfg, model, optimiser, scheduler) -> RunState:
    import torch

    path = _ckpt_path(cfg)
    if not path.is_file():
        return RunState()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimiser.load_state_dict(payload["optimiser"])
    scheduler.load_state_dict(payload["scheduler"])
    st = RunState(**payload["state"])
    logger.info("resumed from epoch %d", st.epoch)
    return st


# --- tracking -----------------------------------------------------------------


def _make_tracker(cfg: Config, run_name: str):
    """Weights & Biases when keyed, else TensorBoard, else stdout only."""
    import os

    tc = cfg.train_cfg["train"]["tracking"]
    if os.environ.get(tc["wandb_api_key_env_var"]):
        try:
            import wandb

            wandb.init(project=tc["wandb_project"], name=run_name,
                       config=cfg.train_cfg["train"])
            logger.info("logging to Weights & Biases")
            return lambda d, step: wandb.log(d, step=step)
        except ImportError:
            logger.warning("WANDB_API_KEY set but wandb not installed")
    try:
        from torch.utils.tensorboard import SummaryWriter

        w = SummaryWriter(str(cfg.root / tc["tensorboard_dir"] / run_name))
        logger.info("logging to TensorBoard: %s", tc["tensorboard_dir"])
        return lambda d, step: [w.add_scalar(k, v, step) for k, v in d.items()]
    except ImportError:
        logger.info("no tracker available (install tensorboard); stdout only")
        return lambda d, step: None


# --- throughput estimate ------------------------------------------------------


def estimate(cfg: Config, freeze_blocks: int, batch_size: int, steps: int = 12) -> None:
    """Measure real throughput and print an honest ETA before committing hours."""
    import torch

    from birdcam.utils.runtime import setup_torch

    setup_torch(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = cfg.train_cfg["backbone"]["student"]["image_size"]
    model, _, trainable = build_and_freeze(cfg, freeze_blocks)
    model.to(device).train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    loader, _, n_train, _ = _build_loaders(cfg, size, batch_size)
    from birdcam.models.heads import masked_partial_label_loss

    it = iter(loader)
    t0 = None
    done = 0
    for i in range(steps):
        try:
            x, y, mask = next(it)
        except StopIteration:
            break
        if i == 2:  # skip warm-up steps
            t0 = time.monotonic()
            done = 0
        opt.zero_grad()
        tl, sl = model(x.to(device))
        loss = _taxon_loss(cfg, tl, y.to(device))
        loss = loss + cfg.train_cfg["train"]["loss"]["sex_weight"] * masked_partial_label_loss(
            sl, mask.to(device)
        )
        loss.backward()
        opt.step()
        if t0 is not None:
            done += len(x)
    if t0 is None or done == 0:
        print("not enough batches to estimate")
        return
    rate = done / (time.monotonic() - t0)
    per_epoch = n_train / rate
    epochs = cfg.train_cfg["train"]["epochs"]
    print(f"\ndevice           : {device}")
    print(f"freeze_blocks    : {freeze_blocks}  ({trainable / 1e6:.2f}M trainable params)")
    print(f"batch size       : {batch_size}")
    print(f"throughput       : {rate:.1f} images/sec")
    print(f"train images     : {n_train}")
    print(f"time per epoch   : {per_epoch / 60:.1f} min")
    print(f"{epochs} epochs        : {per_epoch * epochs / 3600:.1f} hours")
    print("\nCheckpoints are written every epoch and --resume continues, so this")
    print("does not need to run in one sitting.")


# --- training -----------------------------------------------------------------


def train(
    cfg: Config,
    freeze_blocks: int = 4,
    batch_size: int | None = None,
    epochs: int | None = None,
    accum: int = 2,
    resume: bool = False,
    limit: int | None = None,
) -> list[EpochStats]:
    import torch

    from birdcam.models.heads import masked_partial_label_loss
    from birdcam.utils.runtime import setup_torch

    setup_torch(cfg)
    tc = cfg.train_cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = cfg.train_cfg["backbone"]["student"]["image_size"]
    batch_size = batch_size or (32 if device.type == "cuda" else 12)
    epochs = epochs or tc["epochs"]

    train_loader, val_loader, n_train, n_val = _build_loaders(cfg, size, batch_size, limit)
    model, name, _ = build_and_freeze(cfg, freeze_blocks)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps_per_epoch = max(1, len(train_loader) // accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tc["lr"], total_steps=epochs * steps_per_epoch,
        pct_start=tc["warmup_epochs"] / max(epochs, 1),
    )

    state = load_checkpoint(cfg, model, opt, sched) if resume else RunState()
    track = _make_tracker(cfg, f"{name}_fb{freeze_blocks}")
    logger.info(
        "training %d epochs on %s: %d train / %d val, batch %d x %d accumulation",
        epochs, device, n_train, n_val, batch_size, accum,
    )

    out: list[EpochStats] = []
    for epoch in range(state.epoch, epochs):
        model.train()
        t0 = time.monotonic()
        running, seen = 0.0, 0
        opt.zero_grad()
        for step, (x, y, mask) in enumerate(train_loader):
            tl, sl = model(x.to(device))
            loss = _taxon_loss(cfg, tl, y.to(device))
            loss = loss + tc["loss"]["sex_weight"] * masked_partial_label_loss(
                sl, mask.to(device)
            )
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
            running += loss.detach().item() * len(x)
            seen += len(x)
            if seen and step % 50 == 0:
                logger.info(
                    "  epoch %d step %d/%d loss %.4f",
                    epoch, step, len(train_loader), running / seen,
                )

        acc, tier_a, ece = evaluate(cfg, model, val_loader, device)
        st = EpochStats(
            epoch=epoch, train_loss=running / max(seen, 1), val_taxon_acc=acc,
            val_tier_a_recall=tier_a, val_ece=ece,
            seconds=time.monotonic() - t0, lr=float(sched.get_last_lr()[0]),
        )
        out.append(st)
        state.epoch = epoch + 1
        state.history.append(asdict(st))
        is_best = tier_a > state.best_val
        if is_best:
            state.best_val = tier_a
        save_checkpoint(cfg, model, opt, sched, state, best=is_best)
        track(
            {"train/loss": st.train_loss, "val/taxon_acc": acc,
             "val/tier_a_recall": tier_a, "val/ece": ece, "lr": st.lr},
            epoch,
        )
        logger.info(
            "epoch %d: loss %.4f  val acc %.4f  tierA recall %.4f  ECE %.4f  (%.1f min)%s",
            epoch, st.train_loss, acc, tier_a, ece, st.seconds / 60,
            "  <- best" if is_best else "",
        )

    hist = cfg.path("reports_dir") / "training_history.json"
    hist.parent.mkdir(parents=True, exist_ok=True)
    hist.write_text(json.dumps(state.history, indent=2), encoding="utf-8")
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    import argparse

    ap = argparse.ArgumentParser(description="Phase 6: end-to-end fine-tune.")
    ap.add_argument("--freeze-blocks", type=int, default=4,
                    help="0=full fine-tune, 4=last two blocks (default), 6=heads only")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="subsample, for smoke runs")
    ap.add_argument("--estimate", action="store_true",
                    help="measure throughput and print an ETA, then exit")
    args = ap.parse_args()

    cfg = load_config()
    if args.estimate:
        estimate(cfg, args.freeze_blocks, args.batch_size or 12)
        return
    stats = train(
        cfg, freeze_blocks=args.freeze_blocks, batch_size=args.batch_size,
        epochs=args.epochs, accum=args.accum, resume=args.resume, limit=args.limit,
    )
    if stats:
        best = max(stats, key=lambda s: s.val_tier_a_recall)
        print(f"\nbest epoch {best.epoch}: val taxon {best.val_taxon_acc:.4f}, "
              f"Tier A recall {best.val_tier_a_recall:.4f}, ECE {best.val_ece:.4f}")


if __name__ == "__main__":
    main()
