"""timm backbone loading, with an explicit logged fallback.

Three roles, three constraints:

* **local** -- CPU embedding extraction for the Phase 5 fast loop. Must be small
  enough to push the corpus through on 4 cores in tens of minutes.
* **teacher** -- best achievable accuracy, trained on Kaggle. Architecture
  unconstrained. Prefers an iNat-2021 checkpoint: it has already learned
  fine-grained bird features and needs dramatically less data than ImageNet
  weights.
* **student** -- the deployment model.

  HAILO CONSTRAINT -- DO NOT SWITCH THE STUDENT TO A TRANSFORMER.
  The Hailo-8L Dataflow Compiler supports standard CNN operations well but has
  limited and awkward support for ViT and ConvNeXt blocks (LayerNorm, GELU,
  attention). A transformer student either fails to compile or falls back to
  CPU for large subgraphs, destroying the latency budget. Keep it a plain CNN.

If a requested checkpoint is unavailable in the installed timm version we fall
back and SAY SO AT WARNING LEVEL. Failing silently here would mean quietly
training on ImageNet weights while believing we had iNat weights, which would
make every downstream data-efficiency conclusion wrong.
"""

from __future__ import annotations

import logging

from birdcam.config import Config

logger = logging.getLogger(__name__)


def load_backbone(cfg: Config, role: str = "local", num_classes: int = 0):
    """Load a timm backbone by role.

    num_classes=0 gives pooled features rather than logits -- what the embedding
    cache wants.
    """
    import timm

    spec = cfg.train_cfg["backbone"][role]
    name = spec["name"]
    fallback = spec.get("fallback_name")

    try:
        model = timm.create_model(name, pretrained=spec["pretrained"], num_classes=num_classes)
        logger.info("loaded backbone %r for role %r", name, role)
    except Exception as exc:  # noqa: BLE001
        if not fallback:
            raise
        # Loud, not silent: the fallback has different pretraining and that
        # changes what the results mean.
        logger.warning(
            "FALLBACK: backbone %r unavailable for role %r (%s). "
            "Using %r instead -- this is NOT the iNat-2021 checkpoint, and "
            "data-efficiency conclusions will differ.",
            name,
            role,
            exc,
            fallback,
        )
        model = timm.create_model(fallback, pretrained=spec["pretrained"], num_classes=num_classes)
        name = fallback

    model.eval()
    return model, name


def build_transform(cfg: Config, model, role: str = "local", training: bool = False):
    """Build the preprocessing transform matching the backbone's pretraining."""
    from timm.data import create_transform, resolve_data_config

    data_cfg = resolve_data_config({}, model=model)
    data_cfg["input_size"] = (
        3,
        cfg.train_cfg["backbone"][role]["image_size"],
        cfg.train_cfg["backbone"][role]["image_size"],
    )
    return create_transform(**data_cfg, is_training=training)


def feature_dim(model, image_size: int = 224) -> int:
    """Actual embedding width, measured by a forward pass.

    NOT `model.num_features`: that attribute disagrees with the real output on
    several architectures (MobileNetV3 reports 960 but emits 1280, because
    timm's conv head sits after the layer the attribute describes). Trusting it
    corrupts the feature matrix silently instead of raising.
    """
    import torch

    with torch.inference_mode():
        return int(model(torch.zeros(1, 3, image_size, image_size)).shape[1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    from birdcam.config import load_config

    cfg = load_config()
    for role in ("local", "student"):
        model, name = load_backbone(cfg, role)
        print(f"{role:<8} {name:<40} features={feature_dim(model)}")


if __name__ == "__main__":
    main()
