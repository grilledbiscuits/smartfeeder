"""Two-head model over a shared backbone, plus the masked partial-label loss.

Head 1 (taxon) and Head 2 (sex/plumage) are deliberately separate output layers
over shared features rather than one flattened class space. Flattening would
fragment already-scarce female data across every species; keeping them separate
lets the sex signal generalise -- a female sunbird looks female in ways that
transfer between species -- and lets the sex head act as a regulariser on the
taxon head.

The module is written to be ONNX-exportable: no data-dependent control flow, no
dynamic shapes beyond the batch dimension, and the rollup/threshold logic lives
*outside* the graph. That last point is deliberate. Rollup thresholds are tuned
per-class from precision-recall curves and change without retraining, so baking
them into the exported graph would force a recompile -- an expensive step on the
Hailo toolchain -- every time a threshold moves.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TwoHeadNet(nn.Module):
    """Backbone + taxon head + sex/plumage head.

    Returns raw logits for both heads. Softmax, rollup and thresholding are
    applied by the caller, not here -- see module docstring.
    """

    def __init__(
        self, backbone: nn.Module, feature_dim: int, n_taxon: int, n_sex: int, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.taxon_head = nn.Linear(feature_dim, n_taxon)
        self.sex_head = nn.Linear(feature_dim, n_sex)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        f = self.dropout(f)
        return self.taxon_head(f), self.sex_head(f)


def masked_partial_label_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over a SET of admissible classes.

        loss = -log( sum_{c in admissible} p_c )

    For a sample with exactly one admissible class this is identical to ordinary
    cross-entropy. For an annotated male -- admissible over {male_breeding,
    male_eclipse} -- it trains "is male" without asserting a plumage state that
    no source records.

    Implemented via logsumexp over the masked log-probabilities rather than
    log(sum(exp(...))) for numerical stability: the direct form underflows once
    the model becomes confident.

    Args:
        logits: (B, C) raw scores.
        mask:   (B, C) 1.0 where the class is admissible, 0.0 elsewhere.
    """
    if torch.any(mask.sum(dim=1) == 0):
        # An all-zero mask yields -inf loss and silently poisons the run.
        raise ValueError("every sample must have at least one admissible class")
    logp = torch.log_softmax(logits, dim=1)
    masked = logp.masked_fill(mask == 0, float("-inf"))
    return (-torch.logsumexp(masked, dim=1)).mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss for the long tail.

    Down-weights easy examples so the abundant, visually obvious classes stop
    dominating the gradient. Configurable alongside class-balanced sampling;
    Phase 7 reports which of the two actually helped.
    """
    logp = torch.log_softmax(logits, dim=1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    loss = -((1 - pt) ** gamma) * logpt
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()


def build_model(cfg, role: str = "student"):
    """Assemble a TwoHeadNet for the given backbone role."""
    from birdcam.models.backbone import feature_dim, load_backbone

    backbone, name = load_backbone(cfg, role, num_classes=0)
    dim = feature_dim(backbone, cfg.train_cfg["backbone"][role]["image_size"])
    model = TwoHeadNet(backbone, dim, len(cfg.taxon_classes), len(cfg.sex_classes))
    logger.info(
        "built two-head model on %s: features=%d, taxon=%d, sex=%d",
        name,
        dim,
        len(cfg.taxon_classes),
        len(cfg.sex_classes),
    )
    return model, name


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    from birdcam.config import load_config

    cfg = load_config()
    model, name = build_model(cfg, "student")
    n = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n / 1e6:.2f}M parameters")


if __name__ == "__main__":
    main()
