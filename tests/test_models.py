"""Tests for the two-head model and the masked partial-label loss.

The masked loss is the piece that lets Head 2 keep a `male_breeding` /
`male_eclipse` distinction that no data source provides. If it silently
degenerated into ordinary cross-entropy against one arbitrary member of the
group, the model would be learning a fabricated plumage state and nothing would
complain -- so these properties are asserted directly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from birdcam.config import load_config  # noqa: E402
from birdcam.models.heads import (  # noqa: E402
    TwoHeadNet,
    focal_loss,
    masked_partial_label_loss,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# --- masked partial-label loss ------------------------------------------------


def test_single_admissible_class_equals_cross_entropy():
    """With one admissible class the masked loss must BE cross-entropy."""
    torch.manual_seed(0)
    logits = torch.randn(8, 6)
    target = torch.randint(0, 6, (8,))
    mask = torch.zeros(8, 6)
    mask[torch.arange(8), target] = 1.0

    masked = masked_partial_label_loss(logits, mask)
    ce = torch.nn.functional.cross_entropy(logits, target)
    assert torch.allclose(masked, ce, atol=1e-6)


def test_loss_is_indifferent_between_group_members():
    """Swapping probability between group members must not change the loss.

    This is the whole point: the model is told "this is a male" and left free
    on which plumage state, because the ground truth genuinely does not say.

    Note the property is invariance *between members at equal group mass*, not
    invariance to any redistribution. Softmax normalises over all six classes,
    so a peaked [2,0,...] and a split [1,1,...] have different group totals and
    correctly give different losses -- the split leaks more mass to the
    non-admissible classes.
    """
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    a = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]])

    la = masked_partial_label_loss(a, mask)
    lb = masked_partial_label_loss(b, mask)
    assert torch.allclose(la, lb, atol=1e-6), "loss must not prefer one group member"


def test_loss_depends_only_on_summed_group_probability():
    """Two logit vectors with equal group mass must give equal loss."""
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    # Both put the same total mass on {0,1}, split differently.
    a = torch.tensor([[3.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[1.0, 3.0, 0.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(
        masked_partial_label_loss(a, mask), masked_partial_label_loss(b, mask), atol=1e-6
    )


def test_loss_falls_when_group_mass_rises():
    """Mass moving INTO the admissible group must reduce the loss."""
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    low = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]])
    high = torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert masked_partial_label_loss(high, mask) < masked_partial_label_loss(low, mask)


def test_all_zero_mask_raises_rather_than_producing_inf():
    """An all-zero mask yields -inf and would silently poison a run."""
    logits = torch.randn(3, 6)
    mask = torch.zeros(3, 6)
    with pytest.raises(ValueError, match="at least one admissible class"):
        masked_partial_label_loss(logits, mask)


def test_loss_is_finite_for_confident_predictions():
    """The logsumexp form must not underflow once the model is confident."""
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    logits = torch.tensor([[-60.0, -60.0, 60.0, 0.0, 0.0, 0.0]])
    loss = masked_partial_label_loss(logits, mask)
    assert torch.isfinite(loss), "underflowed to inf"


def test_loss_is_differentiable():
    logits = torch.randn(4, 6, requires_grad=True)
    mask = torch.zeros(4, 6)
    mask[:, 0] = 1.0
    mask[:, 1] = 1.0
    masked_partial_label_loss(logits, mask).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_gradient_does_not_discriminate_within_group():
    """Gradients on the two group members must match when their logits match."""
    logits = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    masked_partial_label_loss(logits, mask).backward()
    g = logits.grad[0]
    assert torch.allclose(g[0], g[1], atol=1e-6)


# --- focal loss ---------------------------------------------------------------


def test_focal_loss_matches_cross_entropy_at_gamma_zero():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    target = torch.randint(0, 5, (8,))
    fl = focal_loss(logits, target, gamma=0.0)
    ce = torch.nn.functional.cross_entropy(logits, target)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_focal_loss_downweights_easy_examples():
    """An easy example should contribute less under focal than under CE."""
    easy = torch.tensor([[10.0, 0.0, 0.0]])
    target = torch.tensor([0])
    fl = focal_loss(easy, target, gamma=2.0)
    ce = torch.nn.functional.cross_entropy(easy, target)
    assert fl < ce


# --- model --------------------------------------------------------------------


def test_two_head_net_output_shapes(cfg):
    backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 32))
    model = TwoHeadNet(backbone, 32, len(cfg.taxon_classes), len(cfg.sex_classes))
    taxon, sex = model(torch.randn(4, 3, 8, 8))
    assert taxon.shape == (4, len(cfg.taxon_classes))
    assert sex.shape == (4, len(cfg.sex_classes))


def test_heads_are_independent_not_flattened(cfg):
    """Two separate outputs, not one combined class space.

    Flattening would fragment scarce female data across every species.
    """
    backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 32))
    model = TwoHeadNet(backbone, 32, len(cfg.taxon_classes), len(cfg.sex_classes))
    taxon, sex = model(torch.randn(2, 3, 8, 8))
    assert taxon.shape[1] * sex.shape[1] != taxon.shape[1], "heads look flattened"
    assert model.taxon_head.out_features == len(cfg.taxon_classes)
    assert model.sex_head.out_features == len(cfg.sex_classes)
