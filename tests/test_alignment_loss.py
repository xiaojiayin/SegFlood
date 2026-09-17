from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.lightning_module import MultiModalSegmentationModule


def _alignment_module(
    *,
    mode: str = "exclude_changed",
    num_samples: int = 64,
    radius: int = 1,
) -> MultiModalSegmentationModule:
    """Create the minimal module state needed by alignment unit tests."""
    module = MultiModalSegmentationModule.__new__(MultiModalSegmentationModule)
    nn.Module.__init__(module)
    module.alignment_mode = mode
    module.alignment_num_samples = num_samples
    module.alignment_patch_radius = radius
    module.alignment_temperature = 0.1
    module.alignment_layers = [-1]
    module.alignment_position_weight = 0.25
    module.alignment_stop_gradient = True
    module.alignment_semantic_stop_gradient = False
    module.alignment_warmup_epochs = 0
    module.alignment_decay_start_epoch = -1
    module.alignment_decay_end_epoch = -1
    module.fusion = SimpleNamespace(
        strategy=SimpleNamespace(
            project_opt=nn.ModuleList([nn.Identity()]),
            project_sar=nn.ModuleList([nn.Identity()]),
        )
    )
    return module


def test_exclude_changed_uses_only_background_tokens() -> None:
    module = _alignment_module()
    x = torch.randn(1, 4, 2, 3, requires_grad=True)
    y = torch.randn(1, 4, 2, 3, requires_grad=True)
    unchanged = torch.tensor([[[True, False, True], [False, False, True]]])

    loss, stats = module._patch_siglip_loss(
        x, y, radius=1, valid_mask=unchanged
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert stats["align_valid_tokens"].item() == 3
    changed = ~unchanged
    assert torch.count_nonzero(x.grad.permute(0, 2, 3, 1)[changed]) == 0
    assert torch.count_nonzero(y.grad.permute(0, 2, 3, 1)[changed]) == 0


@pytest.mark.parametrize(
    ("valid_mask", "num_samples"),
    [
        (torch.zeros(2, 3, 3, dtype=torch.bool), 64),
        (torch.ones(2, 3, 3, dtype=torch.bool), 0),
    ],
    ids=["all-changed", "no-valid-token"],
)
def test_exclude_changed_empty_selection_is_finite(
    valid_mask: torch.Tensor,
    num_samples: int,
) -> None:
    module = _alignment_module(num_samples=num_samples)
    x = torch.randn(2, 4, 3, 3, requires_grad=True)
    y = torch.randn(2, 4, 3, 3, requires_grad=True)

    loss, stats = module._patch_siglip_loss(
        x, y, radius=1, valid_mask=valid_mask
    )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    assert stats["align_valid_tokens"].item() == 0
    assert torch.count_nonzero(x.grad) == 0
    assert torch.count_nonzero(y.grad) == 0


def test_exclude_changed_resizes_batch_target_per_feature_level() -> None:
    module = _alignment_module()
    x = torch.randn(1, 4, 2, 2, requires_grad=True)
    y = torch.randn(1, 4, 2, 2, requires_grad=True)
    # Nearest downsampling to 2x2 leaves only the top-left token unchanged.
    target = torch.ones(1, 4, 4, dtype=torch.long)
    target[:, :2, :2] = 0

    loss, stats = module._compute_alignment_loss([x], [y], target=target)

    assert torch.isfinite(loss)
    assert stats["align_valid_tokens_l-1"].item() == 1


def test_original_and_proximity_modes_select_expected_radius(monkeypatch) -> None:
    module = _alignment_module(mode="original", radius=3)
    x = torch.randn(1, 4, 2, 2)
    seen_radii = []

    def fake_loss(x, y, radius=1, valid_mask=None, semantic_labels=None):
        seen_radii.append(radius)
        return x.sum() * 0.0, {}

    monkeypatch.setattr(module, "_patch_siglip_loss", fake_loss)
    module._compute_alignment_loss([x], [x])
    module.alignment_mode = "proximity"
    module._compute_alignment_loss([x], [x])

    assert seen_radii == [0, 3]


def test_legacy_module_without_mode_keeps_proximity_behavior(monkeypatch) -> None:
    module = _alignment_module(radius=2)
    del module.alignment_mode
    seen_radii = []

    def fake_loss(x, y, radius=1, valid_mask=None, semantic_labels=None):
        seen_radii.append(radius)
        return x.sum() * 0.0, {}

    monkeypatch.setattr(module, "_patch_siglip_loss", fake_loss)
    x = torch.randn(1, 4, 2, 2)
    module._compute_alignment_loss([x], [x])

    assert seen_radii == [2]


def test_disabled_mode_turns_off_legacy_enable_switch(monkeypatch) -> None:
    encoder = SimpleNamespace(
        feature_channels=[4],
        feature_reductions=[1],
        optical_feature_channels=None,
        sar_feature_channels=None,
    )
    fusion = SimpleNamespace(output_channels=[4])

    def fake_component(self, component, name):
        return {
            "encoder": encoder,
            "fusion": fusion,
            "decoder": nn.Identity(),
            "loss_fn": nn.Identity(),
        }[name]

    monkeypatch.setattr(
        MultiModalSegmentationModule, "_instantiate_component", fake_component
    )
    monkeypatch.setattr(
        MultiModalSegmentationModule, "_init_metrics", lambda self, num_classes: None
    )
    module = MultiModalSegmentationModule(
        encoder={"_target_": "unused"},
        fusion={"_target_": "unused"},
        decoder={"_target_": "unused"},
        loss_fn={"_target_": "unused"},
        alignment_enabled=True,
        alignment_mode="disabled",
    )

    assert module.alignment_enabled is False


def test_soft_alignment_cap_preserves_gradient_when_saturated() -> None:
    module = _alignment_module()
    module.alignment_cap_mode = "soft"
    module.align_cap_ratio = 0.3
    module.register_buffer("align_ema_main", torch.tensor(1.0), persistent=False)
    scaled_loss = torch.tensor(0.6, requires_grad=True)

    capped, cap = module._apply_alignment_cap(scaled_loss)
    capped.backward()

    assert cap.item() == pytest.approx(0.3)
    assert capped.item() <= cap.item()
    assert scaled_loss.grad is not None
    assert scaled_loss.grad.item() > 0.0


def test_alignment_schedule_warms_up_then_decays_to_zero(
    monkeypatch,
) -> None:
    module = _alignment_module()
    module.alignment_target_weight = 0.02
    module.alignment_warmup_epochs = 5
    module.alignment_decay_start_epoch = 15
    module.alignment_decay_end_epoch = 40

    epoch = [0]
    monkeypatch.setattr(
        MultiModalSegmentationModule,
        "current_epoch",
        property(lambda self: epoch[0]),
    )
    assert module._get_alignment_schedule() == pytest.approx(0.004)
    epoch[0] = 10
    assert module._get_alignment_schedule() == pytest.approx(0.02)
    epoch[0] = 15
    assert module._get_alignment_schedule() == pytest.approx(0.02)
    epoch[0] = 40
    assert module._get_alignment_schedule() == pytest.approx(0.0)


def test_semantic_local_alignment_is_finite_and_ignores_boundaries() -> None:
    module = _alignment_module(mode="semantic_local", radius=1)
    module.alignment_exclude_boundaries = True
    x = torch.randn(1, 4, 4, 4, requires_grad=True)
    y = torch.randn(1, 4, 4, 4, requires_grad=True)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[:, :, 2:] = 1

    loss, stats = module._compute_alignment_loss([x], [y], target=target)
    loss.backward()

    assert torch.isfinite(loss)
    assert stats["align_valid_tokens_l-1"].item() > 0


def test_semantic_stop_gradient_removes_cross_modal_second_derivative() -> None:
    module = _alignment_module(mode="semantic_local", radius=1)
    module.alignment_exclude_boundaries = False
    module.alignment_semantic_stop_gradient = True
    x = torch.randn(1, 4, 3, 3, requires_grad=True)
    y = torch.randn(1, 4, 3, 3, requires_grad=True)
    target = torch.tensor(
        [[[0, 0, 0], [0, 1, 1], [0, 1, 1]]]
    )

    loss, _ = module._compute_alignment_loss([x], [y], target=target)
    grad_x, grad_y = torch.autograd.grad(
        loss, (x, y), create_graph=True
    )
    cross_xy = torch.autograd.grad(
        grad_x.sum(), y, allow_unused=True
    )[0]
    cross_yx = torch.autograd.grad(
        grad_y.sum(), x, allow_unused=True
    )[0]

    assert torch.count_nonzero(grad_x) > 0
    assert torch.count_nonzero(grad_y) > 0
    assert cross_xy is None or torch.count_nonzero(cross_xy) == 0
    assert cross_yx is None or torch.count_nonzero(cross_yx) == 0


def test_soft_correspondence_is_finite_and_updates_both_modalities() -> None:
    module = _alignment_module(mode="soft_correspondence", radius=1)
    module.alignment_exclude_boundaries = False
    x = torch.randn(2, 4, 3, 3, requires_grad=True)
    y = torch.randn(2, 4, 3, 3, requires_grad=True)
    target = torch.tensor(
        [
            [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
            [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
        ]
    )

    loss, stats = module._compute_alignment_loss([x], [y], target=target)
    loss.backward()

    assert torch.isfinite(loss)
    assert stats["align_valid_tokens_l-1"].item() > 0
    assert torch.count_nonzero(x.grad) > 0
    assert torch.count_nonzero(y.grad) > 0


def test_prototype_alignment_is_finite_and_class_discriminative() -> None:
    module = _alignment_module(mode="prototype", radius=1)
    module.alignment_exclude_boundaries = False
    x = torch.randn(2, 4, 3, 3, requires_grad=True)
    y = torch.randn(2, 4, 3, 3, requires_grad=True)
    target = torch.tensor(
        [
            [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
            [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
        ]
    )

    loss, stats = module._compute_alignment_loss([x], [y], target=target)
    loss.backward()

    assert torch.isfinite(loss)
    assert 0.0 <= stats["align_proto_acc_l-1"].item() <= 1.0
    assert torch.count_nonzero(x.grad) > 0
    assert torch.count_nonzero(y.grad) > 0


def test_compiled_checkpoint_keys_are_normalized() -> None:
    module = MultiModalSegmentationModule.__new__(MultiModalSegmentationModule)
    nn.Module.__init__(module)
    weight = torch.randn(2, 2)
    checkpoint = {
        "state_dict": {
            "encoder._orig_mod.layer.weight": weight,
            "fusion._orig_mod.block.weight": weight,
            "decoder.layer.weight": weight,
        }
    }

    module.on_load_checkpoint(checkpoint)

    assert set(checkpoint["state_dict"]) == {
        "encoder.layer.weight",
        "fusion.block.weight",
        "decoder.layer.weight",
    }
