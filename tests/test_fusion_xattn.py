"""MA-XAttn 组件消融与标准 cross-attention 的单元测试。"""

import pytest
import torch

from src.models.fusion import FeatureFusion, SpatialAttBlock


@pytest.mark.parametrize(
    ("components", "attention_type", "affinity_order"),
    [
        ("spatial", "ma", "opt_sar"),
        ("spatial", "ma", "sar_opt"),
        ("channel", "ma", "opt_sar"),
        ("channel", "ma", "sar_opt"),
        ("spatial+channel", "ma", "opt_sar"),
        ("spatial+channel", "ma", "sar_opt"),
        ("spatial", "standard", "opt_sar"),
    ],
)
def test_xattn_modes_shape_and_backward(
    components: str, attention_type: str, affinity_order: str
) -> None:
    fusion = FeatureFusion(
        feature_channels=[8],
        optical_channels=[6],
        sar_channels=[4],
        fusion_type="xattn",
        xattn_apply_levels=[0],
        xattn_reduction=2,
        xattn_components=components,
        xattn_affinity_order=affinity_order,
        xattn_attention_type=attention_type,
    )
    optical = torch.randn(2, 6, 3, 4, requires_grad=True)
    # 同时覆盖管理层的空间尺寸对齐。
    sar = torch.randn(2, 4, 2, 3, requires_grad=True)

    output = fusion([optical], [sar])[0]
    assert output.shape == (2, 6, 3, 4)

    output.square().mean().backward()
    assert optical.grad is not None
    assert sar.grad is not None
    assert all(parameter.grad is not None for parameter in fusion.parameters())


def test_standard_attention_matches_bidirectional_qk_t_definition() -> None:
    block = SpatialAttBlock(
        in_channels=1,
        att_channels=1,
        out_channels=1,
        attention_type="standard",
    )
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.fill_(1.0)

    optical = torch.tensor([[[[1.0, 2.0]]]])
    sar = torch.tensor([[[[3.0, 4.0]]]])
    actual = block(optical, sar)

    q_opt = optical.flatten(2).transpose(1, 2)
    k_opt = optical.flatten(2)
    v_opt = q_opt
    q_sar = sar.flatten(2).transpose(1, 2)
    k_sar = sar.flatten(2)
    v_sar = q_sar
    opt_queries_sar = torch.softmax(torch.bmm(q_opt, k_sar), dim=-1)
    sar_queries_opt = torch.softmax(torch.bmm(q_sar, k_opt), dim=-1)
    expected = (
        torch.bmm(opt_queries_sar, v_sar)
        + torch.bmm(sar_queries_opt, v_opt)
    ).transpose(1, 2).reshape_as(actual)

    torch.testing.assert_close(actual, expected)


def test_reverse_affinity_order_changes_ma_xattn_result() -> None:
    torch.manual_seed(7)
    opt_sar = SpatialAttBlock(4, 2, 4, affinity_order="opt_sar")
    sar_opt = SpatialAttBlock(4, 2, 4, affinity_order="sar_opt")
    sar_opt.load_state_dict(opt_sar.state_dict(), strict=True)
    optical = torch.randn(1, 4, 2, 3)
    sar = torch.randn(1, 4, 2, 3)

    forward_order = opt_sar(optical, sar)
    reverse_order = sar_opt(optical, sar)

    assert forward_order.shape == reverse_order.shape == (1, 4, 2, 3)
    assert not torch.allclose(forward_order, reverse_order)


def test_local_cross_alignment_bias_changes_cross_modal_output() -> None:
    torch.manual_seed(9)
    without_bias = SpatialAttBlock(4, 2, 4, enable_align_bias=False)
    local_bias = SpatialAttBlock(
        4,
        2,
        4,
        enable_align_bias=True,
        align_bias_scale=2.0,
        align_bias_mode="local_cross",
        align_bias_radius=1,
    )
    local_bias.load_state_dict(without_bias.state_dict(), strict=True)
    optical = torch.randn(1, 4, 3, 3)
    sar = torch.randn(1, 4, 3, 3)

    plain = without_bias(optical, sar)
    guided = local_bias(optical, sar)

    assert guided.shape == plain.shape
    assert not torch.allclose(guided, plain)


def test_learnable_alignment_bias_starts_from_no_bias() -> None:
    torch.manual_seed(10)
    plain = SpatialAttBlock(4, 2, 4, enable_align_bias=False)
    learnable = SpatialAttBlock(
        4,
        2,
        4,
        enable_align_bias=True,
        align_bias_learnable=True,
        align_bias_init=0.0,
        align_bias_mode="local_cross",
        align_bias_radius=1,
    )
    missing, unexpected = learnable.load_state_dict(
        plain.state_dict(), strict=False
    )
    assert missing == ["align_bias_weight"]
    assert unexpected == []
    optical = torch.randn(2, 4, 3, 3)
    sar = torch.randn(2, 4, 3, 3)

    expected = plain(optical, sar)
    actual = learnable(optical, sar)
    torch.testing.assert_close(actual, expected)

    actual.square().mean().backward()
    assert learnable.align_bias_weight is not None
    assert learnable.align_bias_weight.grad is not None


def test_gated_component_fusion_starts_from_original_sum() -> None:
    torch.manual_seed(13)
    summed = FeatureFusion(
        feature_channels=[8],
        optical_channels=[8],
        sar_channels=[8],
        fusion_type="xattn",
        xattn_apply_levels=[0],
        xattn_components="spatial+channel",
        xattn_component_fusion="sum",
    )
    gated = FeatureFusion(
        feature_channels=[8],
        optical_channels=[8],
        sar_channels=[8],
        fusion_type="xattn",
        xattn_apply_levels=[0],
        xattn_components="spatial+channel",
        xattn_component_fusion="gated",
    )
    missing, unexpected = gated.load_state_dict(summed.state_dict(), strict=False)
    assert missing == ["strategy.component_logits.0"]
    assert unexpected == []
    optical = [torch.randn(2, 8, 2, 3)]
    sar = [torch.randn(2, 8, 2, 3)]

    expected = summed(optical, sar)[0]
    actual = gated(optical, sar)[0]
    torch.testing.assert_close(actual, expected)

    actual.mean().backward()
    logits = gated.strategy.component_logits
    assert logits is not None
    assert logits[0].grad is not None


def test_residual_gated_attention_starts_from_dual_addition() -> None:
    torch.manual_seed(17)
    fusion = FeatureFusion(
        feature_channels=[8],
        optical_channels=[8],
        sar_channels=[8],
        fusion_type="xattn",
        xattn_apply_levels=[0],
        xattn_components="spatial+channel",
        xattn_component_fusion="residual_gated",
        xattn_attention_residual_init=0.0,
    )
    optical = [torch.randn(2, 8, 2, 3)]
    sar = [torch.randn(2, 8, 2, 3)]

    expected = (
        fusion.strategy.project_opt[0](optical[0])
        + fusion.strategy.project_sar[0](sar[0])
    )
    actual = fusion(optical, sar)[0]
    torch.testing.assert_close(actual, expected)

    actual.square().mean().backward()
    scales = fusion.strategy.attention_residual_scale
    logits = fusion.strategy.component_logits
    assert scales is not None and scales[0].grad is not None
    assert logits is not None and logits[0].grad is not None


def test_default_configuration_keeps_legacy_state_dict_and_behavior() -> None:
    torch.manual_seed(11)
    legacy_default = FeatureFusion(
        feature_channels=[8],
        optical_channels=[8],
        sar_channels=[8],
        fusion_type="xattn",
        xattn_apply_levels=[0],
    )
    explicit_default = FeatureFusion(
        feature_channels=[8],
        optical_channels=[8],
        sar_channels=[8],
        fusion_type="xattn",
        xattn_apply_levels=[0],
        xattn_components="spatial+channel",
        xattn_affinity_order="opt_sar",
        xattn_attention_type="ma",
    )
    explicit_default.load_state_dict(legacy_default.state_dict(), strict=True)
    optical = [torch.randn(1, 8, 2, 3)]
    sar = [torch.randn(1, 8, 2, 3)]

    assert legacy_default.state_dict().keys() == explicit_default.state_dict().keys()
    torch.testing.assert_close(
        legacy_default(optical, sar)[0],
        explicit_default(optical, sar)[0],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xattn_components": "unknown"},
        {"xattn_affinity_order": "unknown"},
        {"xattn_attention_type": "unknown"},
        {"xattn_reduction": 0},
        {"xattn_attention_type": "standard"},
        {
            "xattn_attention_type": "standard",
            "xattn_components": "spatial",
            "xattn_affinity_order": "sar_opt",
        },
        {
            "xattn_attention_type": "standard",
            "xattn_components": "spatial",
            "xattn_align_bias": True,
        },
    ],
)
def test_invalid_xattn_configuration_fails_fast(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        FeatureFusion(
            feature_channels=[8],
            optical_channels=[8],
            sar_channels=[8],
            fusion_type="xattn",
            **kwargs,
        )


def test_attention_block_rejects_mismatched_shapes() -> None:
    block = SpatialAttBlock(4, 2, 4)
    with pytest.raises(ValueError, match="shape"):
        block(torch.randn(1, 4, 2, 2), torch.randn(1, 4, 2, 3))
