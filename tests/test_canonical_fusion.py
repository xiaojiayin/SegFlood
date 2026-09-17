"""Tests for canonical baselines without MA-XAttn-specific enhancements."""

import pytest
import torch

from src.models.fusion import FeatureFusion


def test_identity_fusion_preserves_single_stream_features() -> None:
    fusion = FeatureFusion(
        feature_channels=[8],
        optical_channels=[8],
        sar_channels=None,
        fusion_type="identity",
    )
    feature = torch.randn(2, 8, 3, 4)

    output = fusion(optical_features=[feature], sar_features=None)

    assert output[0] is feature
    assert fusion.output_channels == [8]


def test_identity_fusion_rejects_dual_stream_configuration() -> None:
    with pytest.raises(ValueError, match="单编码器"):
        FeatureFusion(
            feature_channels=[8],
            optical_channels=[8],
            sar_channels=[8],
            fusion_type="identity",
        )


def test_canonical_add_is_exact_sum_when_channels_match() -> None:
    fusion = FeatureFusion(
        feature_channels=[8],
        optical_channels=[8],
        sar_channels=[8],
        fusion_type="canonical_add",
    )
    optical = torch.randn(2, 8, 3, 4)
    sar = torch.randn(2, 8, 3, 4)

    actual = fusion([optical], [sar])[0]

    torch.testing.assert_close(actual, optical + sar)
    assert not list(fusion.parameters())


@pytest.mark.parametrize(
    "fusion_type", ["canonical_gated", "canonical_cross"]
)
def test_canonical_learned_fusion_shape_and_backward(
    fusion_type: str,
) -> None:
    fusion = FeatureFusion(
        feature_channels=[8],
        optical_channels=[8],
        sar_channels=[8],
        fusion_type=fusion_type,
        xattn_apply_levels=[0],
        xattn_reduction=2,
    )
    optical = torch.randn(2, 8, 3, 4, requires_grad=True)
    sar = torch.randn(2, 8, 3, 4, requires_grad=True)

    output = fusion([optical], [sar])[0]
    output.square().mean().backward()

    assert output.shape == optical.shape
    assert optical.grad is not None
    assert sar.grad is not None
    assert all(parameter.grad is not None for parameter in fusion.parameters())
    assert not hasattr(fusion.strategy, "gamma")
    assert not hasattr(fusion.strategy, "res_proj")
