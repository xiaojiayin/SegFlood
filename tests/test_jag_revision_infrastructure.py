from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from scripts.run.jag_supplement.profile_model import (
    _ProfileWrapper,
    format_optional,
    make_batch,
    profile_operations,
)
from scripts.run.summarize_jag_revision_results import aggregate_runs, collect_runs
from src.eval import instantiate_model


def test_eval_model_instantiation_preserves_dynamic_component_chain() -> None:
    cfg = OmegaConf.create({"model": {"_target_": "example.Model", "fusion": {"_target_": "example.Fusion"}}})
    sentinel = object()

    with patch("src.eval.hydra.utils.instantiate", return_value=sentinel) as instantiate:
        assert instantiate_model(cfg) is sentinel

    instantiate.assert_called_once_with(cfg.model, _recursive_=False)


def test_jag_summary_collects_completed_baseline_and_multiseed(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "fusion_baselines"
    multiseed_dir = tmp_path / "multiseed"
    baseline_dir.mkdir()
    multiseed_dir.mkdir()
    metric_table = (
        "│ test/iou │ 0.80 │\n"
        "│ test/water_iou │ 0.75 │\n"
        "│ test/f1 │ 0.90 │\n"
        "│ test/acc │ 0.95 │\n"
        "Best ckpt path: /tmp/model.ckpt\n"
    )
    (baseline_dir / "resnet50_resnet50_gffloodnet_jagfusion_concat_101.out").write_text(
        metric_table, encoding="utf-8"
    )
    for seed, value in ((42, "0.70"), (123, "0.90")):
        (multiseed_dir / f"resnet50_resnet50_gffloodnet_seed-{seed}_concat_{seed}.out").write_text(
            f"│ test/iou │ {value} │\n", encoding="utf-8"
        )
    (multiseed_dir / "resnet50_resnet50_gffloodnet_seed-2026_concat_2026.out").write_text(
        "training still running", encoding="utf-8"
    )

    runs, skipped = collect_runs(tmp_path)
    assert len(runs) == 3
    assert len(skipped) == 1
    assert {run["source"] for run in runs} == {"fusion_baselines", "multiseed"}
    assert {run["seed"] for run in runs if run["source"] == "multiseed"} == {"42", "123"}

    summary = aggregate_runs(runs)
    multiseed_iou = next(
        row
        for row in summary
        if row["source"] == "multiseed" and row["metric"] == "test/iou"
    )
    assert multiseed_iou["n"] == 2
    assert multiseed_iou["mean"] == pytest.approx(0.8)
    assert multiseed_iou["std"] == pytest.approx(2**0.5 / 10)


class _DictionaryModel(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"main_logits": batch["image"] * 2}


def test_profile_helpers_support_cpu_and_dictionary_models() -> None:
    cfg = OmegaConf.create(
        {"data": {"modal_type": "optical", "optical_channels": 3, "sar_channels": 1}}
    )
    batch = make_batch(cfg, batch_size=2, image_size=8, device=torch.device("cpu"))
    wrapper = _ProfileWrapper(_DictionaryModel(), ["image"])

    output = wrapper(batch["image"])
    flops, macs, backend = profile_operations(_DictionaryModel(), batch)
    assert output.shape == (2, 3, 8, 8)
    assert (flops is None) == (macs is None)
    assert backend
    assert format_optional(None) == ""
    assert format_optional(1_500_000_000, 1e9) == "1.500"


def test_profile_batch_uses_concatenated_channels_for_early_fusion() -> None:
    cfg = OmegaConf.create(
        {
            "data": {
                "modal_type": "dual",
                "optical_channels": 4,
                "sar_channels": 2,
            },
            "model": {
                "encoder": {
                    "optical_channels": 6,
                    "sar_channels": 0,
                }
            },
        }
    )

    batch = make_batch(
        cfg, batch_size=2, image_size=8, device=torch.device("cpu")
    )

    assert set(batch) == {"image", "mask"}
    assert batch["image"].shape == (2, 6, 8, 8)
