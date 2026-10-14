"""Evidence artifacts of the standalone CIFAR-10 smoke, without downloads or a real ViT."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.image
import torch
import yaml
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset

from scripts.smoke import smoke_vit_lora_cifar10 as smoke
from scripts.smoke.lora_verification_reporting import (
    FAIL, NOT_RUN, PASS, evaluate_checks, render_verification_summary,
    save_lora_grad_norm_plot, save_loss_plot,
)


ARTIFACTS = {
    "adapter_head.pt", "resolved_config.yaml", "report.json",
    "loss.png", "lora_grad_norm.png", "verification_summary.md",
}


class FakeBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attention = nn.Module()
        self.attention.q_proj = nn.Linear(d, d)
        self.attention.k_proj = nn.Linear(d, d)
        self.attention.v_proj = nn.Linear(d, d)


class FakeViT(nn.Module):
    def __init__(self, d=8, blocks=12):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=d, num_hidden_layers=blocks)
        self.layers = nn.ModuleList(FakeBlock(d) for _ in range(blocks))

    def forward(self, pixel_values):
        x = pixel_values.flatten(1)[:, :self.config.hidden_size]
        for block in self.layers:
            attn = block.attention
            x = x + .01 * (attn.q_proj(x) + attn.k_proj(x) + attn.v_proj(x))
        return SimpleNamespace(last_hidden_state=x[:, None, :])


class FakeCIFAR10(Dataset):
    def __init__(self, root, train, download, transform):
        assert train and not download
        self.images = torch.rand(80, 3, 8, 8, generator=torch.Generator().manual_seed(0))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index]), index % 10


class FakeProcessor:
    def __call__(self, images, return_tensors):
        return {"pixel_values": images.unsqueeze(0)}


def run_smoke(output_dir: Path, **overrides) -> dict:
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "conf" / "lora_standalone.yaml")
    cfg.pop("hydra")
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    hydra_config = SimpleNamespace(get=lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(output_dir))))
    with patch.object(smoke, "CIFAR10", FakeCIFAR10), \
            patch.object(smoke, "AutoImageProcessor", SimpleNamespace(from_pretrained=lambda _: FakeProcessor())), \
            patch.object(smoke, "HydraConfig", hydra_config), \
            patch("model.vit.vit_lora_frame_encoder.ViTModel.from_pretrained", return_value=FakeViT()):
        return smoke.run(cfg)


def test_default_smoke_writes_six_artifacts_from_measurements(tmp_path):
    returned = run_smoke(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == ARTIFACTS
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report == json.loads(json.dumps(returned))
    assert yaml.safe_load((tmp_path / "resolved_config.yaml").read_text(encoding="utf-8"))["lora"]["rank"] == 8
    checkpoint = torch.load(tmp_path / "adapter_head.pt", map_location="cpu", weights_only=True)
    assert set(checkpoint) == {"metadata", "adapter", "classifier"}
    for name in ("loss.png", "lora_grad_norm.png"):
        assert matplotlib.image.imread(tmp_path / name).ndim == 3

    history = report["history"]
    assert [item["step"] for item in history] == list(range(1, 21))
    for item in history:
        assert set(item) >= {"loss", "train_accuracy", "adapter_grad_norm_sum", "head_grad_norm_sum",
                             "lora_a_grad_norm_sum", "lora_b_grad_norm_sum"}
        assert math.isclose(item["adapter_grad_norm_sum"],
                            item["lora_a_grad_norm_sum"] + item["lora_b_grad_norm_sum"], rel_tol=1e-6)
    # B starts at zero, so the first A gradient is exactly zero and the first B gradient is not.
    assert history[0]["lora_a_grad_norm_sum"] == 0.0
    assert history[0]["lora_b_grad_norm_sum"] > 0
    assert history[1]["lora_a_grad_norm_sum"] > 0

    verification = report["verification"]
    assert verification["base_frozen_requires_grad"] is True
    assert verification["base_parameter_tensors"] == 12 * 3 * 2
    assert verification["base_gradient_tensors_max"] == 0
    assert verification["base_changed_tensors"] == 0
    assert verification["optimizer_matches_trainable_parameters"] is True
    assert verification["lora_a_tensors"] == verification["lora_b_tensors"] == 24
    assert 0 < verification["lora_a_changed_tensors"] <= 24
    assert 0 < verification["lora_b_changed_tensors"] <= 24
    assert verification["lora_a_update_norm_sum"] > 0 and verification["lora_b_update_norm_sum"] > 0
    assert verification["head_tensors"] == verification["head_changed_tensors"] == 2
    assert verification["checkpoint_roundtrip_passed"] is True
    assert verification["reload_logits_max_abs_diff"] <= 1e-6
    assert report["checks"] == evaluate_checks(report)
    assert {check["status"] for check in report["checks"].values()} == {PASS}

    summary = (tmp_path / "verification_summary.md").read_text(encoding="utf-8")
    assert summary == render_verification_summary(report)
    assert f"max abs diff {verification['reload_logits_max_abs_diff']:.6g}" in summary
    assert f"| 1 | {history[0]['loss']:.6g} | 0 | {history[0]['lora_b_grad_norm_sum']:.6g} |" in summary
    assert "| Checkpoint保存・再読込後のeval logits一致 | PASS |" in summary
    assert f"HEAD `{report['git']['commit']}`" in summary
    assert "LoRAのみの学習ではない" in summary and "mini-batch lossの低下だけでは" in summary
    assert all(f"`{name}`" in summary for name in ARTIFACTS)
    assert NOT_RUN not in summary and FAIL not in summary


def test_one_step_run_marks_lora_a_update_not_run(tmp_path):
    report = run_smoke(tmp_path, **{"training.max_steps": 1})

    assert report["history"][0]["lora_a_grad_norm_sum"] == 0.0
    assert report["verification"]["lora_a_changed_tensors"] == 0
    assert report["checks"]["lora_a_updated"]["status"] == NOT_RUN
    assert report["checks"]["lora_b_updated"]["status"] == PASS
    summary = (tmp_path / "verification_summary.md").read_text(encoding="utf-8")
    assert "| LoRA A: 学習前後で変化 | NOT_RUN |" in summary
    assert report["checks"]["lora_a_updated"]["reason"] in summary


def test_disabled_lora_marks_a_and_b_not_run(tmp_path):
    report = run_smoke(tmp_path, **{"lora.enabled": False, "training.max_steps": 3})

    assert {path.name for path in tmp_path.iterdir()} == ARTIFACTS
    assert all(item["lora_a_grad_norm_sum"] is None and item["lora_b_grad_norm_sum"] is None
               for item in report["history"])
    assert report["checks"]["lora_a_updated"]["status"] == NOT_RUN
    assert report["checks"]["lora_b_updated"]["status"] == NOT_RUN
    assert report["checks"]["head_updated"]["status"] == PASS
    assert report["checks"]["base_unchanged"]["status"] == PASS
    summary = (tmp_path / "verification_summary.md").read_text(encoding="utf-8")
    assert "| 1 | " in summary and "| n/a | n/a |" in summary


def measured_report(**verification_overrides) -> dict:
    verification = {
        "base_parameter_tensors": 4, "base_parameters": 100, "base_frozen_requires_grad": True,
        "base_gradient_tensors_max": 0, "base_changed_tensors": 0,
        "optimizer_matches_trainable_parameters": True,
        "lora_a_tensors": 2, "lora_a_changed_tensors": 2, "lora_a_update_norm_sum": 0.1,
        "lora_b_tensors": 2, "lora_b_changed_tensors": 2, "lora_b_update_norm_sum": 0.2,
        "head_tensors": 2, "head_changed_tensors": 2,
        "reload_logits_max_abs_diff": 0.0, "checkpoint_roundtrip_passed": True,
        "checkpoint_roundtrip_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        **verification_overrides,
    }
    report = {
        "model_checkpoint": "test-vit", "selected_targets": ["q", "v"], "dataset": "CIFAR10 train",
        "device": "cpu", "git": {"commit": "abc", "dirty": None}, "steps": 2,
        "trainable_parameters": 10, "head_update_norm_sum": 0.3,
        "lora": {"enabled": True, "rank": 8, "alpha": 8.0, "dropout": 0.0},
        "seed": 0, "train_subset_size": 64, "batch_size": 16, "verification": verification,
        "history": [
            {"step": 1, "loss": 2.5, "head_grad_norm_sum": 1.0,
             "lora_a_grad_norm_sum": 0.0, "lora_b_grad_norm_sum": 0.1},
            {"step": 2, "loss": 2.0, "head_grad_norm_sum": 1.0,
             "lora_a_grad_norm_sum": 0.02, "lora_b_grad_norm_sum": 0.1},
        ],
    }
    report["checks"] = evaluate_checks(report)
    return report


def test_checks_fail_from_measured_values_instead_of_assuming_pass():
    report = measured_report(lora_a_changed_tensors=0, base_changed_tensors=1,
                             checkpoint_roundtrip_passed=False, reload_logits_max_abs_diff=0.5)
    checks = report["checks"]
    assert checks["lora_a_updated"]["status"] == FAIL
    assert checks["base_unchanged"]["status"] == FAIL
    assert checks["checkpoint_roundtrip"]["status"] == FAIL
    assert checks["lora_b_updated"]["status"] == PASS

    summary = render_verification_summary(report)
    assert "| LoRA A: 学習前後で変化 | FAIL |" in summary
    assert "| Checkpoint保存・再読込後のeval logits一致 | FAIL | max abs diff 0.5" in summary
    assert "Base凍結の検証はPASSしていない: base_unchanged=FAIL" in summary
    assert "lora_a_updated=FAIL" in summary
    assert "dirty: unknown" in summary


def test_grad_plot_draws_zero_and_leaves_missing_values_as_gaps(tmp_path):
    history = [
        {"step": 1, "lora_a_grad_norm_sum": 0.0, "lora_b_grad_norm_sum": 0.12},
        {"step": 2, "lora_a_grad_norm_sum": 0.02, "lora_b_grad_norm_sum": 0.13},
        {"step": 3, "lora_a_grad_norm_sum": None, "lora_b_grad_norm_sum": 0.11},
    ]
    figure = save_lora_grad_norm_plot(history, tmp_path / "lora_grad_norm.png")
    axis = figure.axes[0]
    lines = {line.get_label(): line for line in axis.lines}
    assert set(lines) == {"LoRA A", "LoRA B"}
    assert [label.get_text() for label in axis.get_legend().get_texts()] == ["LoRA A", "LoRA B"]
    assert list(lines["LoRA A"].get_xdata()) == [1, 2, 3]
    a_values = list(lines["LoRA A"].get_ydata())
    assert a_values[:2] == [0.0, 0.02] and math.isnan(a_values[2])
    assert list(lines["LoRA B"].get_ydata()) == [0.12, 0.13, 0.11]
    assert list(axis.get_xticks()) == [1, 2, 3]
    assert matplotlib.image.imread(tmp_path / "lora_grad_norm.png").ndim == 3


def test_grad_plot_without_lora_values_draws_no_zero_line(tmp_path):
    history = [{"step": step, "lora_a_grad_norm_sum": None, "lora_b_grad_norm_sum": None} for step in (1, 2)]
    figure = save_lora_grad_norm_plot(history, tmp_path / "lora_grad_norm.png")
    assert len(figure.axes[0].lines) == 0
    assert (tmp_path / "lora_grad_norm.png").exists()


def test_loss_plot_uses_history_values_for_every_step(tmp_path):
    history = [{"step": step, "loss": loss} for step, loss in zip(range(1, 21), torch.linspace(2.5, 0.1, 20).tolist())]
    figure = save_loss_plot(history, tmp_path / "loss.png")
    axis = figure.axes[0]
    (line,) = axis.lines
    assert list(line.get_xdata()) == list(range(1, 21))
    assert list(line.get_ydata()) == [item["loss"] for item in history]
    assert list(axis.get_xticks()) == list(range(1, 21))
    assert axis.get_ylabel() == "Loss"
    assert axis.get_xlabel().startswith("Step") and "mini-batch" in axis.get_xlabel()
