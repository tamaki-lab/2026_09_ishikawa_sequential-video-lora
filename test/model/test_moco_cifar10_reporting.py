"""Standalone MoCo CIFAR-10 runner: 2-view data, artifacts, checkpoint/resume and failure records, without downloads."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.image
import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from scripts.smoke import smoke_moco_cifar10 as smoke
from scripts.smoke.moco_verification_reporting import (
    ARTIFACTS, FAIL, NOT_RUN, PASS, build_diagnostics, evaluate_checks, gate_statuses, render_verification_summary,
    write_verification_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_NAMES = {name for name, _ in ARTIFACTS}
SMALL = {"moco.queue_size": 32, "training.batch_size": 8}
HISTORY_KEYS = {
    "global_step", "sample_count", "lr", "loss", "positive_cosine_mean", "negative_cosine_mean",
    "hardest_negative_cosine_mean", "hardest_negative_cosine_max", "z_q_norm_min", "z_q_norm_mean",
    "z_k_norm_min", "z_k_norm_mean", "q_variance_mean", "q_variance_min", "k_variance_mean", "k_variance_min",
    "query_backbone_grad_norm", "query_projector_grad_norm", "query_backbone_update_norm",
    "query_projector_update_norm", "ema_max_abs_error", "ema_within_tolerance", "key_grad_tensors",
    "queue_ptr", "real_key_count", "random_remaining_count",
}


class FakeCIFAR10(Dataset):
    """640 random 32x32 PIL images with labels; mirrors torchvision.datasets.CIFAR10's interface."""

    size = 640

    def __init__(self, root, train, download, transform):
        assert train and not download
        rng = np.random.default_rng(0)
        self.images = [Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)) for _ in range(self.size)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index]), index % 10


def load_cfg(**overrides):
    cfg = OmegaConf.load(ROOT / "conf" / "moco_standalone.yaml")
    cfg.pop("hydra")
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def run_smoke(output_dir: Path, dataset=FakeCIFAR10, **overrides) -> dict:
    hydra_config = SimpleNamespace(get=lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(output_dir))))
    with patch.object(smoke, "HydraConfig", hydra_config):
        if dataset is None:
            return smoke.run(load_cfg(**overrides))
        with patch.object(smoke, "CIFAR10", dataset):
            return smoke.run(load_cfg(**overrides))


def read_report(output_dir: Path) -> dict:
    return json.loads((output_dir / "report.json").read_text(encoding="utf-8"))


def test_hydra_composes_the_standalone_config_without_touching_config_yaml():
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = compose(config_name="moco_standalone", overrides=["data.root=/data/CIFAR10", "training.device=cpu"],
                      return_hydra_config=True)
    assert cfg.data.root == "/data/CIFAR10" and cfg.data.download is False
    assert cfg.training.scheduler == "none" and cfg.training.resume is None and cfg.training.max_steps == 20
    assert (cfg.moco.queue_size, cfg.moco.momentum, cfg.moco.temperature) == (1024, 0.999, 0.2)
    assert (cfg.training.batch_size, cfg.training.learning_rate, cfg.training.weight_decay) == (32, 0.00375, 0.0001)
    assert cfg.logging.comet_enabled is False
    assert cfg.hydra.job.chdir is False and cfg.hydra.run.dir.startswith("log/moco_standalone/")


def test_augmentation_matches_the_adopted_moco_v2_cifar_pipeline():
    steps = smoke.moco_v2_cifar_augmentation().transforms
    assert [type(step).__name__ for step in steps] == [
        "RandomResizedCrop", "RandomApply", "RandomGrayscale", "RandomApply", "RandomHorizontalFlip",
        "ToTensor", "Normalize",
    ]
    crop, jitter, gray, blur, flip, _, normalize = steps
    assert crop.size == (32, 32) and crop.scale == (0.2, 1.0) and crop.ratio == pytest.approx((0.75, 4 / 3))
    color = jitter.transforms[0]
    assert jitter.p == 0.8 and color.brightness == color.contrast == color.saturation == pytest.approx((0.6, 1.4))
    assert color.hue == pytest.approx((-0.1, 0.1))
    assert gray.p == 0.2 and blur.p == 0.5 and flip.p == 0.5
    assert blur.transforms[0].kernel_size == (3, 3) and blur.transforms[0].sigma == pytest.approx((0.1, 2.0))
    assert tuple(normalize.mean) == (0.485, 0.456, 0.406) and tuple(normalize.std) == (0.229, 0.224, 0.225)


def test_a02_two_view_dataset_pairs_independent_draws_of_one_index_without_labels():
    calls = []

    def recording_transform(image):
        calls.append(image)
        return torch.tensor(len(calls))

    base = FakeCIFAR10(root=None, train=True, download=False, transform=smoke.TwoCropsTransform(recording_transform))
    im_q, im_k, index = smoke.TwoViewDataset(base)[7]
    assert index == 7 and len(calls) == 2 and calls[0] is calls[1] is base.images[7]
    assert (im_q.item(), im_k.item()) == (1, 2)  # two separate draws

    augmented = smoke.TwoViewDataset(FakeCIFAR10(
        root=None, train=True, download=False,
        transform=smoke.TwoCropsTransform(smoke.moco_v2_cifar_augmentation())))
    torch.manual_seed(0)
    for i in range(5):
        item = augmented[i]
        assert len(item) == 3 and item[2] == i
        assert item[0].shape == item[1].shape == (3, 32, 32) and item[0].dtype == torch.float32
        assert not torch.equal(item[0], item[1])

    order = smoke.single_epoch_order(len(augmented), 0)
    loader = DataLoader(augmented, batch_size=32, sampler=order[:64].tolist(), drop_last=True, num_workers=0)
    batches = list(loader)
    assert len(batches) == 2 and all(len(batch) == 3 for batch in batches)
    for step, (batch_q, batch_k, batch_index) in enumerate(batches):
        assert batch_q.shape == batch_k.shape == (32, 3, 32, 32)
        assert torch.equal(batch_index, order[step * 32:(step + 1) * 32])  # sample indices, not labels


def test_default_20step_fake_smoke_writes_all_artifacts_from_measurements(tmp_path):
    returned = run_smoke(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == ARTIFACT_NAMES
    report = read_report(tmp_path)
    assert report == json.loads(json.dumps(returned, ensure_ascii=False))
    assert report["schema_version"] == smoke.SCHEMA_VERSION and report["status"] == "completed"
    assert report["run_kind"] == "custom"  # 640 fake images are not the 50,000-image Gate B dataset
    assert report["gates"]["B"]["status"] == NOT_RUN
    assert yaml.safe_load((tmp_path / "resolved_config.yaml").read_text(encoding="utf-8"))["moco"]["queue_size"] == 1024
    for name in ("loss.png", "similarity.png", "grad_update.png", "queue.png"):
        assert matplotlib.image.imread(tmp_path / name).ndim == 3

    history = report["history"]
    assert [item["global_step"] for item in history] == list(range(1, 21))
    assert [item["sample_count"] for item in history] == [32 * n for n in range(1, 21)]
    for item in history:
        assert set(item) == HISTORY_KEYS and all(value is not None for value in item.values())
        assert item["lr"] == 0.00375 and item["key_grad_tensors"] == 0 and item["ema_within_tolerance"] is True
    assert [item["random_remaining_count"] for item in history] == [1024 - 32 * n for n in range(1, 21)]
    assert (history[-1]["queue_ptr"], history[-1]["real_key_count"], history[-1]["random_remaining_count"]) == \
        (640, 640, 384)

    verification = report["verification"]
    assert verification["observed_shapes"] == {
        "im_q": [[32, 3, 32, 32]], "im_k": [[32, 3, 32, 32]], "q": [[32, 128]], "k": [[32, 128]],
        "logits": [[32, 1025]],
    }
    assert verification["final_queue"]["queue_ptr"] == 640 and verification["final_queue"]["finite"] is True
    assert verification["query_backbone_changed_tensors"] > 0 and verification["query_projector_changed_tensors"] > 0
    assert verification["key_requires_grad_tensors"] == 0 and verification["key_grad_tensors_max"] == 0
    assert verification["checkpoint_roundtrip"]["passed"] is True
    assert verification["bn_running_buffers_query_key_differing"]["differing"] > 0
    assert report["checks"] == evaluate_checks(report)
    assert {check["status"] for check in report["checks"].values()} == {PASS}
    assert report["dataset"]["sample_count"] == 640 and report["dataset"]["samples_consumed"] == 640
    assert report["provenance"]["git"]["commit"] and report["reference"]["commit"].startswith("a419ef2")
    assert report["model"]["initial_query_state_sha256"] and report["moco"]["initial_queue_sha256"]

    checkpoint = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["format"] == smoke.CHECKPOINT_FORMAT and checkpoint["resumable"] is True
    assert checkpoint["global_step"] == 20 and checkpoint["scheduler"] is None
    assert checkpoint["sampler"]["cursor"] == 640 and checkpoint["history"] == history
    assert {"queue", "queue_ptr", "real_key_count", "backbone_k.net.bn1.running_mean",
            "backbone_q.net.bn1.running_mean", "projector_k.2.weight"} <= checkpoint["model"].keys()
    assert all(t.device.type == "cpu" and not t.requires_grad and t.grad_fn is None
               for t in checkpoint["model"].values())
    assert set(checkpoint["rng"]) == {"python", "numpy", "torch", "cuda"} and checkpoint["rng"]["cuda"] is None

    summary = (tmp_path / "verification_summary.md").read_text(encoding="utf-8")
    assert summary == render_verification_summary(report)
    assert "| B | NOT_RUN |" in summary and "| A | NOT_RUN |" in summary
    assert "| Queue pointer / 実key数 / random残数 | PASS | 観測 640 / 640 / 384、期待 640 / 640 / 384 |" in summary
    assert "a419ef282fc9885015674e77f58c59447206a27f" in summary and "公式MoCo v2との差分" in summary
    assert "GaussianBlurはkernel 3" in summary and "表現品質" in summary
    assert all(f"[`{name}`]({name})" in summary for name in ARTIFACT_NAMES)
    assert FAIL not in summary


def test_queue_random_keys_are_replaced_after_k_over_b_steps(tmp_path):
    report = run_smoke(tmp_path, **SMALL, **{"training.max_steps": 6})
    history = report["history"]
    assert [item["random_remaining_count"] for item in history] == [24, 16, 8, 0, 0, 0]
    assert [item["real_key_count"] for item in history] == [8, 16, 24, 32, 32, 32]
    assert [item["queue_ptr"] for item in history] == [8, 16, 24, 0, 8, 16]
    assert {check["status"] for check in report["checks"].values()} == {PASS}


def test_cosine_schedule_uses_completed_optimizer_steps(tmp_path):
    report = run_smoke(tmp_path, **SMALL, **{"training.max_steps": 4, "training.scheduler": "cosine"})
    expected = [0.00375 * 0.5 * (1 + math.cos(math.pi * t / 4)) for t in range(4)]
    assert [item["lr"] for item in report["history"]] == pytest.approx(expected, rel=1e-12)
    checkpoint = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["scheduler"]["last_epoch"] == 4
    assert report["verification"]["checkpoint_roundtrip"]["scheduler_equal"] is True


def test_run_kind_requires_fresh_runs_with_approved_settings_on_full_cifar10():
    assert smoke.run_kind(load_cfg(), 50_000) == "gate_b_smoke20"
    assert smoke.run_kind(load_cfg(), None) == "gate_b_smoke20"
    pilot = {"training.max_steps": 500, "training.scheduler": "cosine"}
    assert smoke.run_kind(load_cfg(**pilot), 50_000) == "gate_c_pilot500"
    assert smoke.run_kind(load_cfg(**{"training.max_steps": 500}), 50_000) == "custom"  # 500 steps, constant LR
    assert smoke.run_kind(load_cfg(), 640) == "custom"
    assert smoke.run_kind(load_cfg(**{"training.batch_size": 16}), 50_000) == "custom"
    assert smoke.run_kind(load_cfg(**{"training.resume": "checkpoint.pt"}), 50_000) == "custom"


def test_gate_d_resume_reproduces_the_uninterrupted_run(tmp_path):
    straight = run_smoke(tmp_path / "straight", **SMALL, **{"training.max_steps": 6})
    first = run_smoke(tmp_path / "first", **SMALL, **{"training.max_steps": 3})
    again = run_smoke(tmp_path / "again", **SMALL, **{"training.max_steps": 3})
    resumed = run_smoke(tmp_path / "resumed", **SMALL, **{
        "training.max_steps": 6, "training.resume": str(tmp_path / "first" / "checkpoint.pt")})

    # Same seed and CPU conditions: identical loss, EMA error and queue.
    assert first["history"] == again["history"] == straight["history"][:3]

    def load(name):
        return torch.load(tmp_path / name / "checkpoint.pt", map_location="cpu", weights_only=True)

    assert smoke._nested_equal(load("first")["model"], load("again")["model"])

    assert resumed["resume"] == {"resumed_from": str(tmp_path / "first" / "checkpoint.pt"), "start_step": 3}
    assert resumed["run_kind"] == "custom"
    assert resumed["history"] == straight["history"]
    for part in ("model", "optimizer", "sampler"):
        assert smoke._nested_equal(load("resumed")[part], load("straight")[part]), part
    assert torch.equal(load("resumed")["rng"]["torch"], load("straight")["rng"]["torch"])
    assert {check["status"] for check in resumed["checks"].values()} == {PASS}


def test_gate_d_checkpoint_reload_gives_identical_state_eval_and_next_step(tmp_path):
    cfg = load_cfg(**SMALL)
    device = torch.device("cpu")
    generator = torch.Generator().manual_seed(5)
    batches = [tuple(torch.randn(8, 3, 32, 32, generator=generator) for _ in range(2)) for _ in range(3)]
    torch.manual_seed(0)
    model, optimizer, scheduler = smoke.build_training(cfg, device)
    for im_q, im_k in batches[:2]:
        smoke.train_step(model, optimizer, im_q, im_k)
    path = tmp_path / "checkpoint.pt"
    sampler = {"order_sha256": "synthetic", "batch_size": 8, "cursor": 16}
    smoke.save_checkpoint(path, model, optimizer, scheduler, global_step=2, history=[], sampler=sampler,
                          config=OmegaConf.to_container(cfg), device=device, status="completed", resumable=True)

    torch.manual_seed(123)  # a differently initialized instance
    fresh, fresh_optimizer, fresh_scheduler = smoke.build_training(cfg, device)
    assert not smoke._nested_equal(model.state_dict(), fresh.state_dict())
    smoke.restore_training_state(smoke.load_checkpoint(path), fresh, fresh_optimizer, fresh_scheduler)
    assert smoke._nested_equal(model.state_dict(), fresh.state_dict())
    assert smoke._nested_equal(optimizer.state_dict(), fresh_optimizer.state_dict())
    assert int(fresh.queue_ptr) == 16 and int(fresh.real_key_count) == 16

    model.eval()
    fresh.eval()
    probe = batches[2][0]
    with torch.no_grad():
        torch.testing.assert_close(fresh.encode_query(probe), model.encode_query(probe), rtol=0, atol=0)
        torch.testing.assert_close(fresh.encode_key(probe), model.encode_key(probe), rtol=0, atol=0)
    model.train()
    fresh.train()
    out, record = smoke.train_step(model, optimizer, *batches[2])
    out_fresh, record_fresh = smoke.train_step(fresh, fresh_optimizer, *batches[2])
    assert record == record_fresh and out.loss.item() == out_fresh.loss.item()
    assert smoke._nested_equal(model.state_dict(), fresh.state_dict())
    assert smoke._nested_equal(optimizer.state_dict(), fresh_optimizer.state_dict())


def test_non_finite_input_fails_fast_and_keeps_a_measured_failure_record(tmp_path):
    bad_index = int(smoke.single_epoch_order(FakeCIFAR10.size, 0)[2 * 8])  # first sample of step 3

    class NaNAtStep3(FakeCIFAR10):
        def __getitem__(self, index):
            (im_q, im_k), label = super().__getitem__(index)
            if index == bad_index:
                im_q = torch.full_like(im_q, float("nan"))
            return [im_q, im_k], label

    with pytest.raises(FloatingPointError, match="non-finite im_q"):
        run_smoke(tmp_path, dataset=NaNAtStep3, **SMALL, **{"training.max_steps": 5})

    assert {path.name for path in tmp_path.iterdir()} == ARTIFACT_NAMES
    report = read_report(tmp_path)
    assert report["status"] == "failed"
    failure = report["failure"]
    assert (failure["stage"], failure["step"], failure["type"]) == ("train", 3, "FloatingPointError")
    assert "non-finite im_q" in failure["message"]
    assert [item["global_step"] for item in report["history"]] == [1, 2]
    checks = report["checks"]
    assert checks["optimizer_steps_completed"]["status"] == FAIL
    assert checks["checkpoint_roundtrip"]["status"] == NOT_RUN
    assert checks["queue_pointer_and_counts"]["status"] == PASS  # the bad step was rejected before enqueue
    assert checks["ema_within_tolerance"]["status"] == PASS and checks["key_no_gradient"]["status"] == PASS
    summary = (tmp_path / "verification_summary.md").read_text(encoding="utf-8")
    assert "run status: **failed**" in summary and "step 3（train）で `FloatingPointError`" in summary

    checkpoint = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["status"] == "failed" and checkpoint["resumable"] is False and checkpoint["global_step"] == 2
    with pytest.raises(ValueError, match="cannot be resumed"):
        run_smoke(tmp_path / "resume", **SMALL,
                  **{"training.max_steps": 5, "training.resume": str(tmp_path / "checkpoint.pt")})


def test_missing_dataset_fails_without_downloading_and_is_not_a_gate_pass(tmp_path):
    root = tmp_path / "absent"
    with pytest.raises(RuntimeError, match="Dataset not found"):
        run_smoke(tmp_path / "run", dataset=None, **{"data.root": str(root)})
    report = read_report(tmp_path / "run")
    assert report["status"] == "failed" and report["failure"]["stage"] == "data"
    assert report["run_kind"] == "gate_b_smoke20" and report["gates"]["B"]["status"] == FAIL
    assert report["history"] == [] and report["artifacts"]["checkpoint.pt"] is False
    assert not (tmp_path / "run" / "checkpoint.pt").exists()
    assert not root.exists() or not any(root.iterdir())
    summary = (tmp_path / "run" / "verification_summary.md").read_text(encoding="utf-8")
    assert "run status: **failed**" in summary and "| B | FAIL |" in summary


def test_cuda_is_used_only_when_requested_and_fails_explicitly_when_unavailable(tmp_path):
    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA is unavailable"):
            run_smoke(tmp_path, **{"training.device": "cuda"})
    assert read_report(tmp_path)["failure"]["stage"] == "setup"
    with pytest.raises(ValueError, match="cpu or cuda"):
        smoke.resolve_device("auto")


def test_existing_artifacts_are_never_overwritten(tmp_path):
    (tmp_path / "report.json").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_smoke(tmp_path)
    assert (tmp_path / "report.json").read_text(encoding="utf-8") == "keep"
    assert {path.name for path in tmp_path.iterdir()} == {"report.json"}


@pytest.mark.parametrize("overrides, match", [
    ({"logging.comet_enabled": True}, "Comet"),
    ({"moco.queue_size": 30, "training.batch_size": 8}, "multiple"),
    ({"training.scheduler": "step"}, "none or cosine"),
    ({**SMALL, "training.max_steps": 81}, "exceed one epoch"),
])
def test_invalid_configs_fail_with_a_reason(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        run_smoke(tmp_path, **overrides)
    assert read_report(tmp_path)["status"] == "failed"


def test_resume_rejects_a_changed_config(tmp_path):
    run_smoke(tmp_path / "first", **SMALL, **{"training.max_steps": 2})
    resume = {"training.max_steps": 4, "training.resume": str(tmp_path / "first" / "checkpoint.pt")}
    with pytest.raises(ValueError, match="config differs"):
        run_smoke(tmp_path / "changed", **SMALL, **resume, **{"moco.temperature": 0.1})


def test_long_pilot_report_renders_sampled_rows_queue_marker_and_descriptive_diagnostics(tmp_path):
    base = run_smoke(tmp_path / "base", **SMALL, **{"training.max_steps": 2})
    report = copy.deepcopy(base)
    template = report["history"][-1]
    report["history"] = [{**template, "global_step": step, "sample_count": 32 * step,
                          "query_backbone_grad_norm": 10.0 if step == 250 else 1.0} for step in range(1, 501)]
    report["moco"]["queue_size"], report["optimizer"]["batch_size"] = 1024, 32
    report["optimizer"]["max_steps"] = report["optimizer"]["completed_steps"] = 500
    report["run_kind"] = "gate_c_pilot500"
    report["diagnostics"] = build_diagnostics(report["history"])
    report["checks"] = evaluate_checks(report)

    assert report["diagnostics"]["query_backbone_grad_norm"]["max_step"] == 250
    assert report["diagnostics"]["query_backbone_grad_norm"]["max_over_median"] == 10.0
    assert gate_statuses(report)["C"]["status"] == PASS and gate_statuses(report)["B"]["status"] == NOT_RUN
    output = tmp_path / "long"
    output.mkdir()
    write_verification_artifacts(output, report)
    summary = (output / "verification_summary.md").read_text(encoding="utf-8")
    assert "表は抜粋（全500 stepは `report.json` の `history`）" in summary
    for step in (1, 31, 32, 33, 50, 500):
        assert f"\n| {step} | " in summary
    assert "\n| 34 | " not in summary
    assert "| C | PASS |" in summary and "自動判定なし" in summary
    assert matplotlib.image.imread(output / "loss.png").ndim == 3
