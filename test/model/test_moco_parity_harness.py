"""Spec A harness on CPU (gloo, implementation=self): config, state mapping, fixture, phase traces, within-run checks,
checkpoint resume and comparator verdicts. The reference run itself needs CUDA (its builder calls .cuda()) and is
exercised by the GPU gates, not here; the reference clone is used on CPU only to build the fixture.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.parity import compare_moco_parity as comparator
from scripts.parity import moco_imagenet_parity as runner
from scripts.parity.moco_parity_state import (
    canonical_to_self, compare_tensors, expected_shuffle, mapping_table, self_to_canonical, shuffle_seed,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "conf" / "moco_imagenet_parity.yaml"
CLONE = Path(OmegaConf.load(CONFIG).reference.root)
needs_clone = pytest.mark.skipif(not (CLONE / "moco" / "builder.py").exists(), reason="reference clone not present")
SMALL = {"moco.queue_size": 8, "training.local_batch_size": 2, "data.image_size": 32}
CPU = {"implementation": "self", "training.device": "cpu", "distributed.backend": "gloo",
       "logging.comet_enabled": False, **SMALL}


def load_cfg(**overrides):
    cfg = OmegaConf.load(CONFIG)
    cfg.pop("hydra")
    for key, value in {**CPU, **overrides}.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def run_parity(output_dir: Path, **overrides) -> dict:
    hydra_config = SimpleNamespace(get=lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=str(output_dir))))
    with patch.object(runner, "HydraConfig", hydra_config):
        return runner.run(load_cfg(**overrides))


def test_hydra_composes_the_spec_a_defaults_without_touching_other_configs():
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        cfg = compose(config_name="moco_imagenet_parity")
    assert (cfg.model.arch, cfg.model.projection_dim, cfg.model.mlp) == ("resnet50", 128, True)
    assert (cfg.moco.queue_size, cfg.moco.momentum, cfg.moco.temperature) == (65536, 0.999, 0.2)
    assert (cfg.distributed.world_size, cfg.training.local_batch_size) == (2, 32)
    assert (cfg.training.learning_rate, cfg.training.momentum, cfg.training.weight_decay) == (0.0075, 0.9, 0.0001)
    assert (cfg.precision.dtype, cfg.precision.amp, cfg.precision.fp32_precision) == ("float32", False, "ieee")
    assert (cfg.verification.rtol, cfg.verification.atol) == (1e-5, 1e-6)
    assert cfg.data.root == "/mnt/NAS-TVS872XT/dataset/ImageNet" and cfg.data.image_size == 224
    assert cfg.reference.commit == "a419ef282fc9885015674e77f58c59447206a27f"
    for name in ("config.yaml", "moco_standalone.yaml"):
        assert "moco_imagenet_parity" not in (ROOT / "conf" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("overrides, match", [
    ({"implementation": "meta_reference"}, "only for gloo parity tests"),
    ({"moco.queue_size": 10}, "queue_size %"),
    ({"precision.amp": True}, "FP32 without AMP"),
    ({"distributed.broadcast_buffers": False}, "broadcast_buffers"),
    ({"fixture.path": None}, "needs fixture.path"),
    ({"training.checkpoint_steps": [9]}, "checkpoint_steps"),
])
def test_invalid_configs_fail_with_a_reason(overrides, match):
    with pytest.raises(ValueError, match=match):
        runner.validate_config(load_cfg(**{"fixture.path": "fixture.pt", **overrides}))


def test_state_names_map_explicitly_and_mismatches_raise():
    assert self_to_canonical("backbone_q.net.layer1.0.conv1.weight") == "encoder_q.layer1.0.conv1.weight"
    assert self_to_canonical("projector_k.2.bias") == "encoder_k.fc.2.bias"
    assert self_to_canonical("real_key_count") is None and self_to_canonical("queue_ptr") == "queue_ptr"
    assert canonical_to_self("encoder_q.fc.0.weight") == "projector_q.0.weight"
    assert canonical_to_self("encoder_k.bn1.running_var") == "backbone_k.net.bn1.running_var"
    self_state = {"backbone_q.net.conv1.weight": torch.zeros(2, 3), "projector_q.0.weight": torch.zeros(4, 2),
                  "queue": torch.zeros(4, 8), "real_key_count": torch.zeros(1, dtype=torch.long)}
    canonical = {"encoder_q.conv1.weight": torch.ones(2, 3), "encoder_q.fc.0.weight": torch.ones(4, 2),
                 "queue": torch.ones(4, 8)}
    assert [row["self"] for row in mapping_table(canonical, self_state)] == \
        ["backbone_q.net.conv1.weight", "projector_q.0.weight", "queue"]
    with pytest.raises(KeyError, match="reference-only"):
        mapping_table({**canonical, "encoder_q.bn1.weight": torch.ones(2)}, self_state)
    with pytest.raises(ValueError, match="encoder_q.conv1.weight"):
        mapping_table({**canonical, "encoder_q.conv1.weight": torch.ones(3, 2)}, self_state)
    with pytest.raises(KeyError):
        self_to_canonical("encoder_q.conv1.weight")


def test_compare_tensors_follows_assert_close_semantics():
    reference = torch.tensor([1.0, 0.0, 1e-9, 100.0])
    assert compare_tensors(reference, reference.clone(), 1e-5, 1e-6)["bitwise_equal"]
    near = compare_tensors(reference, reference + torch.tensor([5e-6, 9e-7, 0.0, 1e-3]), 1e-5, 1e-6)
    assert near["within_tolerance"] and not near["bitwise_equal"] and near["small_denominator_count"] == 2
    far = compare_tensors(reference, reference + torch.tensor([0.0, 2e-6, 0.0, 0.0]), 1e-5, 1e-6)
    assert not far["within_tolerance"] and far["mismatch_count"] == 1
    nan = compare_tensors(reference, torch.tensor([float("nan"), 0.0, 1e-9, 100.0]), 1e-5, 1e-6)
    assert not nan["within_tolerance"] and nan["nonfinite_candidate"] == 1
    ints = compare_tensors(torch.tensor([3, 1]), torch.tensor([3, 2]), 1.0, 1.0)
    assert ints["kind"] == "exact" and ints["mismatch_count"] == 1 and not ints["within_tolerance"]
    assert not compare_tensors(reference, reference.double(), 1e-5, 1e-6)["within_tolerance"]  # dtype differs


def test_shuffle_seed_reproduces_the_rank0_randperm_of_the_global_rng():
    torch.default_generator.manual_seed(shuffle_seed(0, 3, 0))
    assert torch.equal(torch.randperm(64), expected_shuffle(0, 3, 64))
    assert shuffle_seed(0, 3, 0) != shuffle_seed(0, 3, 1) != shuffle_seed(0, 4, 1)


def test_self_aug_plus_matches_the_reference_pipeline_definition():
    augmentation = runner.moco_v2_imagenet_augmentation(224)
    names = [type(t).__name__ for t in augmentation.transforms]
    assert names == ["RandomResizedCrop", "RandomApply", "RandomGrayscale", "RandomApply", "RandomHorizontalFlip",
                     "ToTensor", "Normalize"]
    crop, jitter, gray, blur = augmentation.transforms[:4]
    assert crop.size == (224, 224) and crop.scale == (0.2, 1.0)
    assert jitter.p == 0.8 and gray.p == 0.2 and blur.p == 0.5
    assert isinstance(blur.transforms[0], runner.PILGaussianBlur) and blur.transforms[0].sigma == (0.1, 2.0)
    assert runner.epoch_cosine_lr(0.0075, 0, 10) == 0.0075


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not (CLONE / "moco" / "builder.py").exists():
        pytest.skip("reference clone not present")
    base = tmp_path_factory.mktemp("parity")
    built = subprocess.run([sys.executable, "-m", "scripts.parity.build_moco_parity_fixture",
                            *[f"{k}={v}" for k, v in SMALL.items()], "--output-dir", str(base / "fixture")],
                           cwd=ROOT, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr[-3000:]
    fixture = str(base / "fixture" / "fixture.pt")
    common = {"fixture.path": fixture, "training.max_steps": 5}
    reports = {
        "A": run_parity(base / "A", **common, **{"trace.full_steps": [4], "training.checkpoint_steps": [3]}),
        "B": run_parity(base / "B", **common, **{"trace.full_steps": []}),
        "C": run_parity(base / "C", **common, **{"trace.full_steps": [4],
                                                 "training.resume": str(base / "A" / "checkpoint_step000003.pt")}),
    }
    return base, reports


@needs_clone
def test_fixture_holds_the_reference_initial_state_and_input_definition(runs):
    base, _ = runs
    manifest = json.loads((base / "fixture" / "fixture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "synthetic" and manifest["reference"]["commit"] == runner.REFERENCE["commit"]
    assert manifest["reference"]["clone"]["dirty"] is False
    assert manifest["state_tensors"] == len(manifest["state_tensor_sha256"]) == 646
    names = manifest["state_tensor_sha256"]
    assert {"queue", "queue_ptr", "encoder_q.fc.0.weight", "encoder_k.fc.2.bias"} <= set(names)
    assert all(names[n] == names[n.replace("encoder_q.", "encoder_k.", 1)] for n in names if n.startswith("encoder_q."))


@needs_clone
def test_self_run_passes_every_within_run_check_and_writes_all_artifacts(runs):
    base, reports = runs
    report = reports["A"]
    assert report["status"] == "completed" and report["completed_steps"] == 5
    assert {name: result["status"] for name, result in report["verification"].items()
            if name in runner.WITHIN_RUN_CHECKS} == {name: "PASS" for name in runner.WITHIN_RUN_CHECKS}
    for name in (*runner.RUN_ARTIFACTS, "checkpoint_manifest.json", "checkpoint_step000003.pt",
                 "checkpoint_step000003.rank0.sidecar.pt", "checkpoint_step000003.rank1.sidecar.pt"):
        assert (base / "A" / name).exists(), name
    assert sorted(p.name for p in (base / "A" / "trace" / "rank1").iterdir()) == \
        [f"step_{s:06d}.pt" for s in range(1, 6)]
    # K=8, global batch 4: the pointer wraps every 2 steps and stays identical across ranks.
    for rank in report["ranks"]:
        assert [h["metrics"]["queue_ptr"] for h in rank["history"]] == [4, 0, 4, 0, 4]
    info = report["verification"]["key_bn_buffers_differing_from_rank0_after_key_forward"]["per_rank_step"]
    assert all(value > 0 for value in info[1]) and set(info[0]) == {0}  # local shuffled minibatches differ
    manifest = json.loads((base / "A" / "fixture_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["mapping_table"]) == 646
    assert [s["input_sha256"] for s in manifest["per_rank"][0]["steps"]] != \
        [s["input_sha256"] for s in manifest["per_rank"][1]["steps"]]
    trace = torch.load(base / "A" / "trace" / "rank0" / "step_000004.pt", weights_only=True)
    assert set(trace["phases"]) == {"before_forward", "after_buffer_broadcast", "after_query", "after_key_ema",
                                    "after_key_forward", "after_logits_before_enqueue", "after_enqueue",
                                    "after_backward", "after_sgd"}
    assert trace["phases"]["after_sgd"]["full"]["momentum"].keys() == \
        trace["phases"]["after_sgd"]["full"]["query_params"].keys()
    provenance = json.loads((base / "A" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["reference_clone"]["diff_empty"] and provenance["backend_flags"][0]["fp32_precision"] == "ieee"


@needs_clone
def test_rerun_with_the_same_fixture_is_identical(runs):
    base, _ = runs
    report = comparator.compare(base / "A", base / "B", "rerun", 1e-5, 1e-6, "fixed")
    assert report["status"] == "PASS", report["reason"]
    assert report["steps_compared"] == [1, 2, 3, 4, 5] and report["first_failure"] is None
    assert report["counts"]["bitwise_equal"] == report["counts"]["judged"] > 0
    provisional = comparator.compare(base / "A", base / "B", "rerun", 1e-5, 1e-6, "provisional")
    assert provisional["status"] == "BLOCKED" and "provisional" in provisional["reason"]
    mislabeled = comparator.compare(base / "A", base / "B", "implementation", 1e-5, 1e-6, "fixed")
    assert mislabeled["status"] == "BLOCKED" and "meta_reference vs self" in mislabeled["reason"]


@needs_clone
def test_resume_from_the_checkpoint_reproduces_the_next_steps(runs):
    base, reports = runs
    resumed = reports["C"]
    assert resumed["status"] == "completed" and [h["global_step"] for h in resumed["ranks"][0]["history"]] == [4, 5]
    report = comparator.compare(base / "A", base / "C", "resume", 1e-5, 1e-6, "fixed")
    assert report["status"] == "PASS", report["reason"]
    assert report["steps_compared"] == [4, 5] and report["full_tensor_steps"] == [4]
    groups = {key.split("/")[1] for key in report["summary_by_phase_group"]}
    assert {"query_params", "key_params", "query_bn", "key_bn", "queue", "query_grads", "momentum", "logits"} <= groups
    # Rank-1 BN buffers before the first resumed forward come from the rank-0 checkpoint: informational only.
    assert {(e["rank"], e["phase"]) for e in report["informational"]} == {(1, "before_forward")}


@needs_clone
def test_comparator_locates_the_first_perturbed_tensor(runs, tmp_path):
    base, _ = runs
    shutil.copytree(base / "B", tmp_path / "D")
    path = tmp_path / "D" / "trace" / "rank1" / "step_000002.pt"
    trace = torch.load(path, weights_only=True)
    trace["small"]["z_q"][0, 0] += 1e-3
    torch.save(trace, path)
    report = comparator.compare(base / "A", tmp_path / "D", "rerun", 1e-5, 1e-6, "fixed")
    assert report["status"] == "FAIL"
    first = report["first_failure"]
    assert (first["step"], first["rank"], first["phase"], first["tensor"]) == (2, 1, "after_query", "z_q")
    assert first["mismatch_count"] == 1
    summary = comparator.render_summary(report)
    assert "**FAIL**" in summary and "`z_q`" in summary


@needs_clone
def test_existing_run_directory_is_never_overwritten(runs):
    base, _ = runs
    with pytest.raises(FileExistsError):
        run_parity(base / "A", **{"fixture.path": str(base / "fixture" / "fixture.pt")})


def test_comet_logging_calls_work_with_a_disabled_experiment(monkeypatch):
    """API compatibility of the rank-0 Comet calls without sending anything (Experiment(disabled=True))."""
    import comet_ml

    experiment_class, created = comet_ml.Experiment, []

    def disabled_experiment(**kwargs):
        assert kwargs["auto_metric_logging"] is False and kwargs["auto_param_logging"] is False
        created.append(experiment_class(disabled=True, **kwargs))
        return created[-1]

    monkeypatch.setattr(comet_ml, "Experiment", disabled_experiment)
    cfg = load_cfg(**{"logging.comet_enabled": True})
    assert runner.CometRun(cfg, rank=1, params={}).experiment is None  # rank 0 only
    comet = runner.CometRun(cfg, rank=0, params=runner.comet_parameters(cfg, world_size=2))
    assert created and comet.info()["key"]
    comet.log_metrics({"loss": 1.0, "queue_ptr": 64, "note": "skipped"}, step=1)
    comet.end("completed")
