import logging
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image
from huggingface_hub import try_to_load_from_cache
from transformers import AutoImageProcessor

from scripts.smoke import smoke_vit_lora_cifar10 as smoke


REPO_ROOT = Path(__file__).resolve().parents[2]
# One ViT layer with the ViT-B feature size and 224x224 / 16x16 patches.
VIT_224 = dict(hidden_size=768, num_attention_heads=12, num_hidden_layers=1, image_size=224, patch_size=16)


def compose_lora_config(overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "conf")):
        return compose(config_name="lora_standalone", overrides=list(overrides))


class FakeCIFAR10:
    instances = []

    def __init__(self, root, train, download, transform):
        self.root, self.train, self.download, self.transform = root, train, download, transform
        generator = np.random.default_rng(0)
        self.images = [Image.fromarray(generator.integers(0, 256, (32, 32, 3), dtype=np.uint8)) for _ in range(20)]
        self.targets = [index % 10 for index in range(20)]
        FakeCIFAR10.instances.append(self)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]


class FakeProcessor:
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        pixels = torch.from_numpy(np.array(images)).permute(2, 0, 1).float()[None] / 255
        pixels = F.interpolate(pixels, size=(224, 224), mode="bilinear")
        return {"pixel_values": (pixels - 0.5) / 0.5}


def test_preset_matches_approved_spec():
    cfg = compose_lora_config()
    assert OmegaConf.to_container(cfg) == {
        "model": {"checkpoint_id": "google/vit-base-patch16-224"},
        "lora": {"enabled": True, "target_modules": ["q", "v"], "rank": 8, "alpha": 8, "dropout": 0.0},
        "data": {"name": "CIFAR10", "root": "./downloaded_data", "download": False, "train_subset_size": 64},
        "training": {"seed": 0, "batch_size": 16, "max_steps": 20, "optimizer": "adamw",
                     "learning_rate": 0.001, "weight_decay": 0.0},
        "logging": {"comet_enabled": False},
    }
    assert compose_lora_config(["lora.target_modules=[k]"]).lora.target_modules == ["k"]
    assert compose_lora_config(["lora.target_modules=[q,k,v]", "lora.rank=4", "lora.alpha=16"]).lora.rank == 4


def test_checkpoint_preprocess_uses_vit_processor_settings():
    checkpoint_id = "google/vit-base-patch16-224"
    if not isinstance(try_to_load_from_cache(checkpoint_id, "preprocessor_config.json"), str):
        pytest.skip(f"{checkpoint_id} processor is not in the local Hugging Face cache")
    processor = AutoImageProcessor.from_pretrained(checkpoint_id)
    assert tuple(processor.image_mean) == (0.5, 0.5, 0.5) and tuple(processor.image_std) == (0.5, 0.5, 0.5)
    image = Image.fromarray(np.full((32, 32, 3), [255, 0, 128], dtype=np.uint8))
    pixels = smoke.checkpoint_preprocess(processor)(image)
    assert pixels.shape == (3, 224, 224) and pixels.dtype == torch.float32
    torch.testing.assert_close(pixels[:, 100, 100], torch.tensor([1.0, -1.0, 128 / 255 * 2 - 1]))


def test_cli_prints_config_without_training(tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.smoke.smoke_vit_lora_cifar10",
         "lora.target_modules=[k,v]", "--cfg", "job", "--resolve"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    cfg = OmegaConf.create(result.stdout)
    assert cfg.lora.target_modules == ["k", "v"]
    assert cfg.data.download is False
    assert not (tmp_path / "log").exists()
    assert not (tmp_path / "downloaded_data").exists()


@pytest.fixture
def fake_run(monkeypatch, fake_pretrained_vit, tmp_path):
    FakeCIFAR10.instances.clear()
    monkeypatch.setattr(smoke, "CIFAR10", FakeCIFAR10)
    monkeypatch.setattr(smoke.AutoImageProcessor, "from_pretrained", lambda checkpoint_id: FakeProcessor())
    monkeypatch.setattr(smoke, "git_output", lambda repository, *args: "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    source, calls = fake_pretrained_vit(**VIT_224)

    def run(*overrides):
        cfg = compose_lora_config(["data.train_subset_size=8", "training.batch_size=4", "training.max_steps=3",
                                   *overrides])
        return smoke.run(cfg, tmp_path)
    return run, calls


def test_stage_b_trains_lora_and_head_saves_and_reloads(fake_run, caplog, tmp_path):
    run, calls = fake_run
    with caplog.at_level(logging.INFO):
        summary = run()

    dataset = FakeCIFAR10.instances[0]
    assert dataset.train is True and dataset.download is False
    assert len(calls) == 2  # trained model + rebuilt model for the reload check
    assert summary["steps"] == 3 and len(summary["history"]) == 3
    assert len(summary["subset_indices"]) == 8
    assert summary["base_changed"] == 0 and summary["base_update_norm"] == 0
    assert summary["changed"]["lora_b"] > 0 and summary["changed"]["head"] == 2
    assert summary["reload_max_abs_diff"] < 1e-5
    first = summary["history"][0]
    assert first["grad_norm/lora_a"] == 0 and first["update_norm/lora_a"] == 0  # B == 0 at init
    assert first["grad_norm/lora_b"] > 0 and first["update_norm/lora_b"] > 0
    assert any(record["grad_norm/lora_a"] > 0 for record in summary["history"][1:])

    saved = torch.load(tmp_path / smoke.CHECKPOINT_FILENAME, map_location="cpu", weights_only=True)
    assert saved["lora"] == {"target_modules": ["q", "v"], "rank": 8, "alpha": 8, "dropout": 0.0}
    assert saved["checkpoint_id"] == "google/vit-base-patch16-224"
    assert saved["seed"] == 0 and saved["steps"] == 3
    assert saved["data"]["subset_indices"] == summary["subset_indices"]
    assert len(saved["adapter"]) == 4 and all(".lora_" in name for name in saved["adapter"])
    assert set(saved["head"]) == {"weight", "bias"}

    log = caplog.text
    assert "target modules: ['q', 'v'] -> 2 wrapped projections" in log
    assert "LoRA A/B + classifier head" in log
    assert "Stage B checks passed" in log


def test_stage_b_is_reproducible_with_same_seed(fake_run):
    run, _ = fake_run
    first = run("lora.target_modules=[k]")
    second = run("lora.target_modules=[k]")
    assert first["subset_indices"] == second["subset_indices"]
    assert [record["loss"] for record in first["history"]] == [record["loss"] for record in second["history"]]


@pytest.mark.parametrize("override,match", [
    ("lora.enabled=false", "lora.enabled=false"),
    ("training.optimizer=sgd", "adamw"),
    ("data.name=ImageFolder", "CIFAR10"),
])
def test_unsupported_settings_fail_before_loading_data(fake_run, override, match):
    run, calls = fake_run
    with pytest.raises(ValueError, match=match):
        run(override)
    assert FakeCIFAR10.instances == [] and calls == []
