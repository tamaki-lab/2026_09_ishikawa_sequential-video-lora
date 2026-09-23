import os
from pathlib import Path
import subprocess
import sys

import pytest
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_python(args, cwd):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize("entrypoint", ["main.py", "main_pl.py"])
@pytest.mark.parametrize("overrides", [[], [
    "dataset=imagefolder", "model=vit_b", "optimizer=adam",
    "loader.batch_size=3", "model.use_pretrained=false", "scheduler.enabled=true",
]])
def test_config_print_without_training(entrypoint, overrides, tmp_path):
    result = run_python([str(REPO_ROOT / entrypoint), *overrides, "--cfg", "job", "--resolve"], tmp_path)
    assert result.returncode == 0, result.stderr
    cfg = OmegaConf.create(result.stdout)
    assert cfg.dataset.name == ("ImageFolder" if overrides else "CIFAR10")
    assert cfg.model.name == ("vit_b" if overrides else "resnet18")
    assert cfg.optimizer.name == ("Adam" if overrides else "SGD")
    assert cfg.loader.batch_size == (3 if overrides else 8)
    assert cfg.model.use_pretrained is (not bool(overrides))
    assert cfg.scheduler.enabled is bool(overrides)
    assert not (tmp_path / "log").exists()


@pytest.mark.parametrize("entrypoint", ["main.py", "main_pl.py"])
def test_old_training_cli_is_rejected(entrypoint, tmp_path):
    result = run_python([str(REPO_ROOT / entrypoint), "-b", "8", "--cfg", "job"], tmp_path)
    assert result.returncode != 0
    assert "unrecognized arguments: -b" in result.stderr


@pytest.mark.parametrize("entrypoint", ["main.py", "main_pl.py"])
def test_runtime_artifacts_and_resolved_config(entrypoint, tmp_path):
    # Stop at the first training dependency, after the real Hydra entrypoint
    # logs its config. No dataset, model, logger, GPU, or training is started.
    script = '''
import os
import runpy
import sys
from unittest.mock import patch

def stop_before_training(*args, **kwargs):
    print("RUNTIME_CWD=" + os.getcwd())
    raise SystemExit(0)

entrypoint = sys.argv[1]
sys.argv = sys.argv[1:]
with patch("logger.configure_logger", side_effect=stop_before_training), \\
     patch("torch.cuda.is_available", side_effect=stop_before_training):
    runpy.run_path(entrypoint, run_name="__main__")
'''
    result = run_python([
        "-c", script, str(REPO_ROOT / entrypoint),
        "dataset.root=relative-data", "checkpoint.save_dir=${dataset.root}/checkpoints",
    ], tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"RUNTIME_CWD={tmp_path}" in result.stdout
    artifacts = list((tmp_path / "log" / "hydra").glob("*/*/.hydra/config.yaml"))
    assert len(artifacts) == 1
    cfg = OmegaConf.load(artifacts[0])
    assert cfg.dataset.root == "relative-data"
    assert cfg.checkpoint.save_dir == "relative-data/checkpoints"
    assert (artifacts[0].parent / "hydra.yaml").is_file()
    assert (artifacts[0].parent / "overrides.yaml").is_file()
    log_path = artifacts[0].parent.parent / (Path(entrypoint).stem + ".log")
    log = log_path.read_text()
    assert "Resolved config:" in log
    assert "save_dir: relative-data/checkpoints" in log
    assert "${dataset.root}" not in log
