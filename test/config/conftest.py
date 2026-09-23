from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def compose_config():
    def make(overrides=(), *, return_hydra_config=False):
        with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "conf")):
            return compose(
                config_name="config",
                overrides=list(overrides),
                return_hydra_config=return_hydra_config,
            )
    return make
