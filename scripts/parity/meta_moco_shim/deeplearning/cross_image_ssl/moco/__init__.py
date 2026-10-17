"""Alias the clone's top-level ``moco.builder`` / ``moco.loader`` under the internal package path."""

import importlib
import sys

builder = importlib.import_module("moco.builder")
loader = importlib.import_module("moco.loader")
sys.modules[f"{__name__}.builder"] = builder
sys.modules[f"{__name__}.loader"] = loader
