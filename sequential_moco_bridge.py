"""Compatibility import for :mod:`integration.sequential_moco`."""

import sys

from integration import sequential_moco as _implementation

sys.modules[__name__] = _implementation
