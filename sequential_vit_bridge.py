"""Compatibility import for :mod:`integration.sequential_vit`."""

import sys

from integration import sequential_vit as _implementation

sys.modules[__name__] = _implementation
