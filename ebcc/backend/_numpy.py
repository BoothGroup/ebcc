"""NumPy backend."""

from __future__ import annotations

import numpy


def __getattr__(name: str) -> object:
    """Get the attribute from NumPy."""
    return getattr(numpy, name)
