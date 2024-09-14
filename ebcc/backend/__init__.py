"""Backend for NumPy operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import importlib

if TYPE_CHECKING:
    from types import ModuleType


def __getattr__(name: str) -> ModuleType:
    """Get the backend module."""
    if name.lower() == "numpy":
        return importlib.import_module("ebcc.backend._numpy")
    else:
        raise ValueError(f"Unknown backend: {name}")
