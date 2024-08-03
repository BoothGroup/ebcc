"""Base classes."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class ERIs(ABC):
    """Base class for electronic repulsion integrals."""

    def __getitem__(self, key: str) -> Any:
        """Get an item."""
        return self.__dict__[key]


class Fock(ABC):
    """Base class for Fock matrices."""

    def __getitem__(self, key: str) -> Any:
        """Get an item."""
        return self.__dict__[key]
