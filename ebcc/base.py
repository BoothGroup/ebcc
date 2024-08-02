"""Base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ebcc import util

if TYPE_CHECKING:
    from dataclasses import dataclass
    from logging import Logger
    from typing import Any, Optional, Union

    from pyscf.scf import SCF

    from ebcc.ansatz import Ansatz
    from ebcc.space import Space


class EOM(ABC):
    """Base class for equation-of-motion methods."""

    pass


class BruecknerEBCC(ABC):
    """Base class for Brueckner orbital methods."""

    pass


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
