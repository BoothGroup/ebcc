"""Base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ebcc import util

if TYPE_CHECKING:
    from logging import Logger
    from typing import Optional, Union, Any
    from dataclasses import dataclass

    from pyscf.scf import SCF

    from ebcc.ansatz import Ansatz
    from ebcc.space import Space


class EOM(ABC):
    """Base class for equation-of-motion methods."""

    pass


class BruecknerEBCC(ABC):
    """Base class for Brueckner orbital methods."""

    pass


class ERIs(ABC, util.Namespace):
    """Base class for electronic repulsion integrals."""

    pass


class Fock(ABC, util.Namespace):
    """Base class for Fock matrices."""

    pass
