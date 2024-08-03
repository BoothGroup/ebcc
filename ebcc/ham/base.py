"""Base classes for `ebcc.ham`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, TypeVar

    from ebcc.cc.base import BaseEBCC
    from ebcc.util import Namespace

    T = TypeVar("T")


class BaseFock(ABC):
    """Base class for Fock matrices.

    Attributes:
        cc: Coupled cluster object.
        space: Space object.
        mo_coeff: Molecular orbital coefficients.
        array: Fock matrix in the MO basis.
        g: Namespace containing blocks of the electron-boson coupling matrix.
        shift: Shift parameter.
        xi: Boson parameters.
    """

    def __init__(
        self,
        cc: BaseEBCC,
        array: T = None,
        space: Any = None,
        mo_coeff: T = None,
        g: Namespace[Any] = None,
    ) -> None:
        """Initialise the Fock matrix.

        Args:
            cc: Coupled cluster object.
            array: Fock matrix in the MO basis.
            space: Space object.
            mo_coeff: Molecular orbital coefficients.
            g: Namespace containing blocks of the electron-boson coupling matrix.
        """
        # Parameters:
        self.cc = cc
        self.space = space if space is not None else cc.space
        self.mo_coeff = mo_coeff if mo_coeff is not None else cc.mo_coeff
        self.array = array if array is not None else self._get_fock()
        self.g = g if g is not None else cc.g

        # Boson parameters:
        self.shift = cc.options.shift
        self.xi = cc.xi

    @abstractmethod
    def _get_fock(self) -> T:
        """Get the Fock matrix."""
        pass

    @abstractmethod
    def __getattr__(self, key: str) -> Any:
        """Just-in-time attribute getter.

        Args:
            key: Key to get.

        Returns:
            Slice of the Fock matrix.
        """
        pass

    def __getitem__(self, key: str) -> Any:
        """Get an item."""
        return self.__getattr__(key)


class BaseERIs(ABC):
    """Base class for electronic repulsion integrals."""

    def __getitem__(self, key: str) -> Any:
        """Get an item."""
        return self.__dict__[key]
