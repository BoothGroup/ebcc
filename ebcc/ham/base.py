"""Base classes for `ebcc.ham`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ebcc.util import Namespace

if TYPE_CHECKING:
    from typing import Optional

    from ebcc.cc.base import BaseEBCC
    from ebcc.cc.gebcc import GEBCC
    from ebcc.cc.rebcc import REBCC
    from ebcc.cc.uebcc import UEBCC


class BaseHamiltonian(Namespace[Any], ABC):
    """Base class for Hamiltonians."""

    cc: BaseEBCC

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Slice of the Hamiltonian.
        """
        pass


class BaseRHamiltonian(BaseHamiltonian):
    """Base class for restricted Hamiltonians."""

    cc: REBCC


class BaseUHamiltonian(BaseHamiltonian):
    """Base class for unrestricted Hamiltonians."""

    cc: UEBCC


class BaseGHamiltonian(BaseHamiltonian):
    """Base class for general Hamiltonians."""

    cc: GEBCC


class BaseFock(BaseHamiltonian):
    """Base class for Fock matrices."""

    def __init__(
        self,
        cc: BaseEBCC,
        array: Optional[Any] = None,
        space: Optional[tuple[Any, ...]] = None,
        mo_coeff: Optional[tuple[Any, ...]] = None,
        g: Optional[Namespace[Any]] = None,
    ) -> None:
        """Initialise the Fock matrix.

        Args:
            cc: Coupled cluster object.
            array: Fock matrix in the MO basis.
            space: Space object for each index.
            mo_coeff: Molecular orbital coefficients for each index.
            g: Namespace containing blocks of the electron-boson coupling matrix.
        """
        Namespace.__init__(self)

        # Parameters:
        self.__dict__["cc"] = cc
        self.__dict__["space"] = space if space is not None else (cc.space,) * 2
        self.__dict__["mo_coeff"] = mo_coeff if mo_coeff is not None else (cc.mo_coeff,) * 2
        self.__dict__["array"] = array if array is not None else self._get_fock()
        self.__dict__["g"] = g if g is not None else cc.g

        # Boson parameters:
        self.__dict__["shift"] = cc.options.shift if g is not None else None
        self.__dict__["xi"] = cc.xi if g is not None else None

    @abstractmethod
    def _get_fock(self) -> Any:
        """Get the Fock matrix."""
        pass


class BaseERIs(BaseHamiltonian):
    """Base class for electronic repulsion integrals."""

    def __init__(
        self,
        cc: BaseEBCC,
        array: Optional[Any] = None,
        space: Optional[tuple[Any, ...]] = None,
        mo_coeff: Optional[tuple[Any, ...]] = None,
    ) -> None:
        """Initialise the ERIs.

        Args:
            cc: Coupled cluster object.
            array: ERIs in the MO basis.
            space: Space object for each index.
            mo_coeff: Molecular orbital coefficients for each index.
        """
        Namespace.__init__(self)

        # Parameters:
        self.__dict__["cc"] = cc
        self.__dict__["space"] = space if space is not None else (cc.space,) * 4
        self.__dict__["mo_coeff"] = mo_coeff if mo_coeff is not None else (cc.mo_coeff,) * 4
        self.__dict__["array"] = array if array is not None else None


class BaseElectronBoson(BaseHamiltonian):
    """Base class for electron-boson coupling matrices."""

    def __init__(
        self,
        cc: BaseEBCC,
        array: Optional[Any] = None,
        space: Optional[tuple[Any, ...]] = None,
    ) -> None:
        """Initialise the electron-boson coupling matrix.

        Args:
            cc: Coupled cluster object.
            array: Electron-boson coupling matrix in the MO basis.
            space: Space object for each index.
        """
        Namespace.__init__(self)

        # Parameters:
        self.__dict__["cc"] = cc
        self.__dict__["space"] = space if space is not None else (cc.space,) * 2
        self.__dict__["array"] = array if array is not None else self._get_g()
