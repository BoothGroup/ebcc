"""Base classes for `ebcc.ham`."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy

from ebcc import BACKEND
from ebcc import numpy as np
from ebcc.backend import to_numpy
from ebcc.core.precision import types
from ebcc.util import Namespace

if TYPE_CHECKING:
    from typing import Optional

    from numpy import float64
    from numpy.typing import NDArray
    from pyscf.scf.hf import SCF

    from ebcc.cc.base import SpaceType
    from ebcc.cc.gebcc import SpaceType as GSpaceType
    from ebcc.cc.rebcc import SpaceType as RSpaceType
    from ebcc.cc.uebcc import SpaceType as USpaceType

    CoeffType = Any
    RCoeffType = NDArray[float64]
    UCoeffType = tuple[NDArray[float64], NDArray[float64]]
    GCoeffType = NDArray[float64]


class BaseHamiltonian(Namespace[Any], ABC):
    """Base class for Hamiltonians."""

    mf: SCF
    space: tuple[SpaceType, ...]
    mo_coeff: tuple[CoeffType, ...]

    def __init__(
        self,
        mf: SCF,
        space: tuple[SpaceType, ...],
        mo_coeff: Optional[tuple[CoeffType, ...]] = None,
    ) -> None:
        """Initialise the Hamiltonian.

        Args:
            mf: Mean-field object.
            space: Space object for each index.
            mo_coeff: Molecular orbital coefficients for each index.
        """
        Namespace.__init__(self)

        # Parameters:
        self.__dict__["mf"] = mf
        self.__dict__["space"] = space
        self.__dict__["mo_coeff"] = mo_coeff if mo_coeff is not None else (mf.mo_coeff,) * 4

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Slice of the Hamiltonian.
        """
        pass

    def _to_pyscf_backend(self, array: NDArray[Any]) -> NDArray[Any]:
        """Convert an array to the NumPy backend used by PySCF."""
        if BACKEND == "jax" and "pyscfad" in sys.modules:
            return array
        else:
            return to_numpy(array, dtype=numpy.float64)

    def _to_ebcc_backend(self, array: NDArray[Any]) -> NDArray[Any]:
        """Convert an array to the NumPy backend used by `ebcc`."""
        return np.asarray(array, dtype=types[float])

    def _get_slices(self, key: str) -> tuple[slice, ...]:
        """Get the slices for the given key.

        Args:
            key: Key to get.

        Returns:
            Slices for the given key.
        """
        slices = tuple(s.slice(k) for s, k in zip(self.space, key))
        return slices

    def _get_coeffs(self, key: str, offset: int = 0) -> tuple[NDArray[Any], ...]:
        """Get the coefficients for the given key.

        Args:
            key: Key to get.

        Returns:
            Coefficients for the given key.
        """
        coeffs = tuple(
            (
                self.mo_coeff[i + offset][:, self.space[i + offset].slice(k)]
                if k != "p"
                else np.eye(self.mo_coeff[i + offset].shape[0])
            )
            for i, k in enumerate(key)
        )
        return coeffs


class BaseRHamiltonian(BaseHamiltonian):
    """Base class for restricted Hamiltonians."""

    space: tuple[RSpaceType, ...]
    mo_coeff: tuple[RCoeffType, ...]


class BaseUHamiltonian(BaseHamiltonian):
    """Base class for unrestricted Hamiltonians."""

    space: tuple[USpaceType, ...]
    mo_coeff: tuple[UCoeffType, ...]


class BaseGHamiltonian(BaseHamiltonian):
    """Base class for general Hamiltonians."""

    space: tuple[GSpaceType, ...]
    mo_coeff: tuple[GCoeffType, ...]


class BaseFock(BaseHamiltonian):
    """Base class for Fock matrices."""

    def __init__(
        self,
        mf: SCF,
        space: tuple[SpaceType, ...],
        mo_coeff: Optional[tuple[CoeffType, ...]] = None,
        g: Optional[Namespace[Any]] = None,
        shift: Optional[bool] = None,
        xi: Optional[NDArray[float64]] = None,
    ) -> None:
        """Initialise the Hamiltonian.

        Args:
            mf: Mean-field object.
            space: Space object for each index.
            mo_coeff: Molecular orbital coefficients for each index.
            g: Namespace containing blocks of the electron-boson coupling matrix.
            shift: Shift the boson operators such that the Hamiltonian is normal-ordered with
                respect to a coherent state. This removes the bosonic coupling to the static
                mean-field density, introducing a constant energy shift.
            xi: Shift in the bosonic operators to diagonalise the photon Hamiltonian.
        """
        super().__init__(mf, space, mo_coeff=mo_coeff)

        # Boson parameters:
        self.__dict__["g"] = g
        self.__dict__["shift"] = shift
        self.__dict__["xi"] = xi


class BaseERIs(BaseHamiltonian):
    """Base class for electronic repulsion integrals."""

    pass


class BaseElectronBoson(BaseHamiltonian):
    """Base class for electron-boson coupling matrices."""

    def __init__(
        self,
        mf: SCF,
        g: NDArray[float64],
        space: tuple[SpaceType, ...],
        mo_coeff: Optional[tuple[CoeffType, ...]] = None,
    ) -> None:
        """Initialise the Hamiltonian.

        Args:
            mf: Mean-field object.
            g: The electron-boson coupling matrix array.
            space: Space object for each index.
            mo_coeff: Molecular orbital coefficients for each index.
        """
        super().__init__(mf, space, mo_coeff=mo_coeff)

        # Boson parameters:
        self.__dict__["g"] = g
