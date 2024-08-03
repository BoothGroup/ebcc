"""Fock matrix containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.ham.base import BaseFock
from ebcc.precision import types

if TYPE_CHECKING:
    from ebcc.numpy.typing import NDArray


class RFock(BaseFock):
    """Restricted Fock matrix container class.

    Attributes:
        cc: Coupled cluster object.
        space: Space object.
        mo_coeff: Molecular orbital coefficients.
        array: Fock matrix in the MO basis.
        g: Namespace containing blocks of the electron-boson coupling matrix.
        shift: Shift parameter.
        xi: Boson parameters.
    """

    def _get_fock(self) -> NDArray[float]:
        fock_ao = self.mf.get_fock().astype(types[float])
        return util.einsum("pq,pi,qj->ij", fock_ao, self.mo_coeff, self.mo_coeff)

    def __getattr__(self, key: str) -> NDArray[float]:
        """Just-in-time attribute getter.

        Args:
            key: Key to get.

        Returns:
            Fock matrix for the given spin.
        """
        if key not in self.__dict__:
            i = self.space.mask(key[0])
            j = self.space.mask(key[1])
            self.__dict__[key] = self.array[i][:, j].copy()

            if self.shift:
                xi = self.xi
                g = self.g.__getattr__(f"b{key}")
                g += self.g.__getattr__(f"b{key[::-1]}").transpose(0, 2, 1)
                self.__dict__[key] -= np.einsum("I,Ipq->pq", xi, g)

        return self.__dict__[key]


class UFock(BaseFock):
    """Unrestricted Fock matrix container class."""

    def _get_fock(self) -> tuple[NDArray[float]]:
        fock_ao = self.mf.get_fock().astype(types[float])
        return (
            util.einsum("pq,pi,qj->ij", fock_ao[0], self.mo_coeff[0], self.mo_coeff[0]),
            util.einsum("pq,pi,qj->ij", fock_ao[1], self.mo_coeff[1], self.mo_coeff[1]),
        )

    def __getattr__(self, key: str) -> RFock:
        """Just-in-time attribute getter.

        Args:
            key: Key to get.

        Returns:
            Slice of the Fock matrix.
        """
        if key not in ("aa", "bb"):
            raise KeyError(f"Invalid key: {key}")
        if key not in self.__dict__:
            i = "ab".index(key[0])
            self.__dict__[key] = RFock(
                self.cc,
                array=self.array[i],
                space=self.space[i],
                mo_coeff=self.mo_coeff[i],
                g=self.g[key] if self.g is not None else None,
            )
        return self.__dict__[key]


class GFock(BaseFock):
    """Generalised Fock matrix container class.

    Attributes:
        cc: Coupled cluster object.
        space: Space object.
        mo_coeff: Molecular orbital coefficients.
        array: Fock matrix in the MO basis.
        g: Namespace containing blocks of the electron-boson coupling matrix.
        shift: Shift parameter.
        xi: Boson parameters.
    """

    def _get_fock(self) -> NDArray[float]:
        fock_ao = self.mf.get_fock().astype(types[float])
        return util.einsum("pq,pi,qj->ij", fock_ao, self.mo_coeff, self.mo_coeff)

    def __getattr__(self, key: str) -> NDArray[float]:
        """Just-in-time attribute getter.

        Args:
            key: Key to get.

        Returns:
            Fock matrix for the given spin.
        """
        if key not in self.__dict__:
            i = self.space.mask(key[0])
            j = self.space.mask(key[1])
            self.__dict__[key] = self.array[i][:, j].copy()

            if self.shift:
                xi = self.xi
                g = self.g.__getattr__(f"b{key}")
                g += self.g.__getattr__(f"b{key[::-1]}").transpose(0, 2, 1)
                self.__dict__[key] -= np.einsum("I,Ipq->pq", xi, g)

        return self.__dict__[key]
