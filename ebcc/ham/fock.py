"""Fock matrix containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import util
from ebcc.core.precision import types
from ebcc.ham.base import BaseFock, BaseGHamiltonian, BaseRHamiltonian, BaseUHamiltonian

if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray

    T = float64


class RFock(BaseFock, BaseRHamiltonian):
    """Restricted Fock matrix container class."""

    _members: dict[str, NDArray[T]]

    def _get_fock(self) -> NDArray[T]:
        fock_ao = self.cc.mf.get_fock().astype(types[float])
        return util.einsum("pq,pi,qj->ij", fock_ao, self.mo_coeff[0], self.mo_coeff[1])

    def __getitem__(self, key: str) -> NDArray[T]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Fock matrix for the given spaces.
        """
        if key not in self._members:
            i = self.space[0].mask(key[0])
            j = self.space[1].mask(key[1])
            self._members[key] = self.array[i][:, j].copy()

            if self.shift:
                xi = self.xi
                g = self.g.__getattr__(f"b{key}").copy()
                g += self.g.__getattr__(f"b{key[::-1]}").transpose(0, 2, 1)
                self._members[key] -= util.einsum("I,Ipq->pq", xi, g)

        return self._members[key]


class UFock(BaseFock, BaseUHamiltonian):
    """Unrestricted Fock matrix container class."""

    _members: dict[str, RFock]

    def _get_fock(self) -> tuple[NDArray[T], NDArray[T]]:
        fock_ao = self.cc.mf.get_fock().astype(types[float])
        return (
            util.einsum("pq,pi,qj->ij", fock_ao[0], self.mo_coeff[0][0], self.mo_coeff[1][0]),
            util.einsum("pq,pi,qj->ij", fock_ao[1], self.mo_coeff[0][1], self.mo_coeff[1][1]),
        )

    def __getitem__(self, key: str) -> RFock:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Fock matrix for the given spin.
        """
        if key not in ("aa", "bb"):
            raise KeyError(f"Invalid key: {key}")
        if key not in self._members:
            i = "ab".index(key[0])
            self._members[key] = RFock(
                self.cc,
                array=self.array[i],
                space=(self.space[0][i], self.space[1][i]),
                mo_coeff=(self.mo_coeff[0][i], self.mo_coeff[1][i]),
                g=self.g[key] if self.g is not None else None,
            )
        return self._members[key]


class GFock(BaseFock, BaseGHamiltonian):
    """Generalised Fock matrix container class."""

    _members: dict[str, NDArray[T]]

    def _get_fock(self) -> NDArray[T]:
        fock_ao = self.cc.mf.get_fock().astype(types[float])
        return util.einsum("pq,pi,qj->ij", fock_ao, self.mo_coeff[0], self.mo_coeff[1])

    def __getitem__(self, key: str) -> NDArray[T]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Fock matrix for the given spin.
        """
        if key not in self._members:
            i = self.space[0].mask(key[0])
            j = self.space[1].mask(key[1])
            self._members[key] = self.array[i][:, j].copy()

            if self.shift:
                xi = self.xi
                g = self.g.__getattr__(f"b{key}").copy()
                g += self.g.__getattr__(f"b{key[::-1]}").transpose(0, 2, 1)
                self._members[key] -= util.einsum("I,Ipq->pq", xi, g)

        return self._members[key]
