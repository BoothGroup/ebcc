"""Fock matrix containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import types
from ebcc.ham.base import BaseFock
from ebcc.core.tensor import Tensor, initialise_from_array, einsum as tensor_einsum

if TYPE_CHECKING:
    from ebcc.numpy.typing import NDArray


class RFock(BaseFock):
    """Restricted Fock matrix container class."""

    _members: dict[str, NDArray[float]]

    def _get_fock(self) -> NDArray[float]:
        fock_ao = self.cc.mf.get_fock().astype(types[float])
        return util.einsum("pq,pi,qj->ij", fock_ao, self.mo_coeff[0], self.mo_coeff[1])

    def __getitem__(self, key: str) -> NDArray[float]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Fock matrix for the given spaces.
        """
        if key not in self._members:
            i = self.space[0].mask(key[0])
            j = self.space[1].mask(key[1])
            self._members[key] = initialise_from_array(
                self.array[i][:, j].copy(),
                permutations=[((0, 1), 1), ((1, 0), 1)] if key[0] == key[1] else [((0, 1), 1)],
            )

            if self.shift:
                xi = initialise_from_array(self.xi, permutations=[((0,), 1)])
                g = self.g.__getattr__(f"b{key}").copy()
                g += self.g.__getattr__(f"b{key[::-1]}").transpose(0, 2, 1)
                self._members[key] -= tensor_einsum("I,Ipq->pq", xi, g)

        return self._members[key]


class UFock(BaseFock):
    """Unrestricted Fock matrix container class."""

    _members: dict[str, RFock]

    def _get_fock(self) -> tuple[NDArray[float], NDArray[float]]:
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


class GFock(BaseFock):
    """Generalised Fock matrix container class."""

    _members: dict[str, NDArray[float]]

    def _get_fock(self) -> NDArray[float]:
        fock_ao = self.cc.mf.get_fock().astype(types[float])
        return util.einsum("pq,pi,qj->ij", fock_ao, self.mo_coeff[0], self.mo_coeff[1])

    def __getitem__(self, key: str) -> NDArray[float]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Fock matrix for the given spin.
        """
        if key not in self._members:
            i = self.space[0].mask(key[0])
            j = self.space[1].mask(key[1])
            self._members[key] = initialise_from_array(
                self.array[i][:, j].copy(),
                permutations=[((0, 1), 1), ((1, 0), 1)] if key[0] == key[1] else [((0, 1), 1)],
            )

            if self.shift:
                xi = initialise_from_array(self.xi, permutations=[((0,), 1)])
                g = self.g.__getattr__(f"b{key}").copy()
                g += self.g.__getattr__(f"b{key[::-1]}").transpose(0, 2, 1)
                self._members[key] -= tensor_einsum("I,Ipq->pq", xi, g)

        return self._members[key]
