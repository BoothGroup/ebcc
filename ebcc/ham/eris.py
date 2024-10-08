"""Electronic repulsion integral containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy  # PySCF uses true numpy, no backend stuff here
from pyscf import ao2mo

from ebcc import numpy as np
from ebcc.backend import to_numpy
from ebcc.core.precision import types
from ebcc.ham.base import BaseERIs, BaseGHamiltonian, BaseRHamiltonian, BaseUHamiltonian

if TYPE_CHECKING:
    from typing import Any, Optional

    from numpy import float64
    from numpy.typing import NDArray

    T = float64


class RERIs(BaseERIs, BaseRHamiltonian):
    """Restricted ERIs container class."""

    _members: dict[str, NDArray[T]]
    array: Optional[NDArray[T]]

    def __getitem__(self, key: str) -> NDArray[T]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            ERIs for the given spaces.
        """
        if self.array is None:
            if key not in self._members.keys():
                coeffs = [
                    to_numpy(self.mo_coeff[i][:, self.space[i].slice(k)], dtype=numpy.float64)
                    for i, k in enumerate(key)
                ]
                if getattr(self.cc.mf, "_eri", None) is not None:
                    block = ao2mo.incore.general(self.cc.mf._eri, coeffs, compact=False)
                else:
                    block = ao2mo.kernel(self.cc.mf.mol, coeffs, compact=False)
                block = np.reshape(block, [c.shape[-1] for c in coeffs])
                self._members[key] = np.asarray(block, dtype=types[float])
            return self._members[key]
        else:
            ijkl = tuple(self.space[i].slice(k) for i, k in enumerate(key))
            return self.array[ijkl]  # type: ignore


class UERIs(BaseERIs, BaseUHamiltonian):
    """Unrestricted ERIs container class."""

    _members: dict[str, RERIs]
    array: Optional[tuple[NDArray[T], ...]]

    def __getitem__(self, key: str) -> RERIs:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            ERIs for the given spins.
        """
        if key not in ("aaaa", "aabb", "bbaa", "bbbb"):
            raise KeyError(f"Invalid key: {key}")
        if key not in self._members:
            i = "ab".index(key[0])
            j = "ab".index(key[2])
            ij = i * (i + 1) // 2 + j

            if self.array is not None:
                array = self.array[ij]
                if key == "bbaa":
                    array = np.transpose(array, (2, 3, 0, 1))
            elif isinstance(self.cc.mf._eri, tuple):
                # Support spin-dependent integrals in the mean-field
                coeffs = [
                    to_numpy(self.mo_coeff[x][y], dtype=numpy.float64)
                    for y, x in enumerate(sorted((i, i, j, j)))
                ]
                if getattr(self.cc.mf, "_eri", None) is not None:
                    array = ao2mo.incore.general(self.cc.mf.mol, coeffs, compact=False)
                else:
                    array = ao2mo.kernel(self.cc.mf.mol, coeffs, compact=False)
                if key == "bbaa":
                    array = np.transpose(array, (2, 3, 0, 1))
                array = np.asarray(array, dtype=types[float])
            else:
                array = None

            self._members[key] = RERIs(
                self.cc,
                array=array,
                space=(self.space[0][i], self.space[1][i], self.space[2][j], self.space[3][j]),
                mo_coeff=(
                    self.mo_coeff[0][i],
                    self.mo_coeff[1][i],
                    self.mo_coeff[2][j],
                    self.mo_coeff[3][j],
                ),
            )
        return self._members[key]


class GERIs(BaseERIs, BaseGHamiltonian):
    """Generalised ERIs container class."""

    _members: dict[str, NDArray[T]]
    array: NDArray[T]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the class."""
        super().__init__(*args, **kwargs)
        if self.array is None:
            mo_a = [to_numpy(mo[: self.cc.mf.mol.nao], dtype=numpy.float64) for mo in self.mo_coeff]
            mo_b = [to_numpy(mo[self.cc.mf.mol.nao :], dtype=numpy.float64) for mo in self.mo_coeff]
            if getattr(self.cc.mf, "_eri", None) is not None:
                array = ao2mo.incore.general(self.cc.mf._eri, mo_a)
                array += ao2mo.incore.general(self.cc.mf._eri, mo_b)
                array += ao2mo.incore.general(self.cc.mf._eri, mo_a[:2] + mo_b[2:])
                array += ao2mo.incore.general(self.cc.mf._eri, mo_b[:2] + mo_a[2:])
            else:
                array = ao2mo.kernel(self.cc.mf.mol, mo_a)
                array += ao2mo.kernel(self.cc.mf.mol, mo_b)
                array += ao2mo.kernel(self.cc.mf.mol, mo_a[:2] + mo_b[2:])
                array += ao2mo.kernel(self.cc.mf.mol, mo_b[:2] + mo_a[2:])
            array = np.reshape(ao2mo.addons.restore(1, array, self.cc.nmo), (self.cc.nmo,) * 4)
            array = np.asarray(array, dtype=types[float])
            array = np.transpose(array, (0, 2, 1, 3)) - np.transpose(array, (0, 2, 3, 1))
            self.__dict__["array"] = array

    def __getitem__(self, key: str) -> NDArray[T]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            ERIs for the given spaces.
        """
        ijkl = tuple(self.space[i].slice(k) for i, k in enumerate(key))
        return self.array[ijkl]  # type: ignore
