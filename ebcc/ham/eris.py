"""Electronic repulsion integral containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf import ao2mo

from ebcc import numpy as np
from ebcc.core.precision import types
from ebcc.ham.base import BaseERIs

if TYPE_CHECKING:
    from typing import Any, Optional

    from ebcc.numpy.typing import NDArray


class RERIs(BaseERIs):
    """Restricted ERIs container class.

    Attributes:
        cc: Coupled cluster object.
        space: Space object for each index.
        mo_coeff: Molecular orbital coefficients for each index.
        array: ERIs in the MO basis.
    """

    _members: dict[str, NDArray[float]]

    def __getitem__(self, key: str) -> NDArray[float]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            ERIs for the given spaces.
        """
        if self.array is None:
            if key not in self._members.keys():
                coeffs = [
                    self.mo_coeff[i][:, self.space[i].mask(k)].astype(np.float64)
                    for i, k in enumerate(key)
                ]
                block = ao2mo.kernel(self.cc.mf.mol, coeffs, compact=False, max_memory=1e6)
                block = block.reshape([c.shape[-1] for c in coeffs])
                self._members[key] = block.astype(types[float])
            return self._members[key]
        else:
            i, j, k, l = [self.space[i].mask(k) for i, k in enumerate(key)]
            block = self.array[i][:, j][:, :, k][:, :, :, l]
            return block


class UERIs(BaseERIs):
    """Unrestricted ERIs container class.

    Attributes:
        cc: Coupled cluster object.
        space: Space object for each index.
        mo_coeff: Molecular orbital coefficients for each index.
        array: ERIs in the MO basis.
    """

    _members: dict[str, RERIs]

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

            array: Optional[NDArray[float]] = None
            if self.array is not None:
                array = self.array[ij]
                if key == "bbaa":
                    array = array.transpose(2, 3, 0, 1)
            elif isinstance(self.cc.mf._eri, tuple):
                # Support spin-dependent integrals in the mean-field
                coeffs = [
                    self.mo_coeff[x][y].astype(np.float64)
                    for y, x in enumerate(sorted((i, i, j, j)))
                ]
                array = ao2mo.kernel(self.cc.mf.mol, coeffs, compact=False, max_memory=1e6)
                if key == "bbaa":
                    array = array.transpose(2, 3, 0, 1)
                array = array.astype(types[float])

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


class GERIs(BaseERIs):
    """Generalised ERIs container class.

    Attributes:
        cc: Coupled cluster object.
        space: Space object for each index.
        mo_coeff: Molecular orbital coefficients for each index.
        array: ERIs in the MO basis.
    """

    _members: dict[str, UERIs]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the class."""
        super().__init__(*args, **kwargs)
        if self.array is None:
            mo_a = [mo[: self.cc.mf.mol.nao].astype(np.float64) for mo in self.mo_coeff]
            mo_b = [mo[self.cc.mf.mol.nao :].astype(np.float64) for mo in self.mo_coeff]
            array = ao2mo.kernel(self.cc.mf.mol, mo_a)
            array += ao2mo.kernel(self.cc.mf.mol, mo_b)
            array += ao2mo.kernel(self.cc.mf.mol, mo_a[:2] + mo_b[2:])
            array += ao2mo.kernel(self.cc.mf.mol, mo_b[:2] + mo_a[2:])
            array = ao2mo.addons.restore(1, array, self.cc.nmo).reshape((self.cc.nmo,) * 4)
            array = array.astype(types[float])
            array = array.transpose(0, 2, 1, 3) - array.transpose(0, 2, 3, 1)
            self.__dict__["array"] = array

    def __getitem__(self, key: str) -> NDArray[float]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            ERIs for the given spaces.
        """
        i, j, k, l = [self.space[i].mask(k) for i, k in enumerate(key)]
        block = self.array[i][:, j][:, :, k][:, :, :, l]
        return block
