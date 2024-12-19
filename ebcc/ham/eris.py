"""Electronic repulsion integral containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import pyscf
from ebcc.core.precision import types
from ebcc.ham.base import BaseERIs, BaseGHamiltonian, BaseRHamiltonian, BaseUHamiltonian

if TYPE_CHECKING:
    from typing import Any

    from numpy import floating
    from numpy.typing import NDArray

    T = floating


class RERIs(BaseERIs, BaseRHamiltonian):
    """Restricted ERIs container class."""

    _members: dict[str, NDArray[T]]

    def __getitem__(self, key: str) -> NDArray[T]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            ERIs for the given spaces.
        """
        if key not in self._members.keys():
            # Get the coefficients and shape
            coeffs = self._get_coeffs(key)
            coeffs = tuple(self._to_pyscf_backend(c) for c in coeffs)
            shape = tuple(c.shape[-1] for c in coeffs)

            # Transform the block
            # TODO: Optimise for patially AO
            if getattr(self.mf, "_eri", None) is not None:
                block = pyscf.ao2mo.incore.general(self.mf._eri, coeffs, compact=False)
            else:
                block = pyscf.ao2mo.kernel(self.mf.mol, coeffs, compact=False)
            block = np.reshape(block, shape)

            # Store the block
            self._members[key] = np.asarray(block, dtype=types[float])

        return self._members[key]


class UERIs(BaseERIs, BaseUHamiltonian):
    """Unrestricted ERIs container class."""

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

            self._members[key] = RERIs(
                self.mf,
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
    _array: NDArray[T]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the class."""
        super().__init__(*args, **kwargs)

        # Get the coefficients and shape
        mo_a = [self._to_pyscf_backend(mo[: self.mf.mol.nao]) for mo in self.mo_coeff]
        mo_b = [self._to_pyscf_backend(mo[self.mf.mol.nao :]) for mo in self.mo_coeff]
        shape = tuple(mo.shape[-1] for mo in self.mo_coeff)
        if len(set(shape)) != 1:
            raise ValueError(
                "MO coefficients must have the same number of basis functions for "
                f"{self.__class__.__name__}."
            )
        nmo = shape[0]

        if getattr(self.mf, "_eri", None) is not None:
            array = pyscf.ao2mo.incore.general(self.mf._eri, mo_a)
            array += pyscf.ao2mo.incore.general(self.mf._eri, mo_b)
            array += pyscf.ao2mo.incore.general(self.mf._eri, mo_a[:2] + mo_b[2:])
            array += pyscf.ao2mo.incore.general(self.mf._eri, mo_b[:2] + mo_a[2:])
        else:
            array = pyscf.ao2mo.kernel(self.mf.mol, mo_a)
            array += pyscf.ao2mo.kernel(self.mf.mol, mo_b)
            array += pyscf.ao2mo.kernel(self.mf.mol, mo_a[:2] + mo_b[2:])
            array += pyscf.ao2mo.kernel(self.mf.mol, mo_b[:2] + mo_a[2:])
        array = pyscf.ao2mo.addons.restore(1, array, nmo)
        array = np.reshape(array, shape)
        array = self._to_ebcc_backend(array)

        # Transform to antisymmetric Physicist's notation
        array = np.transpose(array, (0, 2, 1, 3)) - np.transpose(array, (0, 2, 3, 1))

        # Store the array
        self.__dict__["_array"] = array

    def __getitem__(self, key: str) -> NDArray[T]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            ERIs for the given spaces.
        """
        if "p" in key:
            raise NotImplementedError(f"AO basis not supported in {self.__class__.__name__}.")
        return self._array[self._get_slices(key)]
