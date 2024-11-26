"""Cholesky-decomposed electron repulsion integral containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf import lib

from ebcc import numpy as np
from ebcc import pyscf, util
from ebcc.ham.base import BaseERIs, BaseRHamiltonian, BaseUHamiltonian

if TYPE_CHECKING:
    from typing import Optional

    from numpy import float64
    from numpy.typing import NDArray

    T = float64


class RCDERIs(BaseERIs, BaseRHamiltonian):
    """Restricted Cholesky-decomposed ERIs container class."""

    _members: dict[str, NDArray[T]]

    def __getitem__(self, key: str, e2: Optional[bool] = False) -> NDArray[T]:
        """Just-in-time getter.

        Args:
            key: Key to get.
            e2: Whether the key is for the second electron.

        Returns:
            CDERIs for the given spaces.
        """
        if len(key) == 4:
            v1 = self.__getitem__("Q" + key[:2])
            v2 = self.__getitem__("Q" + key[2:], e2=True)  # type: ignore
            return util.einsum("Qij,Qkl->ijkl", v1, v2)
        elif len(key) == 3:
            key = key[1:]
        else:
            raise KeyError("Key must be of length 3 or 4.")
        key_e2 = f"{key}_{'e1' if not e2 else 'e2'}"

        # Check the DF is built incore
        if not isinstance(self.mf.with_df._cderi, np.ndarray):
            with lib.temporary_env(self.mf.with_df, max_memory=1e6):
                self.mf.with_df.build()

        if key_e2 not in self._members:
            s = 0 if not e2 else 2

            # Get the coefficients and shape
            coeffs = self._get_coeffs(key, offset=s)
            shape = tuple(c.shape[-1] for c in coeffs)
            ijslice = (0, shape[0], shape[0], shape[0] + shape[1])
            coeffs = np.concatenate(coeffs, axis=1)
            coeffs = self._to_pyscf_backend(coeffs)

            # Transform the block
            # TODO: Optimise for (L|pp)
            block = pyscf.ao2mo._ao2mo.nr_e2(
                self.mf.with_df._cderi, coeffs, ijslice, aosym="s2", mosym="s1"
            )
            block = self._to_ebcc_backend(block)
            block = np.reshape(block, (-1, *shape))

            # Store the block
            self._members[key_e2] = block

        return self._members[key_e2]


class UCDERIs(BaseERIs, BaseUHamiltonian):
    """Unrestricted Cholesky-decomposed ERIs container class."""

    _members: dict[str, RCDERIs]

    def __getitem__(self, key: str) -> RCDERIs:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            CDERIs for the given spins.
        """
        if len(key) == 3:
            key = key[1:]
        if key not in ("aa", "bb", "aaaa", "aabb", "bbaa", "bbbb"):
            raise KeyError(f"Invalid key: {key}")
        if len(key) == 2:
            key = key + key
        i = "ab".index(key[0])
        j = "ab".index(key[2])

        if key not in self._members:
            self._members[key] = RCDERIs(
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
