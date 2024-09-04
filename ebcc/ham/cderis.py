"""Cholesky-decomposed electron repulsion integral containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf import ao2mo, lib

from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import types
from ebcc.ham.base import BaseERIs
from ebcc.core.tensor import Tensor, loop_rank_block_indices, initialise_from_array, einsum

if TYPE_CHECKING:
    from typing import Optional

    from ebcc.numpy.typing import NDArray


class RCDERIs(BaseERIs):
    """Restricted Cholesky-decomposed ERIs container class."""

    def __getitem__(self, key: str, e2: Optional[bool] = False) -> NDArray[float]:
        """Just-in-time getter.

        Args:
            key: Key to get.
            e2: Whether the key is for the second electron.

        Returns:
            CDERIs for the given spaces.
        """
        if self.array is not None:
            raise NotImplementedError("`array` is not supported for CDERIs.")

        if len(key) == 4:
            e1 = self.__getitem__("Q" + key[:2])
            e2 = self.__getitem__("Q" + key[2:], e2=True)  # type: ignore
            return einsum("Qij,Qkl->ijkl", e1, e2)
        elif len(key) == 3:
            key = key[1:]
        else:
            raise KeyError("Key must be of length 3 or 4.")
        key_e2 = f"{key}_{'e1' if not e2 else 'e2'}"

        # Check the DF is built incore
        if not isinstance(self.cc.mf.with_df._cderi, np.ndarray):
            with lib.temporary_env(self.cc.mf.with_df, max_memory=1e6):
                self.cc.mf.with_df.build()

        if key_e2 not in self._members:
            s = 0 if not e2 else 2
            coeffs = [
                self.mo_coeff[i + s][:, self.space[i + s].mask(k)].astype(np.float64)
                for i, k in enumerate(key)
            ]
            block = Tensor(
                tuple(c.shape[1] for c in coeffs),
                permutations=[((0, 1, 2), 1), ((0, 2, 1), 1)] if key[0] == key[1] else [((0, 1, 2), 1)],
            )
            for indices in loop_rank_block_indices(block):
                part_coeffs = [c[:, i] for c, i in zip(coeffs, indices)]
                ijslice = (
                    0,
                    part_coeffs[0].shape[-1],
                    part_coeffs[0].shape[-1],
                    part_coeffs[0].shape[-1] + part_coeffs[1].shape[-1],
                )
                part_coeffs = np.concatenate(part_coeffs, axis=1)
                part = ao2mo._ao2mo.nr_e2(
                    self.cc.mf.with_df._cderi, part_coeffs, ijslice, aosym="s2", mosym="s1"
                )
                part = part.astype(block.dtype)
                part = part.reshape(-1, ijslice[1] - ijslice[0], ijslice[3] - ijslice[2])
            self._members[key_e2] = block

        return self._members[key_e2]


class UCDERIs(BaseERIs):
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
                self.cc,
                space=(self.space[0][i], self.space[1][i], self.space[2][j], self.space[3][j]),
                mo_coeff=(
                    self.mo_coeff[0][i],
                    self.mo_coeff[1][i],
                    self.mo_coeff[2][j],
                    self.mo_coeff[3][j],
                ),
            )

        return self._members[key]
