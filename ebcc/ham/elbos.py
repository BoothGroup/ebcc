"""Electron-boson coupling matrix containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc.ham.base import BaseElectronBoson, BaseGHamiltonian, BaseRHamiltonian, BaseUHamiltonian

if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray

    T = float64


class RElectronBoson(BaseElectronBoson, BaseRHamiltonian):
    """Restricted electron-boson coupling matrices."""

    _members: dict[str, NDArray[T]]

    def __getitem__(self, key: str) -> NDArray[T]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Electron-boson coupling matrix for the given spaces.
        """
        if key not in self._members:
            assert key[0] == "b"
            if "p" in key:
                raise NotImplementedError(f"AO basis not supported in {self.__class__.__name__}.")

            # Get the slices
            slices = (slice(None),) + self._get_slices(key[1:])

            # Store the block
            self._members[key] = np.copy(self.g[slices])

        return self._members[key]


class UElectronBoson(BaseElectronBoson, BaseUHamiltonian):
    """Unrestricted electron-boson coupling matrices."""

    _members: dict[str, RElectronBoson]

    def __getitem__(self, key: str) -> RElectronBoson:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Electron-boson coupling matrix for the given spin.
        """
        if key not in ("aa", "bb"):
            raise KeyError(f"Invalid key: {key}")

        if key not in self._members:
            i = "ab".index(key[0])
            self._members[key] = RElectronBoson(
                self.mf,
                self.g[i] if self.g.ndim == 4 else self.g,
                space=(self.space[0][i], self.space[1][i]),
            )
            self._members[key]._spin_index = i

        return self._members[key]


class GElectronBoson(BaseElectronBoson, BaseGHamiltonian):
    """Generalised electron-boson coupling matrices."""

    _members: dict[str, NDArray[T]]

    def __getitem__(self, key: str) -> NDArray[T]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Electron-boson coupling matrix for the given spaces.
        """
        if key not in self._members:
            assert key[0] == "b"
            if "p" in key:
                raise NotImplementedError(f"AO basis not supported in {self.__class__.__name__}.")

            # Get the slices
            slices = (slice(None),) + self._get_slices(key[1:])

            # Store the block
            self._members[key] = np.copy(self.g[slices])

        return self._members[key]
