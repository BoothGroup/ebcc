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
            i = self.space[0].slice(key[1])
            j = self.space[1].slice(key[2])
            self._members[key] = np.copy(self.array[:, i, j])
        return self._members[key]

    def _get_g(self) -> NDArray[T]:
        """Get the electron-boson coupling matrix."""
        assert self.cc.bare_g is not None
        return self.cc.bare_g


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
                self.cc,
                array=self.array[i] if self.array.ndim == 4 else self.array,
                space=(self.space[0][i], self.space[1][i]),
            )
        return self._members[key]

    def _get_g(self) -> NDArray[T]:
        """Get the electron-boson coupling matrix."""
        assert self.cc.bare_g is not None
        return self.cc.bare_g


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
            i = self.space[0].slice(key[1])
            j = self.space[1].slice(key[2])
            self._members[key] = np.copy(self.array[:, i, j])
        return self._members[key]

    def _get_g(self) -> NDArray[T]:
        """Get the electron-boson coupling matrix."""
        assert self.cc.bare_g is not None
        return self.cc.bare_g
