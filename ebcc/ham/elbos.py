"""Electron-boson coupling matrix containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc.ham.base import BaseElectronBoson

if TYPE_CHECKING:
    from ebcc.numpy.typing import NDArray


class RElectronBoson(BaseElectronBoson):
    """Restricted electron-boson coupling matrices."""

    _members: dict[str, NDArray[float]]

    def __getitem__(self, key: str) -> NDArray[float]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Electron-boson coupling matrix for the given spaces.
        """
        if key not in self._members:
            assert key[0] == "b"
            i = self.space[0].mask(key[1])
            j = self.space[1].mask(key[2])
            self._members[key] = self.array[:, i][:, :, j].copy()
        return self._members[key]


class UElectronBoson(BaseElectronBoson):
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
                array=self.array[i] if np.asarray(self.array).ndim == 4 else self.array,
                space=(self.space[0][i], self.space[1][i]),
            )
        return self._members[key]


class GElectronBoson(BaseElectronBoson):
    """Generalised electron-boson coupling matrices."""

    _members: dict[str, NDArray[float]]

    def __getitem__(self, key: str) -> NDArray[float]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Electron-boson coupling matrix for the given spaces.
        """
        if key not in self._members:
            assert key[0] == "b"
            i = self.space[0].mask(key[1])
            j = self.space[1].mask(key[2])
            self._members[key] = self.array[:, i][:, :, j].copy()
        return self._members[key]
