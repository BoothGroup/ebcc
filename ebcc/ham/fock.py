"""Fock matrix containers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import types
from ebcc.ham.base import BaseFock, BaseRHamiltonian, BaseUHamiltonian

if TYPE_CHECKING:
    from typing import Optional

    from numpy import float64
    from numpy.typing import NDArray

    T = float64


class RFock(BaseFock, BaseRHamiltonian):
    """Restricted Fock matrix container class."""

    _members: dict[str, NDArray[T]]
    _spin_index: Optional[int] = None

    def __getitem__(self, key: str) -> NDArray[T]:
        """Just-in-time getter.

        Args:
            key: Key to get.

        Returns:
            Fock matrix for the given spaces.
        """
        if key not in self._members:
            # Get the coefficients
            coeffs = self._get_coeffs(key)

            # Transform the block
            fock_ao: NDArray[T] = np.asarray(self.mf.get_fock(), dtype=types[float])
            if self._spin_index is not None:
                fock_ao = fock_ao[self._spin_index]
            block = util.einsum("pq,pi,qj->ij", fock_ao, *coeffs)

            # Store the block
            self._members[key] = block

            if self.shift:
                # Shift for bosons
                xi = self.xi
                g = np.copy(self.g.__getattr__(f"b{key}"))
                g += np.transpose(self.g.__getattr__(f"b{key[::-1]}"), (0, 2, 1))
                self._members[key] -= util.einsum("I,Ipq->pq", xi, g)

        return self._members[key]


class UFock(BaseFock, BaseUHamiltonian):
    """Unrestricted Fock matrix container class."""

    _members: dict[str, RFock]

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
                self.mf,
                space=(self.space[0][i], self.space[1][i]),
                mo_coeff=(self.mo_coeff[0][i], self.mo_coeff[1][i]),
                g=self.g[key] if self.g is not None else None,
                shift=self.shift,
                xi=self.xi,
            )
            self._members[key].__dict__["_spin_index"] = i

        return self._members[key]


class GFock(RFock):
    """Generalised Fock matrix container class."""

    pass
