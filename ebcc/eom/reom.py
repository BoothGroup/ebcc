"""Restricted equation-of-motion coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import astype
from ebcc.eom.base import BaseEA_EOM, BaseEE_EOM, BaseEOM, BaseIP_EOM

if TYPE_CHECKING:
    from typing import Optional

    from ebcc.cc.rebcc import REBCC, ERIsInputType, SpinArrayType
    from ebcc.ham.space import Space
    from ebcc.numpy.typing import NDArray
    from ebcc.util import Namespace


class REOM(BaseEOM):
    """Restricted equation-of-motion coupled cluster."""

    # Attributes
    ebcc: REBCC
    space: Space


class IP_REOM(REOM, BaseIP_EOM):
    """Restricted ionisation potential equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(np.abs(diag[: r1.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> float:
        """Get the quasiparticle weight."""
        weight: float = np.dot(r1.ravel(), r1.ravel())
        return astype(weight, float)

    def diag(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the diagonal of the Hamiltonian.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Diagonal of the Hamiltonian.
        """
        parts: Namespace[SpinArrayType] = util.Namespace()

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[:-1]
            parts[f"r{n}"] = self.ebcc.energy_sum(key)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(parts)


class EA_REOM(REOM, BaseEA_EOM):
    """Restricted electron affinity equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(np.abs(diag[: r1.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> float:
        """Get the quasiparticle weight."""
        weight: float = np.dot(r1.ravel(), r1.ravel())
        return astype(weight, float)

    def diag(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the diagonal of the Hamiltonian.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Diagonal of the Hamiltonian.
        """
        parts: Namespace[SpinArrayType] = util.Namespace()

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[n:] + key[: n - 1]
            parts[f"r{n}"] = self.ebcc.energy_sum(key)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(parts)


class EE_REOM(REOM, BaseEE_EOM):
    """Restricted electron-electron equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(diag[: r1.size])
        else:
            arg = np.argsort(diag)
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> float:
        """Get the quasiparticle weight."""
        weight: float = np.dot(r1.ravel(), r1.ravel())
        return astype(weight, float)

    def diag(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the diagonal of the Hamiltonian.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Diagonal of the Hamiltonian.
        """
        parts: Namespace[SpinArrayType] = util.Namespace()

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            parts[f"r{n}"] = -self.ebcc.energy_sum(key)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(parts)
