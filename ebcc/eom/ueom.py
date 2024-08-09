"""Unrestricted equation-of-motion coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import astype
from ebcc.eom.base import BaseEA_EOM, BaseEE_EOM, BaseEOM, BaseIP_EOM

if TYPE_CHECKING:
    from typing import Optional

    from ebcc.cc.uebcc import UEBCC, ERIsInputType, SpinArrayType
    from ebcc.ham.space import Space
    from ebcc.numpy.typing import NDArray
    from ebcc.util import Namespace


class UEOM(BaseEOM):
    """Unrestricted equation-of-motion coupled cluster."""

    # Attributes
    ebcc: UEBCC
    space: tuple[Space, Space]


class IP_UEOM(UEOM, BaseIP_EOM):
    """Unrestricted ionisation potential equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(np.abs(diag[: r1.a.size + r1.b.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> float:
        """Get the quasiparticle weight."""
        weight: float = np.dot(r1.a.ravel(), r1.a.ravel()) + np.dot(r1.b.ravel(), r1.b.ravel())
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
            spin_part: SpinArrayType = util.Namespace()
            for comb in util.generate_spin_combinations(n, excited=True):
                spin_part[comb] = self.ebcc.energy_sum(key, comb)
            parts[f"r{n}"] = spin_part

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(parts)


class EA_UEOM(UEOM, BaseEA_EOM):
    """Unrestricted electron affinity equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(np.abs(diag[: r1.a.size + r1.b.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> float:
        """Get the quasiparticle weight."""
        weight: float = np.dot(r1.a.ravel(), r1.a.ravel()) + np.dot(r1.b.ravel(), r1.b.ravel())
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
            spin_part: SpinArrayType = util.Namespace()
            for comb in util.generate_spin_combinations(n, excited=True):
                spin_part[comb] = -self.ebcc.energy_sum(key, comb)
            parts[f"r{n}"] = spin_part

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(parts)


class EE_UEOM(UEOM, BaseEE_EOM):
    """Unrestricted electron-electron equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(diag[: r1.aa.size + r1.bb.size])
        else:
            arg = np.argsort(diag)
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> float:
        """Get the quasiparticle weight."""
        weight: float = np.dot(r1.aa.ravel(), r1.aa.ravel()) + np.dot(r1.bb.ravel(), r1.bb.ravel())
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
            spin_part: SpinArrayType = util.Namespace()
            for comb in util.generate_spin_combinations(n):
                spin_part[comb] = self.ebcc.energy_sum(key, comb)
            parts[f"r{n}"] = spin_part

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(parts)
