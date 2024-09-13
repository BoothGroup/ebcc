"""Restricted equation-of-motion coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import astype
from ebcc.eom.base import BaseEA_EOM, BaseEE_EOM, BaseEOM, BaseIP_EOM

if TYPE_CHECKING:
    from typing import Optional

    from numpy import float64, int64
    from numpy.typing import NDArray

    from ebcc.cc.rebcc import REBCC, ERIsInputType, SpinArrayType
    from ebcc.ham.space import Space
    from ebcc.util import Namespace

    T = float64


class REOM(BaseEOM):
    """Restricted equation-of-motion coupled cluster."""

    # Attributes
    ebcc: REBCC
    space: Space


class IP_REOM(REOM, BaseIP_EOM):
    """Restricted ionisation potential equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[T]) -> NDArray[int64]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(np.abs(diag[: r1.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> T:
        """Get the quasiparticle weight."""
        return np.vdot(r1, r1)

    def diag(self, eris: Optional[ERIsInputType] = None) -> NDArray[T]:
        """Get the diagonal of the Hamiltonian.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Diagonal of the Hamiltonian.
        """
        parts: Namespace[SpinArrayType] = util.Namespace()

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ip"
        ):
            parts[name] = self.ebcc.energy_sum(key)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ip"):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(parts)

    def amplitudes_to_vector(self, amplitudes: Namespace[SpinArrayType]) -> NDArray[T]:
        """Construct a vector containing all of the IP-EOM amplitudes.

        Args:
            amplitudes: IP-EOM amplitudes.

        Returns:
            IP-EOM amplitudes as a vector.
        """
        vectors = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ip"
        ):
            vectors.append(amplitudes[name].ravel())

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ip"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ip"
        ):
            raise util.ModelNotImplemented

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector: NDArray[T]) -> Namespace[SpinArrayType]:
        """Construct a namespace of IP-EOM amplitudes from a vector.

        Args:
            vector: IP-EOM amplitudes as a vector.

        Returns:
            IP-EOM amplitudes.
        """
        amplitudes: Namespace[SpinArrayType] = util.Namespace()
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ip"
        ):
            shape = tuple(self.space.size(k) for k in key)
            size = util.prod(shape)
            amplitudes[name] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ip"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ip"
        ):
            raise util.ModelNotImplemented

        return amplitudes


class EA_REOM(REOM, BaseEA_EOM):
    """Restricted electron affinity equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[T]) -> NDArray[int64]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(np.abs(diag[: r1.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> T:
        """Get the quasiparticle weight."""
        return np.vdot(r1, r1)

    def diag(self, eris: Optional[ERIsInputType] = None) -> NDArray[T]:
        """Get the diagonal of the Hamiltonian.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Diagonal of the Hamiltonian.
        """
        parts: Namespace[SpinArrayType] = util.Namespace()

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ea"
        ):
            parts[name] = self.ebcc.energy_sum(key)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ea"):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(parts)

    def amplitudes_to_vector(self, amplitudes: Namespace[SpinArrayType]) -> NDArray[T]:
        """Construct a vector containing all of the EA-EOM amplitudes.

        Args:
            amplitudes: EA-EOM amplitudes.

        Returns:
            EA-EOM amplitudes as a vector.
        """
        vectors = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ea"
        ):
            vectors.append(amplitudes[name].ravel())

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ea"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ea"
        ):
            raise util.ModelNotImplemented

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector: NDArray[T]) -> Namespace[SpinArrayType]:
        """Construct a namespace of EA-EOM amplitudes from a vector.

        Args:
            vector: EA-EOM amplitudes as a vector.

        Returns:
            EA-EOM amplitudes.
        """
        amplitudes: Namespace[SpinArrayType] = util.Namespace()
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ea"
        ):
            shape = tuple(self.space.size(k) for k in key)
            size = util.prod(shape)
            amplitudes[name] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ea"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ea"
        ):
            raise util.ModelNotImplemented

        return amplitudes


class EE_REOM(REOM, BaseEE_EOM):
    """Restricted electron-electron equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[T]) -> NDArray[int64]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(diag[: r1.size])
        else:
            arg = np.argsort(diag)
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> T:
        """Get the quasiparticle weight."""
        return np.vdot(r1, r1)

    def diag(self, eris: Optional[ERIsInputType] = None) -> NDArray[T]:
        """Get the diagonal of the Hamiltonian.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Diagonal of the Hamiltonian.
        """
        parts: Namespace[SpinArrayType] = util.Namespace()

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ee"
        ):
            parts[name] = -self.ebcc.energy_sum(key)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ee"):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(parts)

    def amplitudes_to_vector(self, amplitudes: Namespace[SpinArrayType]) -> NDArray[T]:
        """Construct a vector containing all of the EE-EOM amplitudes.

        Args:
            amplitudes: EE-EOM amplitudes.

        Returns:
            EE-EOM amplitudes as a vector.
        """
        vectors = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ee"
        ):
            vectors.append(amplitudes[name].ravel())

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ee"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ee"
        ):
            raise util.ModelNotImplemented

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector: NDArray[T]) -> Namespace[SpinArrayType]:
        """Construct a namespace of EE-EOM amplitudes from a vector.

        Args:
            vector: EE-EOM amplitudes as a vector.

        Returns:
            EE-EOM amplitudes.
        """
        amplitudes: Namespace[SpinArrayType] = util.Namespace()
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ee"
        ):
            shape = tuple(self.space.size(k) for k in key)
            size = util.prod(shape)
            amplitudes[name] = vector[i0 : i0 + size].reshape(shape)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ee"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ee"
        ):
            raise util.ModelNotImplemented

        return amplitudes
