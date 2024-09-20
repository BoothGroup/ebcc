"""Generalised equation-of-motion coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.eom.base import BaseEA_EOM, BaseEE_EOM, BaseEOM, BaseIP_EOM

if TYPE_CHECKING:
    from typing import Optional

    from numpy import float64, int64
    from numpy.typing import NDArray

    from ebcc.cc.gebcc import GEBCC, ERIsInputType, SpinArrayType
    from ebcc.ham.space import Space
    from ebcc.util import Namespace

    T = float64


class GEOM(BaseEOM):
    """Generalised equation-of-motion coupled cluster."""

    # Attributes
    ebcc: GEBCC
    space: Space


class IP_GEOM(GEOM, BaseIP_EOM):
    """Generalised ionisation potential equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[T]) -> NDArray[int64]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = util.argsort(np.abs(diag[: r1.size]))
        else:
            arg = util.argsort(np.abs(diag))
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
            vectors.append(np.ravel(util.compress_axes(key, amplitudes[name])))

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
            size = util.get_compressed_size(key, **{k: self.space.size(k) for k in set(key)})
            shape = tuple(self.space.size(k) for k in key)
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(key, vn_tril, shape=shape)
            amplitudes[name] = vn
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ip"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ip"
        ):
            raise util.ModelNotImplemented

        return amplitudes


class EA_GEOM(GEOM, BaseEA_EOM):
    """Generalised electron affinity equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[T]) -> NDArray[int64]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = util.argsort(np.abs(diag[: r1.size]))
        else:
            arg = util.argsort(np.abs(diag))
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
            parts[name] = -self.ebcc.energy_sum(key)

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
            vectors.append(np.ravel(util.compress_axes(key, amplitudes[name])))

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
            size = util.get_compressed_size(key, **{k: self.space.size(k) for k in set(key)})
            shape = tuple(self.space.size(k) for k in key)
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(key, vn_tril, shape=shape)
            amplitudes[name] = vn
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ea"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ea"
        ):
            raise util.ModelNotImplemented

        return amplitudes


class EE_GEOM(GEOM, BaseEE_EOM):
    """Generalised electron-electron equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[T]) -> NDArray[int64]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = util.argsort(diag[: r1.size])
        else:
            arg = util.argsort(diag)
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
            vectors.append(np.ravel(util.compress_axes(key, amplitudes[name])))

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
            size = util.get_compressed_size(key, **{k: self.space.size(k) for k in set(key)})
            shape = tuple(self.space.size(k) for k in key)
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(key, vn_tril, shape=shape)
            amplitudes[name] = vn
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ee"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ee"
        ):
            raise util.ModelNotImplemented

        return amplitudes
