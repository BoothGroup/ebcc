"""Unrestricted equation-of-motion coupled cluster."""

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

    from ebcc.cc.uebcc import UEBCC, ERIsInputType, SpinArrayType
    from ebcc.ham.space import Space
    from ebcc.util import Namespace

    T = float64


class UEOM(BaseEOM):
    """Unrestricted equation-of-motion coupled cluster."""

    # Attributes
    ebcc: UEBCC
    space: tuple[Space, Space]


class IP_UEOM(UEOM, BaseIP_EOM):
    """Unrestricted ionisation potential equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[T]) -> NDArray[int64]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(np.abs(diag[: r1.a.size + r1.b.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> T:
        """Get the quasiparticle weight."""
        return np.vdot(r1.a, r1.a) + np.vdot(r1.b, r1.b)

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
            spin_part: SpinArrayType = util.Namespace()
            for comb in util.generate_spin_combinations(n, excited=True):
                spin_part[comb] = self.ebcc.energy_sum(key, comb)
            parts[name] = spin_part

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
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                vn = amplitudes[name][spin]
                subscript, _ = util.combine_subscripts(key, spin)
                vectors.append(util.compress_axes(subscript, vn).ravel())

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
        sizes: dict[tuple[str, ...], int] = {
            (o, s): self.space[i].size(o) for o in "ovOVia" for i, s in enumerate("ab")
        }

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ip"
        ):
            amp: SpinArrayType = util.Namespace()
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                subscript, csizes = util.combine_subscripts(key, spin, sizes=sizes)
                size = util.get_compressed_size(subscript, **csizes)
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                amp[spin] = vn
                i0 += size

            amplitudes[name] = amp

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ip"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ip"
        ):
            raise util.ModelNotImplemented

        assert i0 == len(vector)

        return amplitudes


class EA_UEOM(UEOM, BaseEA_EOM):
    """Unrestricted electron affinity equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[T]) -> NDArray[int64]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(np.abs(diag[: r1.a.size + r1.b.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> T:
        """Get the quasiparticle weight."""
        return np.vdot(r1.a, r1.a) + np.vdot(r1.b, r1.b)

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
            spin_part: SpinArrayType = util.Namespace()
            for comb in util.generate_spin_combinations(n, excited=True):
                spin_part[comb] = -self.ebcc.energy_sum(key, comb)
            parts[name] = spin_part

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
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                vn = amplitudes[name][spin]
                subscript, _ = util.combine_subscripts(key, spin)
                vectors.append(util.compress_axes(subscript, vn).ravel())

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
        sizes: dict[tuple[str, ...], int] = {
            (o, s): self.space[i].size(o) for o in "ovOVia" for i, s in enumerate("ab")
        }

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ea"
        ):
            amp: SpinArrayType = util.Namespace()
            for spin in util.generate_spin_combinations(n, excited=True, unique=True):
                subscript, csizes = util.combine_subscripts(key, spin, sizes=sizes)
                size = util.get_compressed_size(subscript, **csizes)
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                amp[spin] = vn
                i0 += size

            amplitudes[name] = amp

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ea"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ea"
        ):
            raise util.ModelNotImplemented

        assert i0 == len(vector)

        return amplitudes


class EE_UEOM(UEOM, BaseEE_EOM):
    """Unrestricted electron-electron equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[T]) -> NDArray[int64]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)["r1"]
            arg = np.argsort(diag[: r1.aa.size + r1.bb.size])
        else:
            arg = np.argsort(diag)
        return arg

    def _quasiparticle_weight(self, r1: SpinArrayType) -> T:
        """Get the quasiparticle weight."""
        return np.vdot(r1.aa, r1.aa) + np.vdot(r1.bb, r1.bb)

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
            spin_part: SpinArrayType = util.Namespace()
            for comb in util.generate_spin_combinations(n, unique=True):
                spin_part[comb] = self.ebcc.energy_sum(key, comb)
            parts[name] = spin_part

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
            for spin in util.generate_spin_combinations(n, unique=True):
                vn = amplitudes[name][spin]
                subscript, _ = util.combine_subscripts(key, spin)
                vectors.append(util.compress_axes(subscript, vn).ravel())

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
        sizes: dict[tuple[str, ...], int] = {
            (o, s): self.space[i].size(o) for o in "ovOVia" for i, s in enumerate("ab")
        }

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="ee"
        ):
            amp: SpinArrayType = util.Namespace()
            for spin in util.generate_spin_combinations(n, unique=True):
                subscript, csizes = util.combine_subscripts(key, spin, sizes=sizes)
                size = util.get_compressed_size(subscript, **csizes)
                shape = tuple(self.space["ab".index(s)].size(k) for s, k in zip(spin, key))
                vn_tril = vector[i0 : i0 + size]
                factor = max(
                    spin[:n].count(s) for s in set(spin[:n])
                )  # FIXME why? untested for n > 2
                vn = util.decompress_axes(subscript, vn_tril, shape=shape) / factor
                amp[spin] = vn
                i0 += size

            amplitudes[name] = amp

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="ee"):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="ee"
        ):
            raise util.ModelNotImplemented

        assert i0 == len(vector)

        return amplitudes
