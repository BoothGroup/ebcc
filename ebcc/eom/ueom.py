"""Unrestricted equation-of-motion coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import types, astype
from ebcc.eom.base import BaseEA_EOM, BaseEE_EOM, BaseEOM, BaseIP_EOM

if TYPE_CHECKING:
    from typing import Optional

    from ebcc.ham.space import Space
    from ebcc.cc.uebcc import UEBCC, AmplitudeType, ERIsInputType
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
            r1 = self.vector_to_amplitudes(diag)[0]
            arg = np.argsort(np.abs(diag[: r1.a.size + r1.b.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: AmplitudeType) -> float:
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
        parts = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[:-1]
            spin_part: AmplitudeType = util.Namespace()
            for comb in util.generate_spin_combinations(n, excited=True):
                spin_part[comb] = self.ebcc.energy_sum(key, comb)
            parts.append(spin_part)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    def bras(self, eris: Optional[ERIsInputType] = None) -> AmplitudeType:
        """Get the bra vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Bra vectors.
        """
        bras_raw = list(self.ebcc.make_ip_mom_bras(eris=eris))
        bras_tmp: Namespace[list[NDArray[float]]] = util.Namespace(a=[], b=[])

        for i in range(self.nmo):
            amps_a: list[AmplitudeType] = []
            amps_b: list[AmplitudeType] = []

            m = 0
            for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
                amp_a: AmplitudeType = util.Namespace()
                amp_b: AmplitudeType = util.Namespace()
                for spin in util.generate_spin_combinations(n, excited=True):
                    shape = tuple(self.space["ab".index(s)].ncocc for s in spin[:n]) + tuple(
                        self.space["ab".index(s)].ncvir for s in spin[n:]
                    )
                    setattr(
                        amp_a,
                        spin,
                        getattr(bras_raw[m], "a" + spin, {i: np.zeros(shape, dtype=types[float])})[
                            i
                        ],
                    )
                    setattr(
                        amp_b,
                        spin,
                        getattr(bras_raw[m], "b" + spin, {i: np.zeros(shape, dtype=types[float])})[
                            i
                        ],
                    )
                amps_a.append(amp_a)
                amps_b.append(amp_b)
                m += 1

            for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
                raise util.ModelNotImplemented

            for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
                raise util.ModelNotImplemented

            bras_tmp.a.append(self.amplitudes_to_vector(*amps_a))
            bras_tmp.b.append(self.amplitudes_to_vector(*amps_b))

        bras: Namespace[NDArray[float]] = util.Namespace(a=np.array(bras_tmp.a), b=np.array(bras_tmp.b))

        return bras

    def kets(self, eris: Optional[ERIsInputType] = None) -> AmplitudeType:
        """Get the ket vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Ket vectors.
        """
        kets_raw = list(self.ebcc.make_ip_mom_kets(eris=eris))
        kets_tmp: Namespace[list[NDArray[float]]] = util.Namespace(a=[], b=[])

        for i in range(self.nmo):
            j = (Ellipsis, i)
            amps_a: list[AmplitudeType] = []
            amps_b: list[AmplitudeType] = []

            m = 0
            for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
                amp_a: AmplitudeType = util.Namespace()
                amp_b: AmplitudeType = util.Namespace()
                for spin in util.generate_spin_combinations(n, excited=True):
                    shape = tuple(self.space["ab".index(s)].ncocc for s in spin[:n]) + tuple(
                        self.space["ab".index(s)].ncvir for s in spin[n:]
                    )
                    setattr(
                        amp_a,
                        spin,
                        getattr(kets_raw[m], spin + "a", {j: np.zeros(shape, dtype=types[float])})[
                            j
                        ],
                    )
                    setattr(
                        amp_b,
                        spin,
                        getattr(kets_raw[m], spin + "b", {j: np.zeros(shape, dtype=types[float])})[
                            j
                        ],
                    )
                amps_a.append(amp_a)
                amps_b.append(amp_b)
                m += 1

            for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
                raise util.ModelNotImplemented

            for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
                raise util.ModelNotImplemented

            kets_tmp.a.append(self.amplitudes_to_vector(*amps_a))
            kets_tmp.b.append(self.amplitudes_to_vector(*amps_b))

        kets: Namespace[NDArray[float]] = util.Namespace(a=np.array(kets_tmp.a), b=np.array(kets_tmp.b))

        return kets

    def moments(
        self,
        nmom: int,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
    ) -> NDArray[float]:
        """Construct the moments of the EOM Hamiltonian.

        Args:
            nmom: Number of moments.
            eris: Electronic repulsion integrals.
            amplitudes: Cluster amplitudes.
            hermitise: Hermitise the moments.

        Returns:
            Moments of the EOM Hamiltonian.
        """
        eris = self.ebcc.get_eris(eris)
        if not amplitudes:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        moments: Namespace[NDArray[float]]
        moments = util.Namespace(
            aa=np.zeros((nmom, self.nmo, self.nmo), dtype=types[float]),
            bb=np.zeros((nmom, self.nmo, self.nmo), dtype=types[float]),
        )

        for spin in util.generate_spin_combinations(1):
            for j in range(self.nmo):
                ket = getattr(kets, spin[1])[j]
                for n in range(nmom):
                    for i in range(self.nmo):
                        bra = getattr(bras, spin[0])[i]
                        getattr(moments, spin)[n, i, j] = self.dot_braket(bra, ket)
                    if n != (nmom - 1):
                        ket = self.matvec(ket, eris=eris)

            if hermitise:
                t = getattr(moments, spin)
                setattr(moments, spin, 0.5 * (t + t.swapaxes(1, 2)))

        return moments


class EA_UEOM(UEOM, BaseEA_EOM):
    """Unrestricted electron affinity equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)[0]
            arg = np.argsort(np.abs(diag[: r1.a.size + r1.b.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: AmplitudeType) -> float:
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
        parts = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[n:] + key[: n - 1]
            spin_part: AmplitudeType = util.Namespace()
            for comb in util.generate_spin_combinations(n, excited=True):
                spin_part[comb] = -self.ebcc.energy_sum(key, comb)
            parts.append(spin_part)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    def bras(self, eris: Optional[ERIsInputType] = None) -> AmplitudeType:
        """Get the bra vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Bra vectors.
        """
        bras_raw = list(self.ebcc.make_ea_mom_bras(eris=eris))
        bras_tmp: Namespace[list[NDArray[float]]] = util.Namespace(a=[], b=[])

        for i in range(self.nmo):
            amps_a: list[AmplitudeType] = []
            amps_b: list[AmplitudeType] = []

            m = 0
            for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
                amp_a: AmplitudeType = util.Namespace()
                amp_b: AmplitudeType = util.Namespace()
                for spin in util.generate_spin_combinations(n, excited=True):
                    shape = tuple(self.space["ab".index(s)].ncvir for s in spin[:n]) + tuple(
                        self.space["ab".index(s)].ncocc for s in spin[n:]
                    )
                    setattr(
                        amp_a,
                        spin,
                        getattr(bras_raw[m], "a" + spin, {i: np.zeros(shape, dtype=types[float])})[
                            i
                        ],
                    )
                    setattr(
                        amp_b,
                        spin,
                        getattr(bras_raw[m], "b" + spin, {i: np.zeros(shape, dtype=types[float])})[
                            i
                        ],
                    )
                amps_a.append(amp_a)
                amps_b.append(amp_b)
                m += 1

            for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
                raise util.ModelNotImplemented

            for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
                raise util.ModelNotImplemented

            bras_tmp.a.append(self.amplitudes_to_vector(*amps_a))
            bras_tmp.b.append(self.amplitudes_to_vector(*amps_b))

        bras: Namespace[NDArray[float]] = util.Namespace(a=np.array(bras_tmp.a), b=np.array(bras_tmp.b))

        return bras

    def kets(self, eris: Optional[ERIsInputType] = None) -> AmplitudeType:
        """Get the ket vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Ket vectors.
        """
        kets_raw = list(self.ebcc.make_ea_mom_kets(eris=eris))
        kets_tmp: Namespace[list[NDArray[float]]] = util.Namespace(a=[], b=[])

        for i in range(self.nmo):
            j = (Ellipsis, i)
            amps_a: list[AmplitudeType] = []
            amps_b: list[AmplitudeType] = []

            m = 0
            for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
                amp_a: AmplitudeType = util.Namespace()
                amp_b: AmplitudeType = util.Namespace()
                for spin in util.generate_spin_combinations(n, excited=True):
                    shape = tuple(self.space["ab".index(s)].ncvir for s in spin[:n]) + tuple(
                        self.space["ab".index(s)].ncocc for s in spin[n:]
                    )
                    setattr(
                        amp_a,
                        spin,
                        getattr(kets_raw[m], spin + "a", {j: np.zeros(shape, dtype=types[float])})[
                            j
                        ],
                    )
                    setattr(
                        amp_b,
                        spin,
                        getattr(kets_raw[m], spin + "b", {j: np.zeros(shape, dtype=types[float])})[
                            j
                        ],
                    )
                amps_a.append(amp_a)
                amps_b.append(amp_b)
                m += 1

            for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
                raise util.ModelNotImplemented

            for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
                raise util.ModelNotImplemented

            kets_tmp.a.append(self.amplitudes_to_vector(*amps_a))
            kets_tmp.b.append(self.amplitudes_to_vector(*amps_b))

        kets: Namespace[NDArray[float]] = util.Namespace(a=np.array(kets_tmp.a), b=np.array(kets_tmp.b))

        return kets

    def moments(
        self,
        nmom: int,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
    ) -> NDArray[float]:
        """Construct the moments of the EOM Hamiltonian.

        Args:
            nmom: Number of moments.
            eris: Electronic repulsion integrals.
            amplitudes: Cluster amplitudes.
            hermitise: Hermitise the moments.

        Returns:
            Moments of the EOM Hamiltonian.
        """
        eris = self.ebcc.get_eris(eris)
        if not amplitudes:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        moments: Namespace[NDArray[float]]
        moments = util.Namespace(
            aa=np.zeros((nmom, self.nmo, self.nmo), dtype=types[float]),
            bb=np.zeros((nmom, self.nmo, self.nmo), dtype=types[float]),
        )

        for spin in util.generate_spin_combinations(1):
            for j in range(self.nmo):
                ket = getattr(kets, spin[1])[j]
                for n in range(nmom):
                    for i in range(self.nmo):
                        bra = getattr(bras, spin[0])[i]
                        getattr(moments, spin)[n, i, j] = self.dot_braket(bra, ket)
                    if n != (nmom - 1):
                        ket = self.matvec(ket, eris=eris)

            if hermitise:
                t = getattr(moments, spin)
                setattr(moments, spin, 0.5 * (t + t.swapaxes(1, 2)))

        return moments


class EE_UEOM(UEOM, BaseEE_EOM):
    """Unrestricted electron-electron equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)[0]
            arg = np.argsort(diag[: r1.aa.size + r1.bb.size])
        else:
            arg = np.argsort(diag)
        return arg

    def _quasiparticle_weight(self, r1: AmplitudeType) -> float:
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
        parts = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            spin_part: AmplitudeType = util.Namespace()
            for comb in util.generate_spin_combinations(n):
                spin_part[comb] = self.ebcc.energy_sum(key, comb)
            parts.append(spin_part)

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return self.amplitudes_to_vector(*parts)

    def bras(self, eris: Optional[ERIsInputType] = None) -> AmplitudeType:
        """Get the bra vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Bra vectors.
        """
        raise util.ModelNotImplemented("EE moments for UEBCC not working.")

    def kets(self, eris: Optional[ERIsInputType] = None) -> AmplitudeType:
        """Get the ket vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Ket vectors.
        """
        raise util.ModelNotImplemented("EE moments for UEBCC not working.")

    def moments(
        self,
        nmom: int,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
        diagonal_only: bool = True,
    ) -> NDArray[float]:
        """Construct the moments of the EOM Hamiltonian.

        Args:
            nmom: Number of moments.
            eris: Electronic repulsion integrals.
            amplitudes: Cluster amplitudes.
            hermitise: Hermitise the moments.
            diagonal_only: Only compute the diagonal elements.

        Returns:
            Moments of the EOM Hamiltonian.
        """
        raise util.ModelNotImplemented("EE moments for UEBCC not working.")
