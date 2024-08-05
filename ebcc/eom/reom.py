"""Restricted equation-of-motion coupled cluster."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import astype, types
from ebcc.eom.base import BaseEA_EOM, BaseEE_EOM, BaseEOM, BaseIP_EOM

if TYPE_CHECKING:
    from typing import Optional

    from ebcc.cc.rebcc import REBCC, AmplitudeType, ERIsInputType
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
            r1 = self.vector_to_amplitudes(diag)[0]
            arg = np.argsort(np.abs(diag[: r1.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: AmplitudeType) -> float:
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
        parts = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[:-1]
            parts.append(self.ebcc.energy_sum(key))

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
        bras = np.array(
            [self.amplitudes_to_vector(*[b[i] for b in bras_raw]) for i in range(self.nmo)]
        )
        return bras

    def kets(self, eris: Optional[ERIsInputType] = None) -> AmplitudeType:
        """Get the ket vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Ket vectors.
        """
        kets_raw = list(self.ebcc.make_ip_mom_kets(eris=eris))
        kets = np.array(
            [self.amplitudes_to_vector(*[k[..., i] for k in kets_raw]) for i in range(self.nmo)]
        )
        return kets

    def moments(
        self,
        nmom: int,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
    ) -> AmplitudeType:
        """Construct the moments of the EOM Hamiltonian.

        Args:
            nmom: Number of moments.
            eris: Electronic repulsion integrals.
            amplitudes: Cluster amplitudes.
            hermitise: Hermitise the moments.

        Returns:
            Moments of the EOM Hamiltonian.
        """
        if eris is None:
            eris = self.ebcc.get_eris()
        if not amplitudes:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        moments: NDArray[float]
        moments = np.zeros((nmom, self.nmo, self.nmo), dtype=types[float])

        for j in range(self.nmo):
            ket = kets[j]
            for n in range(nmom):
                for i in range(self.nmo):
                    bra = bras[i]
                    moments[n, i, j] = self.dot_braket(bra, ket)
                if n != (nmom - 1):
                    ket = self.matvec(ket, eris=eris)

        if hermitise:
            moments = 0.5 * (moments + moments.swapaxes(1, 2))

        return moments


class EA_REOM(REOM, BaseEA_EOM):
    """Restricted electron affinity equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)[0]
            arg = np.argsort(np.abs(diag[: r1.size]))
        else:
            arg = np.argsort(np.abs(diag))
        return arg

    def _quasiparticle_weight(self, r1: AmplitudeType) -> float:
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
        parts = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[n:] + key[: n - 1]
            parts.append(-self.ebcc.energy_sum(key))

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
        bras = np.array(
            [self.amplitudes_to_vector(*[b[i] for b in bras_raw]) for i in range(self.nmo)]
        )
        return bras

    def kets(self, eris: Optional[ERIsInputType] = None) -> AmplitudeType:
        """Get the ket vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Ket vectors.
        """
        kets_raw = list(self.ebcc.make_ea_mom_kets(eris=eris))
        kets = np.array(
            [self.amplitudes_to_vector(*[k[..., i] for k in kets_raw]) for i in range(self.nmo)]
        )
        return kets

    def moments(
        self,
        nmom: int,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
    ) -> AmplitudeType:
        """Construct the moments of the EOM Hamiltonian.

        Args:
            nmom: Number of moments.
            eris: Electronic repulsion integrals.
            amplitudes: Cluster amplitudes.
            hermitise: Hermitise the moments.

        Returns:
            Moments of the EOM Hamiltonian.
        """
        if eris is None:
            eris = self.ebcc.get_eris()
        if not amplitudes:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        moments: NDArray[float]
        moments = np.zeros((nmom, self.nmo, self.nmo), dtype=types[float])

        for j in range(self.nmo):
            ket = kets[j]
            for n in range(nmom):
                for i in range(self.nmo):
                    bra = bras[i]
                    moments[n, i, j] = self.dot_braket(bra, ket)
                if n != (nmom - 1):
                    ket = self.matvec(ket, eris=eris)

        if hermitise:
            moments = 0.5 * (moments + moments.swapaxes(1, 2))

        return moments


class EE_REOM(REOM, BaseEE_EOM):
    """Restricted electron-electron equation-of-motion coupled cluster."""

    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)[0]
            arg = np.argsort(diag[: r1.size])
        else:
            arg = np.argsort(diag)
        return arg

    def _quasiparticle_weight(self, r1: AmplitudeType) -> float:
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
        parts = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            parts.append(-self.ebcc.energy_sum(key))

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
        bras_raw = list(self.ebcc.make_ee_mom_bras(eris=eris))
        bras = np.array(
            [
                [self.amplitudes_to_vector(*[b[i, j] for b in bras_raw]) for j in range(self.nmo)]
                for i in range(self.nmo)
            ]
        )
        return bras

    def kets(self, eris: Optional[ERIsInputType] = None) -> AmplitudeType:
        """Get the ket vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Ket vectors.
        """
        kets_raw = list(self.ebcc.make_ee_mom_kets(eris=eris))
        kets = np.array(
            [
                [
                    self.amplitudes_to_vector(*[k[..., i, j] for k in kets_raw])
                    for j in range(self.nmo)
                ]
                for i in range(self.nmo)
            ]
        )
        return kets

    def moments(
        self,
        nmom: int,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
        diagonal_only: bool = True,
    ) -> AmplitudeType:
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
        if not diagonal_only:
            warnings.warn(
                "Constructing EE moments with `diagonal_only=False` will be very slow.",
                stacklevel=2,
            )

        if eris is None:
            eris = self.ebcc.get_eris()
        if not amplitudes:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        moments: NDArray[float]
        moments = np.zeros((nmom, self.nmo, self.nmo, self.nmo, self.nmo), dtype=types[float])

        for k in range(self.nmo):
            for l in [k] if diagonal_only else range(self.nmo):
                ket = kets[k, l]
                for n in range(nmom):
                    for i in range(self.nmo):
                        for j in [i] if diagonal_only else range(self.nmo):
                            bra = bras[i, j]
                            moments[n, i, j, k, l] = self.dot_braket(bra, ket)
                    if n != (nmom - 1):
                        ket = self.matvec(ket, eris=eris)

        if hermitise:
            moments = 0.5 * (moments + moments.transpose(0, 3, 4, 1, 2))

        return moments
