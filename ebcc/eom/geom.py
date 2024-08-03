"""Generalised equation-of-motion coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.eom.base import BaseEA_EOM, BaseEE_EOM, BaseEOM, BaseIP_EOM, Options
from ebcc.precision import types

if TYPE_CHECKING:
    from typing import Optional, Tuple, Union

    from ebcc.cc.gebcc import AmplitudeType, ERIsInputType
    from ebcc.numpy.typing import NDArray
    from ebcc.util import Namespace


class GEOM(BaseEOM):
    """Generalised equation-of-motion coupled cluster."""

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
        return np.linalg.norm(r1) ** 2

    def moments(
        self,
        nmom: int,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Namespace[AmplitudeType] = None,
        hermitise: bool = True,
    ) -> NDArray[float]:
        """Construct the moments of the EOM Hamiltonian."""
        if eris is None:
            eris = self.ebcc.get_eris()
        if not amplitudes:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

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


class IP_GEOM(BaseIP_EOM, BaseEOM):
    """Generalised ionisation potential equation-of-motion coupled cluster."""

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


class EA_GEOM(BaseEA_EOM, BaseEOM):
    """Generalised electron affinity equation-of-motion coupled cluster."""

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


class EE_GEOM(BaseEE_EOM, BaseEOM):
    """Generalised electron-electron equation-of-motion coupled cluster."""

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
        kets_raw = list(self.ebcc.make_ee_mom_kets(eris=eris))
        kets = np.array(
            [self.amplitudes_to_vector(*[k[..., i] for k in kets_raw]) for i in range(self.nmo)]
        )
        return kets
