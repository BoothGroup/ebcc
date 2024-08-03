"""Unrestricted Brueckner-orbital coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.linalg

from ebcc import numpy as np
from ebcc import util
from ebcc.opt.base import BaseBruecknerEBCC
from ebcc.precision import types

if TYPE_CHECKING:
    from typing import Optional

    from ebcc.cc.uebcc import AmplitudeType
    from ebcc.damping import DIIS
    from ebcc.util import Namespace


class BruecknerUEBCC(BaseBruecknerEBCC):
    """Unrestricted Brueckner-orbital coupled cluster.

    Attributes:
        cc: Parent `BaseEBCC` object.
        options: Options for the EOM calculation.
    """

    def get_rotation_matrix(
        self,
        u_tot: Optional[AmplitudeType] = None,
        diis: Optional[DIIS] = None,
        t1: Optional[AmplitudeType] = None,
    ) -> tuple[AmplitudeType, AmplitudeType]:
        """Update the rotation matrix.

        Also returns the total rotation matrix.

        Args:
            u_tot: Total rotation matrix.
            diis: DIIS object.
            t1: T1 amplitude.

        Returns:
            Rotation matrix and total rotation matrix.
        """
        if t1 is None:
            t1 = self.cc.t1
        if u_tot is None:
            u_tot = util.Namespace(
                aa=np.eye(self.cc.space[0].ncorr, dtype=types[float]),
                bb=np.eye(self.cc.space[1].ncorr, dtype=types[float]),
            )

        t1_block = util.Namespace(
            aa=np.zeros((self.cc.space[0].ncorr, self.cc.space[0].ncorr), dtype=types[float]),
            bb=np.zeros((self.cc.space[1].ncorr, self.cc.space[1].ncorr), dtype=types[float]),
        )
        t1_block.aa[: self.cc.space[0].ncocc, self.cc.space[0].ncocc :] = -t1.aa
        t1_block.aa[self.cc.space[0].ncocc :, : self.cc.space[0].ncocc] = t1.aa.T
        t1_block.bb[: self.cc.space[1].ncocc, self.cc.space[1].ncocc :] = -t1.bb
        t1_block.bb[self.cc.space[1].ncocc :, : self.cc.space[1].ncocc] = t1.bb.T

        u = util.Namespace(
            aa=scipy.linalg.expm(t1_block.aa),
            bb=scipy.linalg.expm(t1_block.bb),
        )

        u_tot.aa = np.dot(u_tot.aa, u.aa)
        u_tot.bb = np.dot(u_tot.bb, u.bb)
        if scipy.linalg.det(u_tot.aa) < 0:
            u_tot.aa[:, 0] *= -1
        if scipy.linalg.det(u_tot.bb) < 0:
            u_tot.bb[:, 0] *= -1

        a = np.concatenate(
            [
                scipy.linalg.logm(u_tot.aa).ravel(),
                scipy.linalg.logm(u_tot.bb).ravel(),
            ],
            axis=0,
        )
        a = a.real.astype(types[float])
        if diis is not None:
            xerr = np.concatenate([t1.aa.ravel(), t1.bb.ravel()])
            a = diis.update(a, xerr=xerr)

        u_tot.aa = scipy.linalg.expm(a[: u_tot.aa.size].reshape(u_tot.aa.shape))
        u_tot.bb = scipy.linalg.expm(a[u_tot.aa.size :].reshape(u_tot.bb.shape))

        return u, u_tot

    def transform_amplitudes(
        self, u: AmplitudeType, amplitudes: Optional[Namespace[AmplitudeType]] = None
    ) -> Namespace[AmplitudeType]:
        """Transform the amplitudes into the Brueckner orbital basis.

        Args:
            u: Rotation matrix.
            amplitudes: Cluster amplitudes.

        Returns:
            Transformed cluster amplitudes.
        """
        if not amplitudes:
            amplitudes = self.cc.amplitudes

        nocc = (self.cc.space[0].ncocc, self.cc.space[1].ncocc)
        ci = {"a": u.aa[: nocc[0], : nocc[0]], "b": u.bb[: nocc[1], : nocc[1]]}
        ca = {"a": u.aa[nocc[0] :, nocc[0] :], "b": u.bb[nocc[1] :, nocc[1] :]}

        # Transform T amplitudes:
        for name, key, n in self.cc.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            for comb in util.generate_spin_combinations(n, unique=True):
                args = [getattr(self.cc.amplitudes[name], comb), tuple(range(n * 2))]
            for i in range(n):
                args += [ci[comb[i]], (i, i + n * 2)]
            for i in range(n):
                args += [ca[comb[i + n]], (i + n, i + n * 3)]
            args += [tuple(range(n * 2, n * 4))]
            setattr(self.cc.amplitudes[name], comb, util.einsum(*args))

        # Transform S amplitudes:
        for name, key, n in self.cc.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented  # TODO

        # Transform U amplitudes:
        for name, key, nf, nb in self.cc.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented  # TODO

        return self.cc.amplitudes

    def get_t1_norm(self, amplitudes: Optional[Namespace[AmplitudeType]] = None) -> float:
        """Get the norm of the T1 amplitude.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Norm of the T1 amplitude.
        """
        if not amplitudes:
            amplitudes = self.cc.amplitudes
        norm_a = np.linalg.norm(amplitudes["t1"].aa)
        norm_b = np.linalg.norm(amplitudes["t1"].bb)
        return np.linalg.norm([norm_a, norm_b])

    def mo_to_correlated(self, mo_coeff: tuple[NDArray[float]]) -> tuple[NDArray[float]]:
        """Transform the MO coefficients into the correlated basis.

        Args:
            mo_coeff: MO coefficients.

        Returns:
            Correlated slice of MO coefficients.
        """
        return (
            mo_coeff[0][:, self.cc.space[0].correlated],
            mo_coeff[1][:, self.cc.space[1].correlated],
        )

    def mo_update_correlated(
        self, mo_coeff: tuple[NDArray[float]], mo_coeff_corr: tuple[NDArray[float]]
    ) -> tuple[NDArray[float]]:
        """Update the correlated slice of a set of MO coefficients.

        Args:
            mo_coeff: MO coefficients.
            mo_coeff_corr: Correlated slice of MO coefficients.

        Returns:
            Updated MO coefficients.
        """
        mo_coeff[0][:, self.cc.space[0].correlated] = mo_coeff_corr[0]
        mo_coeff[1][:, self.cc.space[1].correlated] = mo_coeff_corr[1]
        return mo_coeff

    def update_coefficients(
        self,
        u_tot: AmplitudeType,
        mo_coeff: tuple[NDArray[float]],
        mo_coeff_ref: tuple[NDArray[float]],
    ) -> tuple[NDArray[float]]:
        """Update the MO coefficients.

        Args:
            u_tot: Total rotation matrix.
            mo_coeff: New MO coefficients.
            mo_coeff_ref: Reference MO coefficients.

        Returns:
            Updated MO coefficients.
        """
        mo_coeff_new_corr = (
            util.einsum("pi,ij->pj", mo_coeff_ref[0], u_tot.aa),
            util.einsum("pi,ij->pj", mo_coeff_ref[1], u_tot.bb),
        )
        mo_coeff_new = self.mo_update_correlated(mo_coeff, mo_coeff_new_corr)
        return mo_coeff_new