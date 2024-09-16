"""Unrestricted Brueckner-orbital coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.linalg

from ebcc import numpy as np
from ebcc import util
from ebcc.backend import _put
from ebcc.core.precision import types
from ebcc.opt.base import BaseBruecknerEBCC

if TYPE_CHECKING:
    from typing import Optional

    from numpy import float64
    from numpy.typing import NDArray

    from ebcc.cc.uebcc import UEBCC, SpinArrayType
    from ebcc.core.damping import DIIS
    from ebcc.util import Namespace

    T = float64


class BruecknerUEBCC(BaseBruecknerEBCC):
    """Unrestricted Brueckner-orbital coupled cluster."""

    # Attributes
    cc: UEBCC

    def get_rotation_matrix(
        self,
        u_tot: Optional[SpinArrayType] = None,
        diis: Optional[DIIS] = None,
        t1: Optional[SpinArrayType] = None,
    ) -> tuple[SpinArrayType, SpinArrayType]:
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

        t1_block: Namespace[NDArray[T]] = util.Namespace()
        zocc = np.zeros((self.cc.space[0].ncocc, self.cc.space[0].ncocc))
        zvir = np.zeros((self.cc.space[0].ncvir, self.cc.space[0].ncvir))
        t1_block.aa = np.block([[zocc, -t1.aa], [t1.aa.T, zvir]])
        zocc = np.zeros((self.cc.space[1].ncocc, self.cc.space[1].ncocc))
        zvir = np.zeros((self.cc.space[1].ncvir, self.cc.space[1].ncvir))
        t1_block.bb = np.block([[zocc, -t1.bb], [t1.bb.T, zvir]])

        u = util.Namespace(aa=scipy.linalg.expm(t1_block.aa), bb=scipy.linalg.expm(t1_block.bb))

        u_tot.aa = u_tot.aa @ u.aa
        u_tot.bb = u_tot.bb @ u.bb
        if np.linalg.det(u_tot.aa) < 0:
            u_tot.aa = _put(
                u_tot.aa, np.ix_(np.arange(u_tot.aa.shape[0]), np.array([0])), -u_tot.aa[:, 0]
            )
        if np.linalg.det(u_tot.bb) < 0:
            u_tot.bb = _put(
                u_tot.bb, np.ix_(np.arange(u_tot.aa.shape[0]), np.array([0])), -u_tot.bb[:, 0]
            )

        a = np.concatenate(
            [scipy.linalg.logm(u_tot.aa).ravel(), scipy.linalg.logm(u_tot.bb).ravel()], axis=0
        )
        a = a.real.astype(types[float])
        if diis is not None:
            xerr = np.concatenate([t1.aa.ravel(), t1.bb.ravel()])
            a = diis.update(a, xerr=xerr)

        u_tot.aa = scipy.linalg.expm(a[: u_tot.aa.size].reshape(u_tot.aa.shape))
        u_tot.bb = scipy.linalg.expm(a[u_tot.aa.size :].reshape(u_tot.bb.shape))

        return u, u_tot

    def transform_amplitudes(
        self, u: SpinArrayType, amplitudes: Optional[Namespace[SpinArrayType]] = None
    ) -> Namespace[SpinArrayType]:
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

    def get_t1_norm(self, amplitudes: Optional[Namespace[SpinArrayType]] = None) -> T:
        """Get the norm of the T1 amplitude.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Norm of the T1 amplitude.
        """
        if not amplitudes:
            amplitudes = self.cc.amplitudes
        weight_a = np.linalg.norm(amplitudes["t1"].aa)
        weight_b = np.linalg.norm(amplitudes["t1"].bb)
        weight: T = (weight_a**2 + weight_b**2) ** 0.5
        return weight

    def mo_to_correlated(
        self, mo_coeff: tuple[NDArray[T], NDArray[T]]
    ) -> tuple[NDArray[T], NDArray[T]]:
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
        self,
        mo_coeff: tuple[NDArray[T], NDArray[T]],
        mo_coeff_corr: tuple[NDArray[T], NDArray[T]],
    ) -> tuple[NDArray[T], NDArray[T]]:
        """Update the correlated slice of a set of MO coefficients.

        Args:
            mo_coeff: MO coefficients.
            mo_coeff_corr: Correlated slice of MO coefficients.

        Returns:
            Updated MO coefficients.
        """
        space = self.cc.space
        mo_coeff = (
            _put(
                mo_coeff[0],
                np.ix_(np.arange(mo_coeff[0].shape[0]), space[0].correlated),  # type: ignore
                mo_coeff_corr[0],
            ),
            _put(
                mo_coeff[1],
                np.ix_(np.arange(mo_coeff[1].shape[0]), space[1].correlated),  # type: ignore
                mo_coeff_corr[1],
            ),
        )
        return mo_coeff

    def update_coefficients(
        self,
        u_tot: SpinArrayType,
        mo_coeff: tuple[NDArray[T], NDArray[T]],
        mo_coeff_ref: tuple[NDArray[T], NDArray[T]],
    ) -> tuple[NDArray[T], NDArray[T]]:
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
