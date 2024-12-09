"""Restricted Brueckner-orbital coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.linalg

from ebcc import numpy as np
from ebcc import util
from ebcc.backend import _put
from ebcc.core.precision import types
from ebcc.opt.base import BaseBruecknerEBCC

if TYPE_CHECKING:
    from typing import Optional, Union

    from numpy import floating
    from numpy.typing import NDArray

    from ebcc.cc.rebcc import REBCC, SpinArrayType
    from ebcc.core.damping import BaseDamping
    from ebcc.util import Namespace

    T = floating


class BruecknerREBCC(BaseBruecknerEBCC):
    """Restricted Brueckner-orbital coupled cluster."""

    # Attributes
    cc: REBCC

    def get_rotation_matrix(
        self,
        u_tot: Optional[SpinArrayType] = None,
        damping: Optional[BaseDamping] = None,
        t1: Optional[SpinArrayType] = None,
    ) -> tuple[SpinArrayType, SpinArrayType]:
        """Update the rotation matrix.

        Also returns the total rotation matrix.

        Args:
            u_tot: Total rotation matrix.
            damping: Damping object.
            t1: T1 amplitude.

        Returns:
            Rotation matrix and total rotation matrix.
        """
        if t1 is None:
            t1 = self.cc.t1
        if u_tot is None:
            u_tot = np.eye(self.cc.space.ncorr, dtype=types[float])

        zocc = np.zeros((self.cc.space.ncocc, self.cc.space.ncocc))
        zvir = np.zeros((self.cc.space.ncvir, self.cc.space.ncvir))
        t1_block: NDArray[T] = np.block([[zocc, -t1], [np.transpose(t1), zvir]])

        u = scipy.linalg.expm(t1_block)

        u_tot = u_tot @ u
        if np.linalg.det(u_tot) < 0:
            u_tot = _put(u_tot, np.ix_(np.arange(u_tot.shape[0]), np.array([0])), -u_tot[:, 0])

        a: NDArray[T] = np.asarray(np.real(scipy.linalg.logm(u_tot)), dtype=types[float])
        if damping is not None:
            a = damping(a, error=t1)

        u_tot = scipy.linalg.expm(a)

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

        nocc = self.cc.space.ncocc
        ci = u[:nocc, :nocc]
        ca = u[nocc:, nocc:]

        # Transform T amplitudes:
        for name, key, n in self.cc.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            args: list[Union[SpinArrayType, tuple[int, ...]]] = [
                self.cc.amplitudes[name],
                tuple(range(n * 2)),
            ]
            for i in range(n):
                args += [ci, (i, i + n * 2)]
            for i in range(n):
                args += [ca, (i + n, i + n * 3)]
            args += [tuple(range(n * 2, n * 4))]
            self.cc.amplitudes[name] = util.einsum(*args)

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
        weight: T = np.linalg.norm(amplitudes["t1"])
        return weight

    def mo_to_correlated(self, mo_coeff: NDArray[T]) -> NDArray[T]:
        """Transform the MO coefficients into the correlated basis.

        Args:
            mo_coeff: MO coefficients.

        Returns:
            Correlated slice of MO coefficients.
        """
        return mo_coeff[:, self.cc.space.correlated]

    def mo_update_correlated(self, mo_coeff: NDArray[T], mo_coeff_corr: NDArray[T]) -> NDArray[T]:
        """Update the correlated slice of a set of MO coefficients.

        Args:
            mo_coeff: MO coefficients.
            mo_coeff_corr: Correlated slice of MO coefficients.

        Returns:
            Updated MO coefficients.
        """
        mo_coeff = _put(
            mo_coeff,
            np.ix_(np.arange(mo_coeff.shape[0]), self.cc.space.correlated),  # type: ignore
            mo_coeff_corr,
        )
        return mo_coeff

    def update_coefficients(
        self, u_tot: SpinArrayType, mo_coeff: NDArray[T], mo_coeff_ref: NDArray[T]
    ) -> NDArray[T]:
        """Update the MO coefficients.

        Args:
            u_tot: Total rotation matrix.
            mo_coeff: New MO coefficients.
            mo_coeff_ref: Reference MO coefficients.

        Returns:
            Updated MO coefficients.
        """
        mo_coeff_new_corr = util.einsum("pi,ij->pj", mo_coeff_ref, u_tot)
        mo_coeff_new = self.mo_update_correlated(mo_coeff, mo_coeff_new_corr)
        return mo_coeff_new
