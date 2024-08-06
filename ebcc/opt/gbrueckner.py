"""Generalised Brueckner-orbital coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.linalg

from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import types
from ebcc.opt.base import BaseBruecknerEBCC

if TYPE_CHECKING:
    from typing import Optional

    from ebcc.cc.gebcc import GEBCC, SpinArrayType
    from ebcc.core.damping import DIIS
    from ebcc.numpy.typing import NDArray
    from ebcc.util import Namespace


class BruecknerGEBCC(BaseBruecknerEBCC):
    """Generalised Brueckner-orbital coupled cluster.

    Attributes:
        cc: Parent `BaseEBCC` object.
        options: Options for the EOM calculation.
    """

    # Attributes
    cc: GEBCC

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
            u_tot = np.eye(self.cc.space.ncorr, dtype=types[float])

        t1_block: NDArray[float] = np.zeros(
            (self.cc.space.ncorr, self.cc.space.ncorr), dtype=types[float]
        )
        t1_block[: self.cc.space.ncocc, self.cc.space.ncocc :] = -t1
        t1_block[self.cc.space.ncocc :, : self.cc.space.ncocc] = t1.T

        u = scipy.linalg.expm(t1_block)

        u_tot = np.dot(u_tot, u)
        if scipy.linalg.det(u_tot) < 0:
            u_tot[:, 0] *= -1

        a = scipy.linalg.logm(u_tot)
        a = a.real.astype(types[float])
        if diis is not None:
            a = diis.update(a, xerr=t1)

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
            args = [self.cc.amplitudes[name], tuple(range(n * 2))]
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

    def get_t1_norm(self, amplitudes: Optional[Namespace[SpinArrayType]] = None) -> float:
        """Get the norm of the T1 amplitude.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Norm of the T1 amplitude.
        """
        if not amplitudes:
            amplitudes = self.cc.amplitudes
        weight: float = types[float](np.linalg.norm(amplitudes["t1"]))
        return weight

    def mo_to_correlated(self, mo_coeff: NDArray[float]) -> NDArray[float]:
        """Transform the MO coefficients into the correlated basis.

        Args:
            mo_coeff: MO coefficients.

        Returns:
            Correlated slice of MO coefficients.
        """
        return mo_coeff[:, self.cc.space.correlated]

    def mo_update_correlated(
        self, mo_coeff: NDArray[float], mo_coeff_corr: NDArray[float]
    ) -> NDArray[float]:
        """Update the correlated slice of a set of MO coefficients.

        Args:
            mo_coeff: MO coefficients.
            mo_coeff_corr: Correlated slice of MO coefficients.

        Returns:
            Updated MO coefficients.
        """
        mo_coeff[:, self.cc.space.correlated] = mo_coeff_corr
        return mo_coeff

    def update_coefficients(
        self, u_tot: SpinArrayType, mo_coeff: NDArray[float], mo_coeff_ref: NDArray[float]
    ) -> NDArray[float]:
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
