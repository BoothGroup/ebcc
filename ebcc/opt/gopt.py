"""Generalised orbital-optimised coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.linalg

from ebcc import numpy as np
from ebcc import util
from ebcc.backend import _inflate, _put, ensure_scalar
from ebcc.core.precision import astype, types
from ebcc.opt.base import BaseBruecknerEBCC, BaseOptimisedEBCC, _BaseOptimisedEBCC

if TYPE_CHECKING:
    from typing import Optional, Union

    from numpy import floating
    from numpy.typing import NDArray

    from ebcc.cc.gebcc import GEBCC, SpinArrayType, ERIsInputType
    from ebcc.core.damping import BaseDamping
    from ebcc.util import Namespace

    T = floating


class _OptimisedGEBCC(_BaseOptimisedEBCC):
    """Generalised orbital-optimised coupled cluster."""

    # Attributes
    cc: GEBCC

    def transform_amplitudes(
        self,
        u: SpinArrayType,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        is_lambda: bool = False,
    ) -> Namespace[SpinArrayType]:
        """Transform the amplitudes into the orbital-optimised basis.

        Args:
            u: Rotation matrix.
            amplitudes: Cluster amplitudes.
            is_lambda: Whether the amplitudes are Lambda amplitudes.

        Returns:
            Transformed cluster amplitudes.
        """
        if not amplitudes and not is_lambda:
            amplitudes = self.cc.amplitudes
        elif not amplitudes:
            amplitudes = self.cc.lambdas
        amplitudes = amplitudes.copy()

        nocc = self.cc.space.ncocc
        ci = u[:nocc, :nocc]
        ca = u[nocc:, nocc:]
        if is_lambda:
            ci, ca = ca, ci

        # Transform T amplitudes:
        for name, key, n in self.cc.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            if is_lambda:
                name = name.replace("t", "l")
            args: list[Union[tuple[int, ...], NDArray[T]]] = [
                amplitudes[name],
                tuple(range(n * 2)),
            ]
            for i in range(n):
                args += [ci, (i, i + n * 2)]
            for i in range(n):
                args += [ca, (i + n, i + n * 3)]
            args += [tuple(range(n * 2, n * 4))]
            amplitudes[name] = util.einsum(*args)

        # Transform S amplitudes:
        for name, key, n in self.cc.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented  # TODO

        # Transform U amplitudes:
        for name, key, nf, nb in self.cc.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented  # TODO

        return amplitudes

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


class BruecknerGEBCC(BaseBruecknerEBCC, _OptimisedGEBCC):
    """Generalised Brueckner-orbital coupled cluster."""

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
            t1: T1 amplitudes.

        Returns:
            Rotation matrix and total rotation matrix.
        """
        if u_tot is None:
            u_tot = np.eye(self.cc.space.ncorr, dtype=types[float])
        if t1 is None:
            t1 = self.cc.t1

        # Get the T1 blocks
        zocc = np.zeros((self.cc.space.ncocc, self.cc.space.ncocc))
        zvir = np.zeros((self.cc.space.ncvir, self.cc.space.ncvir))
        t1_block: NDArray[T] = np.block([[zocc, -t1], [np.transpose(t1), zvir]])

        # Get the orbital gradient
        u = scipy.linalg.expm(t1_block)

        u_tot = u_tot @ u
        if np.linalg.det(u_tot) < 0:
            u_tot = _put(u_tot, np.ix_(np.arange(u_tot.shape[0]), np.array([0])), -u_tot[:, 0])

        a: NDArray[T] = np.asarray(np.real(scipy.linalg.logm(u_tot)), dtype=types[float])
        if damping is not None:
            a = damping(a, error=t1)

        u_tot = scipy.linalg.expm(a)

        return u, u_tot


class OptimisedGEBCC(BaseOptimisedEBCC, _OptimisedGEBCC):
    """Generalised optimised coupled cluster."""

    def get_grad_norm(self, u: SpinArrayType) -> T:
        """Get the norm of the gradient.

        Args:
            u: Rotation matrix.

        Returns:
            Norm of the gradient.
        """
        grad: T = np.linalg.norm(scipy.linalg.logm(u))
        return grad

    def get_rotation_matrix(
        self,
        u_tot: Optional[SpinArrayType] = None,
        damping: Optional[BaseDamping] = None,
        eris: Optional[ERIsInputType] = None,
        rdm1: Optional[SpinArrayType] = None,
        rdm2: Optional[SpinArrayType] = None,
    ) -> tuple[SpinArrayType, SpinArrayType]:
        """Update the rotation matrix.

        Also returns the total rotation matrix.

        Args:
            u_tot: Total rotation matrix.
            damping: Damping object.
            eris: Electronic repulsion integrals.
            rdm1: One-particle reduced density matrix.
            rdm2: Two-particle reduced density matrix.

        Returns:
            Rotation matrix and total rotation matrix.
        """
        if u_tot is None:
            u_tot = np.eye(self.cc.space.ncorr, dtype=types[float])
        if rdm1 is None:
            rdm1 = self.cc.make_rdm1_f(eris=eris)
        if rdm2 is None:
            rdm2 = self.cc.make_rdm2_f(eris=eris)
        eris = self.cc.get_eris(eris=eris)

        # Get one-particle contribution to generalised Fock matrix
        mo_coeff = self.cc.mo_coeff[:, self.cc.space.correlated]
        h1e_ao: NDArray[T] = np.asarray(self.cc.mf.get_hcore(), dtype=types[float])
        h1e = util.einsum("pq,pi,qj->ij", h1e_ao, mo_coeff, mo_coeff)
        f = np.dot(h1e, rdm1) * 0.5

        # Get two-particle contribution to generalised Fock matrix
        h2e = eris.xxxx
        f += util.einsum("prst,stqr->pq", h2e, rdm2) * 0.5

        # Get the orbital gradient
        o, v = self.cc.space.xslice("o"), self.cc.space.xslice("v")
        x_vo = (f - np.transpose(f))[v, o] / self.cc.energy_sum("vo")
        x = _inflate(
            (self.cc.space.ncorr, self.cc.space.ncorr),
            np.ix_(self.cc.space.xmask("v"), self.cc.space.xmask("o")),  # type: ignore
            x_vo,
        )
        u = scipy.linalg.expm(x - np.transpose(x))

        u_tot = u_tot @ u
        if np.linalg.det(u_tot) < 0:
            u_tot = _put(u_tot, np.ix_(np.arange(u_tot.shape[0]), np.array([0])), -u_tot[:, 0])

        a: NDArray[T] = np.asarray(np.real(scipy.linalg.logm(u_tot)), dtype=types[float])
        if damping is not None:
            a = damping(a)

        u_tot = scipy.linalg.expm(a)

        return u, u_tot

    def energy(
        self,
        eris: Optional[ERIsInputType] = None,
        rdm1: Optional[SpinArrayType] = None,
        rdm2: Optional[SpinArrayType] = None,
    ) -> float:
        """Calculate the energy.

        Args:
            eris: Electron repulsion integrals.
            rdm1: One-particle reduced density matrix.
            rdm2: Two-particle reduced density matrix.

        Returns:
            Total energy.
        """
        if rdm1 is None:
            rdm1 = self.cc.make_rdm1_f(eris=eris)
        if rdm2 is None:
            rdm2 = self.cc.make_rdm2_f(eris=eris)
        eris = self.cc.get_eris(eris=eris)

        # Get the Hamiltonian
        h1e = util.einsum(
            "pq,pi,qj->ij", self.cc.mf.get_hcore(), self.cc.mo_coeff, self.cc.mo_coeff
        )
        h2e = eris.xxxx

        # Get the energy
        e_tot = util.einsum("pq,qp->", h1e, rdm1)
        e_tot += util.einsum("pqrs,pqrs->", h2e, rdm2) * 0.5
        e_tot += self.cc.mf.energy_nuc()

        res: float = np.real(ensure_scalar(e_tot))
        return astype(res, float)
