"""Unrestricted orbital-optimised coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.linalg

from ebcc import numpy as np
from ebcc import util
from ebcc.backend import _inflate, _put, ensure_scalar
from ebcc.core.precision import astype, types
from ebcc.opt.base import BaseBruecknerEBCC, BaseOptimisedEBCC, _BaseOptimisedEBCC

if TYPE_CHECKING:
    from typing import Optional

    from numpy import floating
    from numpy.typing import NDArray

    from ebcc.cc.uebcc import UEBCC, SpinArrayType, ERIsInputType
    from ebcc.core.damping import BaseDamping
    from ebcc.util import Namespace

    T = floating


class _OptimisedUEBCC(_BaseOptimisedEBCC):
    """Unrestricted orbital-optimised coupled cluster."""

    # Attributes
    cc: UEBCC

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

        nocc = (self.cc.space[0].ncocc, self.cc.space[1].ncocc)
        ci = {"a": u.aa[: nocc[0], : nocc[0]], "b": u.bb[: nocc[1], : nocc[1]]}
        ca = {"a": u.aa[nocc[0] :, nocc[0] :], "b": u.bb[nocc[1] :, nocc[1] :]}
        if is_lambda:
            ci, ca = ca, ci

        # Transform T amplitudes:
        for name, key, n in self.cc.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            if is_lambda:
                name = name.replace("t", "l")
            for comb in util.generate_spin_combinations(n, unique=True):
                args = [getattr(amplitudes[name], comb), tuple(range(n * 2))]
            for i in range(n):
                args += [ci[comb[i]], (i, i + n * 2)]
            for i in range(n):
                args += [ca[comb[i + n]], (i + n, i + n * 3)]
            args += [tuple(range(n * 2, n * 4))]
            setattr(amplitudes[name], comb, util.einsum(*args))

        # Transform S amplitudes:
        for name, key, n in self.cc.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented  # TODO

        # Transform U amplitudes:
        for name, key, nf, nb in self.cc.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented  # TODO

        return amplitudes

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


class BruecknerUEBCC(BaseBruecknerEBCC, _OptimisedUEBCC):
    """Unrestricted Brueckner-orbital coupled cluster."""

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
            u_tot = util.Namespace(
                aa=np.eye(self.cc.space[0].ncorr, dtype=types[float]),
                bb=np.eye(self.cc.space[1].ncorr, dtype=types[float]),
            )
        if t1 is None:
            t1 = self.cc.t1

        # Get the T1 blocks
        t1_block: Namespace[NDArray[T]] = util.Namespace()
        zocc = np.zeros((self.cc.space[0].ncocc, self.cc.space[0].ncocc))
        zvir = np.zeros((self.cc.space[0].ncvir, self.cc.space[0].ncvir))
        t1_block.aa = np.block([[zocc, -t1.aa], [np.transpose(t1.aa), zvir]])
        zocc = np.zeros((self.cc.space[1].ncocc, self.cc.space[1].ncocc))
        zvir = np.zeros((self.cc.space[1].ncvir, self.cc.space[1].ncvir))
        t1_block.bb = np.block([[zocc, -t1.bb], [np.transpose(t1.bb), zvir]])

        # Get the orbital gradient
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
            [np.ravel(scipy.linalg.logm(u_tot.aa)), np.ravel(scipy.linalg.logm(u_tot.bb))], axis=0
        )
        a: NDArray[T] = np.asarray(np.real(a), dtype=types[float])
        if damping is not None:
            xerr = np.concatenate([t1.aa.ravel(), t1.bb.ravel()])
            a = damping(a, error=xerr)

        u_tot.aa = scipy.linalg.expm(np.reshape(a[: u_tot.aa.size], u_tot.aa.shape))
        u_tot.bb = scipy.linalg.expm(np.reshape(a[u_tot.aa.size :], u_tot.bb.shape))

        return u, u_tot


class OptimisedUEBCC(BaseOptimisedEBCC, _OptimisedUEBCC):
    """Unrestricted optimised coupled cluster."""

    def get_grad_norm(self, u: SpinArrayType) -> T:
        """Get the norm of the gradient.

        Args:
            u: Rotation matrix.

        Returns:
            Norm of the gradient.
        """
        grad_a = np.linalg.norm(scipy.linalg.logm(u.aa))
        grad_b = np.linalg.norm(scipy.linalg.logm(u.bb))
        grad: T = (grad_a**2 + grad_b**2) ** 0.5
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
            u_tot = util.Namespace(
                aa=np.eye(self.cc.space[0].ncorr, dtype=types[float]),
                bb=np.eye(self.cc.space[1].ncorr, dtype=types[float]),
            )
        if rdm1 is None:
            rdm1 = self.cc.make_rdm1_f(eris=eris)
        if rdm2 is None:
            rdm2 = self.cc.make_rdm2_f(eris=eris)
        eris = self.cc.get_eris(eris=eris)

        # Get one-particle contribution to generalised Fock matrix
        mo_coeff = util.Namespace(
            a=self.cc.mo_coeff[0][:, self.cc.space[0].correlated],
            b=self.cc.mo_coeff[1][:, self.cc.space[1].correlated],
        )
        h1e_ao: NDArray[T] = np.asarray(self.cc.mf.get_hcore(), dtype=types[float])
        h1e = util.Namespace(
            aa=util.einsum("pq,pi,qj->ij", h1e_ao, mo_coeff.a, mo_coeff.a),
            bb=util.einsum("pq,pi,qj->ij", h1e_ao, mo_coeff.b, mo_coeff.b),
        )
        f = util.Namespace(
            aa=np.dot(h1e.aa, rdm1.aa),
            bb=np.dot(h1e.bb, rdm1.bb),
        )

        # Get two-particle contribution to generalised Fock matrix
        h2e = util.Namespace(
            aaaa=eris.aaaa.xxxx,
            aabb=eris.aabb.xxxx,
            bbbb=eris.bbbb.xxxx,
        )
        f.aa += util.einsum("prst,stqr->pq", h2e.aaaa, rdm2.aaaa)
        f.aa += util.einsum("prst,qrst->pq", h2e.aabb, rdm2.aabb)
        f.bb += util.einsum("prst,stqr->pq", h2e.bbbb, rdm2.bbbb)
        f.bb += util.einsum("stpr,stqr->pq", h2e.aabb, rdm2.aabb)

        # Get the orbital gradient
        oa, va = self.cc.space[0].xslice("o"), self.cc.space[0].xslice("v")
        ob, vb = self.cc.space[1].xslice("o"), self.cc.space[1].xslice("v")
        x_vo = util.Namespace(
            aa=(f.aa - np.transpose(f.aa))[va, oa] / self.cc.energy_sum("vo", "aa"),
            bb=(f.bb - np.transpose(f.bb))[vb, ob] / self.cc.energy_sum("vo", "bb"),
        )
        x = util.Namespace(
            aa=_inflate(
                (self.cc.space[0].ncorr, self.cc.space[0].ncorr),
                np.ix_(self.cc.space[0].xmask("v"), self.cc.space[0].xmask("o")),  # type: ignore
                x_vo.aa,
            ),
            bb=_inflate(
                (self.cc.space[1].ncorr, self.cc.space[1].ncorr),
                np.ix_(self.cc.space[1].xmask("v"), self.cc.space[1].xmask("o")),  # type: ignore
                x_vo.bb,
            ),
        )
        u = util.Namespace(
            aa=scipy.linalg.expm(x.aa - np.transpose(x.aa)),
            bb=scipy.linalg.expm(x.bb - np.transpose(x.bb)),
        )

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
            [np.ravel(scipy.linalg.logm(u_tot.aa)), np.ravel(scipy.linalg.logm(u_tot.bb))], axis=0
        )
        a: NDArray[T] = np.asarray(np.real(a), dtype=types[float])
        if damping is not None:
            a = damping(a)

        u_tot.aa = scipy.linalg.expm(np.reshape(a[: u_tot.aa.size], u_tot.aa.shape))
        u_tot.bb = scipy.linalg.expm(np.reshape(a[u_tot.aa.size :], u_tot.bb.shape))

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
        h1e = util.Namespace(
            aa=util.einsum(
                "pq,pi,qj->ij", self.cc.mf.get_hcore(), self.cc.mo_coeff[0], self.cc.mo_coeff[0]
            ),
            bb=util.einsum(
                "pq,pi,qj->ij", self.cc.mf.get_hcore(), self.cc.mo_coeff[1], self.cc.mo_coeff[1]
            ),
        )
        h2e = util.Namespace(
            aaaa=eris.aaaa.xxxx,
            aabb=eris.aabb.xxxx,
            bbbb=eris.bbbb.xxxx,
        )

        # Get the energy
        e_tot = util.einsum("pq,qp->", h1e.aa, rdm1.aa)
        e_tot += util.einsum("pq,qp->", h1e.bb, rdm1.bb)
        e_tot += util.einsum("pqrs,pqrs->", h2e.aaaa, rdm2.aaaa) * 0.5
        e_tot += util.einsum("pqrs,pqrs->", h2e.aabb, rdm2.aabb)
        e_tot += util.einsum("pqrs,pqrs->", h2e.bbbb, rdm2.bbbb) * 0.5
        e_tot += self.cc.mf.energy_nuc()

        res: float = np.real(ensure_scalar(e_tot))
        return astype(res, float)
