"""Brueckner orbital self-consistency."""

import dataclasses

import scipy.linalg
from pyscf import lib

from ebcc import NullLogger, init_logging
from ebcc import numpy as np
from ebcc import util
from ebcc.damping import DIIS
from ebcc.logging import ANSI
from ebcc.precision import types


@dataclasses.dataclass
class Options:
    """
    Options for Brueckner CC calculations.

    Attributes
    ----------
    e_tol : float, optional
        Threshold for convergence in the correlation energy. Default value
        is `1e-8`.
    t_tol : float, optional
        Threshold for convergence in the amplitude norm. Default value is
        `1e-8`.
    max_iter : int, optional
        Maximum number of iterations. Default value is `20`.
    diis_space : int, optional
        Number of amplitudes to use in DIIS extrapolation. Default value is
        `12`.
    damping : float, optional
        Damping factor for DIIS extrapolation. Default value is `0.0`.
    """

    e_tol: float = 1e-8
    t_tol: float = 1e-8
    max_iter: int = 20
    diis_space: int = 12
    damping: float = 0.0


class BruecknerREBCC:
    """
    Brueckner orbital self-consistency for coupled cluster calculations.
    Iteratively solve for a new mean-field that presents a vanishing T1
    under the given ansatz.

    Parameters
    ----------
    cc : EBCC
        EBCC coupled cluster object.
    log : logging.Logger, optional
        Log to print output to. Default value is `cc.log`.
    options : dataclasses.dataclass, optional
        Object containing the options. Default value is `Options`.
    **kwargs : dict
        Additional keyword arguments used to update `options`.
    """

    Options = Options

    def __init__(self, cc, log=None, options=None, **kwargs):
        # Options:
        if options is None:
            options = self.Options()
        self.options = options
        for key, val in kwargs.items():
            setattr(self.options, key, val)

        # Parameters:
        self.log = cc.log if log is None else log
        self.mf = cc.mf
        self.cc = cc

        # Attributes:
        self.converged = False

        # Logging:
        init_logging(cc.log)
        cc.log.info(f"\n{ANSI.B}{ANSI.U}{self.name}{ANSI.R}")
        cc.log.info(f"{ANSI.B}Options{ANSI.R}:")
        cc.log.info(f" > e_tol:  {ANSI.y}{self.options.e_tol}{ANSI.R}")
        cc.log.info(f" > t_tol:  {ANSI.y}{self.options.t_tol}{ANSI.R}")
        cc.log.info(f" > max_iter:  {ANSI.y}{self.options.max_iter}{ANSI.R}")
        cc.log.info(f" > diis_space:  {ANSI.y}{self.options.diis_space}{ANSI.R}")
        cc.log.info(f" > damping:  {ANSI.y}{self.options.damping}{ANSI.R}")
        cc.log.debug("")

    def get_rotation_matrix(self, u_tot=None, diis=None, t1=None):
        """
        Update the rotation matrix, and also return the total rotation
        matrix.

        Parameters
        ----------
        u_tot : np.ndarray, optional
            Total rotation matrix. If `None`, then it is assumed to be the
            identity matrix. Default value is `None`.
        diis : DIIS, optional
            DIIS object. If `None`, then DIIS is not used. Default value is
            `None`.
        t1 : np.ndarray, optional
            T1 amplitudes. If `None`, then `cc.t1` is used. Default value
            is `None`.

        Returns
        -------
        u : np.ndarray
            Rotation matrix.
        u_tot : np.ndarray
            Total rotation matrix.
        """

        if t1 is None:
            t1 = self.cc.t1
        if u_tot is None:
            u_tot = np.eye(self.cc.space.ncorr, dtype=types[float])

        t1_block = np.zeros((self.cc.space.ncorr, self.cc.space.ncorr), dtype=types[float])
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

    def transform_amplitudes(self, u, amplitudes=None):
        """
        Transform the amplitudes into the Brueckner orbital basis.

        Parameters
        ----------
        u : np.ndarray
            Rotation matrix.
        amplitudes : Namespace, optional
            Amplitudes. If `None`, then `cc.amplitudes` is used. Default
            value is `None`.

        Returns
        -------
        amplitudes : Namespace
            Rotated amplitudes.
        """

        if amplitudes is None:
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

    def get_t1_norm(self, amplitudes=None):
        """
        Get the norm of the T1 amplitude.

        Parameters
        ----------
        amplitudes : Namespace, optional
            Amplitudes. If `None`, then `cc.amplitudes` is used. Default
            value is `None`.

        Returns
        -------
        t1_norm : float
            Norm of the T1 amplitude.
        """

        if amplitudes is None:
            amplitudes = self.cc.amplitudes

        return np.linalg.norm(amplitudes["t1"])

    def mo_to_correlated(self, mo_coeff):
        """
        For a given set of MO coefficients, return the correlated slice.

        Parameters
        ----------
        mo_coeff : np.ndarray
            MO coefficients.

        Returns
        -------
        mo_coeff_corr : np.ndarray
            Correlated slice of the MO coefficients.
        """

        return mo_coeff[:, self.cc.space.correlated]

    def mo_update_correlated(self, mo_coeff, mo_coeff_corr):
        """
        Update the correlated slice of a set of MO coefficients.

        Parameters
        ----------
        mo_coeff : np.ndarray
            MO coefficients.
        mo_coeff_corr : np.ndarray
            Correlated slice of the MO coefficients.

        Returns
        -------
        mo_coeff : np.ndarray
            Updated MO coefficients.
        """

        mo_coeff[:, self.cc.space.correlated] = mo_coeff_corr

        return mo_coeff

    def update_coefficients(self, u_tot, mo_coeff, mo_coeff_ref):
        """Get the updated coefficients.

        Parameters
        ----------
        u_tot : np.ndarray
            Total rotation matrix.
        mo_coeff : np.ndarray
            New MO coefficients.
        mo_coeff_ref : np.ndarray
            Reference MO coefficients.

        Returns
        -------
        mo_coeff_new : np.ndarray
            Updated MO coefficients.
        u : np.ndarray
            Rotation matrix.
        """
        mo_coeff_new_corr = util.einsum("pi,ij->pj", mo_coeff_ref, u_tot)
        mo_coeff_new = self.mo_update_correlated(mo_coeff, mo_coeff_new_corr)
        return mo_coeff_new

    def kernel(self):
        """
        Run the Bruckner orbital coupled cluster calculation.

        Returns
        -------
        e_cc : float
            Correlation energy.
        """

        # Start a timer:
        timer = util.Timer()

        # Make sure the initial CC calculation is converged:
        if not self.cc.converged:
            with lib.temporary_env(self.cc, log=NullLogger()):
                self.cc.kernel()

        # Set up DIIS:
        diis = DIIS()
        diis.space = self.options.diis_space
        diis.damping = self.options.damping

        # Initialise coefficients:
        mo_coeff_new = np.array(self.cc.mo_coeff, copy=True, dtype=types[float])
        mo_coeff_ref = np.array(self.cc.mo_coeff, copy=True, dtype=types[float])
        mo_coeff_ref = self.mo_to_correlated(mo_coeff_ref)
        u_tot = None

        self.cc.log.output("Solving for Brueckner orbitals.")
        self.cc.log.debug("")
        self.log.info(
            f"{ANSI.B}{'Iter':>4s} {'Energy (corr.)':>16s} {'Energy (tot.)':>18s} "
            f"{'Conv.':>8s} {'Δ(Energy)':>13s} {'|T1|':>13s}{ANSI.R}"
        )
        self.log.info(
            f"{0:4d} {self.cc.e_corr:16.10f} {self.cc.e_tot:18.10f} "
            f"{[ANSI.r, ANSI.g][self.cc.converged]}{self.cc.converged!r:>8}{ANSI.R}"
        )

        converged = False
        for niter in range(1, self.options.max_iter + 1):
            # Update rotation matrix:
            u, u_tot = self.get_rotation_matrix(u_tot=u_tot, diis=diis)

            # Update MO coefficients:
            mo_coeff_new = self.update_coefficients(u_tot, mo_coeff_new, mo_coeff_ref)

            # Transform mean-field and amplitudes:
            self.mf.mo_coeff = mo_coeff_new
            self.mf.e_tot = self.mf.energy_tot()
            amplitudes = self.transform_amplitudes(u)

            # Run CC calculation:
            e_prev = self.cc.e_tot
            with lib.temporary_env(self.cc, log=NullLogger()):
                self.cc.__init__(
                    self.mf,
                    log=self.cc.log,
                    ansatz=self.cc.ansatz,
                    space=self.cc.space,
                    omega=self.cc.omega,
                    g=self.cc.bare_g,
                    G=self.cc.bare_G,
                    options=self.cc.options,
                )
                self.cc.amplitudes = amplitudes
                self.cc.kernel()
            de = abs(e_prev - self.cc.e_tot)
            dt = self.get_t1_norm()

            # Log the iteration:
            converged_e = de < self.options.e_tol
            converged_t = dt < self.options.t_tol
            self.log.info(
                f"{niter:4d} {self.cc.e_corr:16.10f} {self.cc.e_tot:18.10f}"
                f" {[ANSI.r, ANSI.g][self.cc.converged]}{self.cc.converged!r:>8}{ANSI.R}"
                f" {[ANSI.r, ANSI.g][converged_e]}{de:13.3e}{ANSI.R}"
                f" {[ANSI.r, ANSI.g][converged_t]}{dt:13.3e}{ANSI.R}"
            )

            # Check for convergence:
            converged = converged_e and converged_t
            if converged:
                self.log.debug("")
                self.log.output(f"{ANSI.g}Converged.{ANSI.R}")
                break
        else:
            self.log.debug("")
            self.log.warning(f"{ANSI.r}Failed to converge.{ANSI.R}")

        self.cc.log.debug("")
        self.cc.log.output("E(corr) = %.10f", self.cc.e_corr)
        self.cc.log.output("E(tot)  = %.10f", self.cc.e_tot)
        self.cc.log.debug("")
        self.cc.log.debug("Time elapsed: %s", timer.format_time(timer()))
        self.cc.log.debug("")

        return self.cc.e_corr

    @property
    def name(self):
        """Get a string representation of the method name."""
        return self.spin_type + "B" + self.cc.ansatz.name

    @property
    def spin_type(self):
        """Return the spin type."""
        return self.cc.spin_type


@util.has_docstring
class BruecknerUEBCC(BruecknerREBCC, metaclass=util.InheritDocstrings):
    @util.has_docstring
    def get_rotation_matrix(self, u_tot=None, diis=None, t1=None):
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

    @util.has_docstring
    def transform_amplitudes(self, u, amplitudes=None):
        if amplitudes is None:
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

    @util.has_docstring
    def get_t1_norm(self, amplitudes=None):
        if amplitudes is None:
            amplitudes = self.cc.amplitudes

        norm_a = np.linalg.norm(amplitudes["t1"].aa)
        norm_b = np.linalg.norm(amplitudes["t1"].bb)

        return np.linalg.norm([norm_a, norm_b])

    @util.has_docstring
    def mo_to_correlated(self, mo_coeff):
        return (
            mo_coeff[0][:, self.cc.space[0].correlated],
            mo_coeff[1][:, self.cc.space[1].correlated],
        )

    @util.has_docstring
    def mo_update_correlated(self, mo_coeff, mo_coeff_corr):
        mo_coeff[0][:, self.cc.space[0].correlated] = mo_coeff_corr[0]
        mo_coeff[1][:, self.cc.space[1].correlated] = mo_coeff_corr[1]

        return mo_coeff

    @util.has_docstring
    def update_coefficients(self, u_tot, mo_coeff, mo_coeff_ref):
        mo_coeff_new_corr = (
            util.einsum("pi,ij->pj", mo_coeff_ref[0], u_tot.aa),
            util.einsum("pi,ij->pj", mo_coeff_ref[1], u_tot.bb),
        )
        mo_coeff_new = self.mo_update_correlated(mo_coeff, mo_coeff_new_corr)
        return mo_coeff_new


@util.has_docstring
class BruecknerGEBCC(BruecknerREBCC):
    pass
