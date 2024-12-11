"""Base classes for `ebcc.opt`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy  # FIXME for pyscfad
from pyscf import lib

from ebcc import numpy as np
from ebcc import util
from ebcc.core.damping import DIIS
from ebcc.core.logging import ANSI, NullLogger, init_logging
from ebcc.core.precision import types
from ebcc.util import _BaseOptions

if TYPE_CHECKING:
    from typing import Any, Optional

    from numpy import floating
    from numpy.typing import NDArray

    from ebcc.cc.base import BaseEBCC, SpinArrayType, ERIsInputType
    from ebcc.core.damping import BaseDamping
    from ebcc.util import Namespace

    T = floating

# FIXME Custom versions of PySCF functions
# FIXME Care with precision for PySCF


@dataclass
class BaseOptions(_BaseOptions):
    """Options for Brueckner-orbital and orbital-optimised calculations.

    Args:
        e_tol: Threshold for converged in the correlation energy.
        t_tol: Threshold for converged in the amplitude norm.
        max_iter: Maximum number of iterations.
        diis_space: Number of amplitudes to use in DIIS extrapolation.
        diis_min_space: Minimum number of amplitudes to use in DIIS extrapolation.
        damping: Damping factor for DIIS extrapolation.
    """

    e_tol: float = 1e-8
    t_tol: float = 1e-7
    max_iter: int = 20
    diis_space: int = 9
    diis_min_space: int = 1
    damping: float = 0.0


class _BaseOptimisedEBCC(ABC):
    """Base class for orbital-optimised coupled cluster approaches."""

    # Types
    Options: type[BaseOptions] = BaseOptions
    Damping: type[BaseDamping] = DIIS

    # Attributes
    cc: BaseEBCC

    def __init__(
        self,
        cc: BaseEBCC,
        options: Optional[BaseOptions] = None,
        **kwargs: Any,
    ) -> None:
        r"""Initialise the orbital-optimised EBCC object.

        Args:
            cc: Parent `EBCC` object.
            options: Options for the EOM calculation.
            **kwargs: Additional keyword arguments used to update `options`.
        """
        # Options:
        if options is None:
            options = self.Options()
        self.options = options
        for key, val in kwargs.items():
            setattr(self.options, key, val)

        # Parameters:
        self.cc = cc
        self.mf = cc.mf
        self.space = cc.space
        self.log = cc.log

        # Attributes:
        self.converged = False

        # Logging:
        init_logging(cc.log)
        cc.log.info(f"\n{ANSI.B}{ANSI.U}{self.name}{ANSI.R}")
        cc.log.debug(f"{ANSI.B}{'*' * len(self.name)}{ANSI.R}")
        cc.log.debug("")
        cc.log.info(f"{ANSI.B}Options{ANSI.R}:")
        cc.log.info(f" > e_tol:  {ANSI.y}{self.options.e_tol}{ANSI.R}")
        cc.log.info(f" > t_tol:  {ANSI.y}{self.options.t_tol}{ANSI.R}")
        cc.log.info(f" > max_iter:  {ANSI.y}{self.options.max_iter}{ANSI.R}")
        cc.log.info(f" > diis_space:  {ANSI.y}{self.options.diis_space}{ANSI.R}")
        cc.log.info(f" > diis_min_space:  {ANSI.y}{self.options.diis_min_space}{ANSI.R}")
        cc.log.info(f" > damping:  {ANSI.y}{self.options.damping}{ANSI.R}")
        cc.log.debug("")

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the method."""
        pass

    @property
    def spin_type(self) -> str:
        """Get the spin type."""
        return self.cc.spin_type

    @abstractmethod
    def kernel(self) -> float:
        """Run the orbital-optimised coupled cluster calculation.

        Returns:
            Correlation energy.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def mo_to_correlated(self, mo_coeff: Any) -> Any:
        """Transform the MO coefficients into the correlated basis.

        Args:
            mo_coeff: MO coefficients.

        Returns:
            Correlated slice of MO coefficients.
        """
        pass

    @abstractmethod
    def mo_update_correlated(self, mo_coeff: Any, mo_coeff_corr: Any) -> Any:
        """Update the correlated slice of a set of MO coefficients.

        Args:
            mo_coeff: MO coefficients.
            mo_coeff_corr: Correlated slice of MO coefficients.

        Returns:
            Updated MO coefficients.
        """
        pass

    @abstractmethod
    def update_coefficients(
        self, u_tot: SpinArrayType, mo_coeff_new: Any, mo_coeff_ref: Any
    ) -> Any:
        """Update the MO coefficients.

        Args:
            u_tot: Total rotation matrix.
            mo_coeff_new: New MO coefficients.
            mo_coeff_ref: Reference MO coefficients.

        Returns:
            Updated MO coefficients.
        """
        pass


class BaseBruecknerEBCC(_BaseOptimisedEBCC):
    """Base class for Brueckner-orbital coupled cluster."""

    @property
    def name(self) -> str:
        """Get the name of the method."""
        return f"{self.spin_type}B{self.cc.ansatz.name}"

    @abstractmethod
    def get_t1_norm(self, amplitudes: Optional[Namespace[SpinArrayType]] = None) -> T:
        """Get the norm of the T1 amplitude.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Norm of the T1 amplitude.
        """
        pass

    @abstractmethod
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
        pass

    def kernel(self) -> float:
        """Run the Brueckner orbital coupled cluster calculation.

        Returns:
            Correlation energy.
        """
        timer = util.Timer()

        # Make sure the initial CC calculation is converged:
        eris = self.cc.get_eris()
        if not self.cc.converged:
            with lib.temporary_env(self.cc, log=NullLogger()):
                self.cc.kernel(eris=eris)

        # Set up DIIS:
        damping = self.Damping(options=self.options)

        # Initialise coefficients:
        mo_coeff_new: NDArray[T] = np.copy(np.asarray(self.cc.mo_coeff, dtype=types[float]))
        mo_coeff_ref: NDArray[T] = np.copy(np.asarray(self.cc.mo_coeff, dtype=types[float]))
        mo_coeff_ref = self.mo_to_correlated(mo_coeff_ref)
        u_tot = None

        self.cc.log.output(f"Solving for {ANSI.m}Brueckner orbitals{ANSI.R}.")
        self.cc.log.debug("")
        self.log.info(
            f"{ANSI.B}{'Iter':>4s} {'Energy (corr.)':>16s} {'Energy (tot.)':>18s} "
            f"{'Conv.':>8s} {'Δ(Energy)':>13s} {'|T1|':>13s}{ANSI.R}"
        )
        self.log.info(
            f"%4d %16.10f %18.10f {[ANSI.r, ANSI.g][self.cc.converged]}%8r{ANSI.R}",
            0,
            self.cc.e_corr,
            self.cc.e_tot,
            self.cc.converged,
        )

        converged = False
        for niter in range(1, self.options.max_iter + 1):
            # Update rotation matrix:
            u, u_tot = self.get_rotation_matrix(u_tot=u_tot, damping=damping, t1=self.cc.t1)

            # Update MO coefficients:
            mo_coeff_new = self.update_coefficients(u_tot, mo_coeff_new, mo_coeff_ref)

            # Transform mean-field and amplitudes:
            self.mf.mo_coeff = numpy.asarray(mo_coeff_new)
            self.mf.e_tot = self.mf.energy_tot()
            amplitudes = self.transform_amplitudes(u)

            # Run CC calculation:
            e_prev = self.cc.e_tot
            with lib.temporary_env(self.cc, log=NullLogger()):
                self.cc.__class__.__init__(
                    self.cc,
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
                eris = self.cc.get_eris()
                self.cc.kernel(eris=eris)
            de = abs(e_prev - self.cc.e_tot)
            dt = self.get_t1_norm()

            # Log the iteration:
            converged_e = bool(de < self.options.e_tol)
            converged_t = bool(dt < self.options.t_tol)
            self.log.info(
                f"%4s %16.10f %18.10f {[ANSI.r, ANSI.g][int(converged)]}%8r{ANSI.R}"
                f" {[ANSI.r, ANSI.g][int(converged_e)]}%13.3e{ANSI.R}"
                f" {[ANSI.r, ANSI.g][int(converged_t)]}%13.3e{ANSI.R}",
                niter,
                self.cc.e_corr,
                self.cc.e_tot,
                self.cc.converged,
                de,
                dt,
            )

            # Check for convergence:
            converged = converged_e and converged_t
            if converged:
                self.log.debug("")
                self.log.output(f"{ANSI.g}Converged{ANSI.R}.")
                break
        else:
            self.log.debug("")
            self.log.warning(f"{ANSI.r}Failed to converge{ANSI.R}.")

        self.cc.log.debug("")
        self.cc.log.output("E(corr) = %.10f", self.cc.e_corr)
        self.cc.log.output("E(tot)  = %.10f", self.cc.e_tot)
        self.cc.log.debug("")
        self.cc.log.debug("Time elapsed: %s", timer.format_time(timer()))
        self.cc.log.debug("")

        return self.cc.e_corr


class BaseOptimisedEBCC(_BaseOptimisedEBCC):
    """Base class for optimised coupled cluster."""

    @property
    def name(self) -> str:
        """Get the name of the method."""
        return f"{self.spin_type}O{self.cc.ansatz.name}"

    @abstractmethod
    def get_grad_norm(self, u: SpinArrayType) -> T:
        """Get the norm of the gradient.

        Args:
            u: Rotation matrix.

        Returns:
            Norm of the gradient.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def kernel(self) -> float:
        """Run the optimised coupled cluster calculation.

        Returns:
            Correlation energy.
        """
        timer = util.Timer()

        # Make sure the initial CC calculation is converged:
        eris = self.cc.get_eris()
        with lib.temporary_env(self.cc, log=NullLogger()):
            if not self.cc.converged:
                self.cc.kernel(eris=eris)
            if not self.cc.converged_lambda:
                self.cc.solve_lambda(eris=eris)

        # Set up DIIS:
        damping = self.Damping(options=self.options)

        # Initialise coefficients:
        mo_coeff_new: NDArray[T] = np.copy(np.asarray(self.cc.mo_coeff, dtype=types[float]))
        mo_coeff_ref: NDArray[T] = np.copy(np.asarray(self.cc.mo_coeff, dtype=types[float]))
        mo_coeff_ref = self.mo_to_correlated(mo_coeff_ref)
        u_tot = None

        # Initialise density matrices:
        rdm1 = self.cc.make_rdm1_f(eris=eris)
        rdm2 = self.cc.make_rdm2_f(eris=eris)
        assert np.allclose(
            self.cc.e_tot, self.energy(rdm1=rdm1, rdm2=rdm2, eris=eris)
        )  # FIXME remove

        self.cc.log.output(f"Solving for {ANSI.m}optimised orbitals{ANSI.R}.")
        self.cc.log.debug("")
        self.log.info(
            f"{ANSI.B}{'Iter':>4s} {'Energy (corr.)':>16s} {'Energy (tot.)':>18s} "
            f"{'Δ(Energy)':>13s} {'|Gradient|':>13s}{ANSI.R}"
        )
        self.log.info(
            "%4d %16.10f %18.10f",
            0,
            self.cc.e_corr,
            self.cc.e_tot,
        )

        converged = False
        for niter in range(1, self.options.max_iter + 1):
            # Update rotation matrix:
            u, u_tot = self.get_rotation_matrix(
                u_tot=u_tot, damping=damping, eris=eris, rdm1=rdm1, rdm2=rdm2
            )

            # Update MO coefficients:
            mo_coeff_new = self.update_coefficients(u_tot, mo_coeff_new, mo_coeff_ref)

            # Transform mean-field and amplitudes:
            self.mf.mo_coeff = numpy.asarray(mo_coeff_new)
            self.mf.e_tot = self.mf.energy_tot()
            amplitudes = self.transform_amplitudes(u)
            lambas = self.transform_amplitudes(u, is_lambda=True)

            # Update CC calculation:
            e_prev = self.cc.e_tot
            with lib.temporary_env(self.cc, log=NullLogger()):
                self.cc.__class__.__init__(
                    self.cc,
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
                self.cc.lambdas = lambas
                eris = self.cc.get_eris()

            # Update the density matrices:
            rdm1 = self.cc.make_rdm1_f(eris=eris)
            rdm2 = self.cc.make_rdm2_f(eris=eris)

            # Update the energy:
            e_tot = self.energy(rdm1=rdm1, rdm2=rdm2, eris=eris)
            self.cc.e_corr = e_tot - self.cc.mf.e_tot

            # Log the iteration:
            de = abs(e_prev - self.cc.e_tot)
            grad = self.get_grad_norm(u)
            converged_e = bool(de < self.options.e_tol)
            converged_grad = bool(grad < self.options.t_tol)
            self.log.info(
                f"%4s %16.10f %18.10f"
                f" {[ANSI.r, ANSI.g][int(converged_e)]}%13.3e{ANSI.R}"
                f" {[ANSI.r, ANSI.g][int(converged_grad)]}%13.3e{ANSI.R}",
                niter,
                self.cc.e_corr,
                self.cc.e_tot,
                de,
                grad,
            )

            # Check for convergence:
            converged = converged_e and converged_grad
            if converged:
                self.log.debug("")
                self.log.output(f"{ANSI.g}Converged{ANSI.R}.")
                break
        else:
            self.log.debug("")
            self.log.warning(f"{ANSI.r}Failed to converge{ANSI.R}.")

        self.cc.log.debug("")
        self.cc.log.output("E(corr) = %.10f", self.cc.e_corr)
        self.cc.log.output("E(tot)  = %.10f", self.cc.e_tot)
        self.cc.log.debug("")
        self.cc.log.debug("Time elapsed: %s", timer.format_time(timer()))
        self.cc.log.debug("")

        return self.cc.e_corr
