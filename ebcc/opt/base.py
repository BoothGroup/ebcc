"""Base classes for `ebcc.opt`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy
from pyscf import lib

from ebcc import util
from ebcc import numpy as np
from ebcc.core.damping import DIIS
from ebcc.core.logging import ANSI, NullLogger, init_logging
from ebcc.core.precision import types

if TYPE_CHECKING:
    from typing import Any, Optional

    from numpy import float64
    from numpy.typing import NDArray

    from ebcc.cc.base import BaseEBCC, SpinArrayType
    from ebcc.util import Namespace

    T = float64

# FIXME Custom versions of PySCF functions


@dataclass
class BaseOptions:
    """Options for Brueckner-orbital calculations.

    Args:
        e_tol: Threshold for converged in the correlation energy.
        t_tol: Threshold for converged in the amplitude norm.
        max_iter: Maximum number of iterations.
        diis_space: Number of amplitudes to use in DIIS extrapolation.
        damping: Damping factor for DIIS extrapolation.
    """

    e_tol: float = 1e-8
    t_tol: float = 1e-8
    max_iter: int = 20
    diis_space: int = 12
    damping: float = 0.0


class BaseBruecknerEBCC(ABC):
    """Base class for Brueckner-orbital coupled cluster."""

    # Types
    Options: type[BaseOptions] = BaseOptions

    # Attributes
    cc: BaseEBCC

    def __init__(
        self,
        cc: BaseEBCC,
        options: Optional[BaseOptions] = None,
        **kwargs: Any,
    ) -> None:
        r"""Initialise the Brueckner EBCC object.

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
        cc.log.info(f"{ANSI.B}Options{ANSI.R}:")
        cc.log.info(f" > e_tol:  {ANSI.y}{self.options.e_tol}{ANSI.R}")
        cc.log.info(f" > t_tol:  {ANSI.y}{self.options.t_tol}{ANSI.R}")
        cc.log.info(f" > max_iter:  {ANSI.y}{self.options.max_iter}{ANSI.R}")
        cc.log.info(f" > diis_space:  {ANSI.y}{self.options.diis_space}{ANSI.R}")
        cc.log.info(f" > damping:  {ANSI.y}{self.options.damping}{ANSI.R}")
        cc.log.debug("")

    @property
    def spin_type(self) -> str:
        """Get the spin type."""
        return self.cc.spin_type

    @property
    def name(self) -> str:
        """Get the name of the method."""
        return f"{self.spin_type}B{self.cc.ansatz.name}"

    def kernel(self) -> float:
        """Run the Bruckner-orbital coupled cluster calculation.

        Returns:
            Correlation energy.
        """
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
        mo_coeff_new: NDArray[T] = np.copy(np.astype(self.cc.mo_coeff, types[float]))
        mo_coeff_ref: NDArray[T] = np.copy(np.astype(self.cc.mo_coeff, types[float]))
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
                self.cc.kernel()
            de = abs(e_prev - self.cc.e_tot)
            dt = self.get_t1_norm()

            # Log the iteration:
            converged_e = bool(de < self.options.e_tol)
            converged_t = bool(dt < self.options.t_tol)
            self.log.info(
                f"{niter:4d} {self.cc.e_corr:16.10f} {self.cc.e_tot:18.10f}"
                f" {[ANSI.r, ANSI.g][int(self.cc.converged)]}{self.cc.converged!r:>8}{ANSI.R}"
                f" {[ANSI.r, ANSI.g][int(converged_e)]}{de:13.3e}{ANSI.R}"
                f" {[ANSI.r, ANSI.g][int(converged_t)]}{dt:13.3e}{ANSI.R}"
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

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

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
