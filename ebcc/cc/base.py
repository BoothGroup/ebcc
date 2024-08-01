"""Base classes for `ebcc.cc`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ebcc import default_log, init_logging
from ebcc import numpy as np
from ebcc import util
from ebcc.ansatz import Ansatz
from ebcc.logging import ANSI
from ebcc.precision import types

if TYPE_CHECKING:
    from logging import Logger
    from typing import Any, Optional, Union

    from pyscf.scf import SCF  # type: ignore

    from ebcc.base import BruecknerEBCC as BaseBrueckner
    from ebcc.base import ERIs as BaseERIs
    from ebcc.base import Fock as BaseFock
    from ebcc.numpy.typing import NDArray  # type: ignore
    from ebcc.util import Namespace


@dataclass
class BaseOptions:
    """Options for EBCC calculations.

    Args:
        shift: Shift the boson operators such that the Hamiltonian is normal-ordered with respect
            to a coherent state. This removes the bosonic coupling to the static mean-field
            density, introducing a constant energy shift.
        e_tol: Threshold for convergence in the correlation energy.
        t_tol: Threshold for convergence in the amplitude norm.
        max_iter: Maximum number of iterations.
        diis_space: Number of amplitudes to use in DIIS extrapolation.
        damping: Damping factor for DIIS extrapolation.
    """

    shift: bool = True
    e_tol: float = 1e-8
    t_tol: float = 1e-8
    max_iter: int = 200
    diis_space: int = 12
    damping: float = 0.0


class BaseEBCC(ABC):
    """Base class for electron-boson coupled cluster.

    Attributes:
        mf: PySCF mean-field object.
        log: Log to write output to.
        options: Options for the EBCC calculation.
        e_corr: Correlation energy.
        amplitudes: Cluster amplitudes.
        converged: Convergence flag.
        lambdas: Cluster lambda amplitudes.
        converged_lambda: Lambda convergence flag.
        name: Name of the method.
    """

    # Types
    Options: type[BaseOptions] = BaseOptions
    ERIs: type[BaseERIs]
    Fock: type[BaseFock]
    CDERIs: type[BaseERIs]
    Brueckner: type[BaseBrueckner]

    # Attributes
    space: Any
    amplitudes: Namespace[Any]
    lambdas: Namespace[Any]

    def __init__(
        self,
        mf: SCF,
        log: Optional[Logger] = None,
        ansatz: Optional[Union[Ansatz, str]] = "CCSD",
        options: Optional[BaseOptions] = None,
        space: Optional[Any] = None,
        omega: Optional[Any] = None,
        g: Optional[Any] = None,
        G: Optional[Any] = None,
        mo_coeff: Optional[Any] = None,
        mo_occ: Optional[Any] = None,
        fock: Optional[Any] = None,
        **kwargs,
    ):
        r"""Initialize the EBCC object.

        Args:
            mf: PySCF mean-field object.
            log: Log to write output to. Default is the global logger, outputting to `stderr`.
            ansatz: Overall ansatz.
            options: Options for the EBCC calculation.
            space: Space containing the frozen, correlated, and active fermionic spaces. Default
                assumes all electrons are correlated.
            omega: Bosonic frequencies.
            g: Electron-boson coupling matrix corresponding to the bosonic annihilation operator
                :math:`g_{bpq} p^\dagger q b`. The creation part is assumed to be the fermionic
                conjugate transpose to retain Hermiticity in the Hamiltonian.
            G: Boson non-conserving term :math:`G_{b} (b^\dagger + b)`.
            mo_coeff: Molecular orbital coefficients. Default is the mean-field coefficients.
            mo_occ: Molecular orbital occupation numbers. Default is the mean-field occupation.
            fock: Fock matrix. Default is the mean-field Fock matrix.
            **kwargs: Additional keyword arguments used to update `options`.
        """
        # Options:
        if options is None:
            options = self.Options()
        self.options = options
        for key, val in kwargs.items():
            setattr(self.options, key, val)

        # Parameters:
        self.log = default_log if log is None else log
        self.mf = self._convert_mf(mf)
        self._mo_coeff = np.asarray(mo_coeff).astype(types[float]) if mo_coeff is not None else None
        self._mo_occ = np.asarray(mo_occ).astype(types[float]) if mo_occ is not None else None

        # Ansatz:
        if isinstance(ansatz, Ansatz):
            self.ansatz = ansatz
        else:
            self.ansatz = Ansatz.from_string(
                ansatz, density_fitting=getattr(self.mf, "with_df", None) is not None
            )
        self._eqns = self.ansatz._get_eqns(self.spin_type)

        # Space:
        if space is not None:
            self.space = space
        else:
            self.space = self.init_space()

        # Boson parameters:
        if bool(self.fermion_coupling_rank) != bool(self.boson_coupling_rank):
            raise ValueError(
                "Fermionic and bosonic coupling ranks must both be zero, or both non-zero."
            )
        self.omega = omega.astype(types[float]) if omega is not None else None
        self.bare_g = g.astype(types[float]) if g is not None else None
        self.bare_G = G.astype(types[float]) if G is not None else None
        if self.boson_ansatz != "":
            self.g = self.get_g(g)
            self.G = self.get_mean_field_G()
            if self.options.shift:
                self.log.info(" > Energy shift due to polaritonic basis:  %.10f", self.const)
        else:
            assert self.nbos == 0
            self.options.shift = False
            self.g = None
            self.G = None

        # Fock matrix:
        if fock is None:
            self.fock = self.get_fock()
        else:
            self.fock = fock

        # Attributes:
        self.e_corr = types[float](0.0)
        self.amplitudes = util.Namespace()
        self.converged = False
        self.lambdas = util.Namespace()
        self.converged_lambda = False

        # Logging:
        init_logging(self.log)
        self.log.info(f"\n{ANSI.B}{ANSI.U}{self.name}{ANSI.R}")
        self.log.debug(f"{ANSI.B}{'*' * len(self.name)}{ANSI.R}")
        self.log.debug("")
        self.log.info(f"{ANSI.B}Options{ANSI.R}:")
        self.log.info(f" > e_tol:  {ANSI.y}{self.options.e_tol}{ANSI.R}")
        self.log.info(f" > t_tol:  {ANSI.y}{self.options.t_tol}{ANSI.R}")
        self.log.info(f" > max_iter:  {ANSI.y}{self.options.max_iter}{ANSI.R}")
        self.log.info(f" > diis_space:  {ANSI.y}{self.options.diis_space}{ANSI.R}")
        self.log.info(f" > damping:  {ANSI.y}{self.options.damping}{ANSI.R}")
        self.log.debug("")
        self.log.info(f"{ANSI.B}Ansatz{ANSI.R}: {ANSI.m}{self.ansatz}{ANSI.R}")
        self.log.debug("")
        self.log.info(f"{ANSI.B}Space{ANSI.R}: {ANSI.m}{self.space}{ANSI.R}")
        self.log.debug("")

    @abstractmethod
    @staticmethod
    def _convert_mf(mf: SCF) -> SCF:
        """Convert the mean-field object to the appropriate type."""
        pass

    @abstractmethod
    @property
    def spin_type(self) -> str:
        """Get a string representation of the spin type."""
        pass

    @abstractmethod
    @property
    def name(self) -> str:
        """Get the name of the method."""
        pass

    @property
    def fermion_ansatz(self) -> str:
        """Get a string representation of the fermion ansatz."""
        return self.ansatz.fermion_ansatz

    @property
    def boson_ansatz(self) -> str:
        """Get a string representation of the boson ansatz."""
        return self.ansatz.boson_ansatz

    @property
    def fermion_coupling_rank(self) -> int:
        """Get an integer representation of the fermion coupling rank."""
        return self.ansatz.fermion_coupling_rank

    @property
    def boson_coupling_rank(self) -> int:
        """Get an integer representation of the boson coupling rank."""
        return self.ansatz.boson_coupling_rank

    @abstractmethod
    def init_space(self) -> Any:
        """Initialise the fermionic space.

        Returns:
            Fermionic space.
        """
        pass

    @abstractmethod
    def get_fock(self) -> Any:
        """Get the Fock matrix.

        Returns:
            Fock matrix.
        """
        pass

    @abstractmethod
    def get_eris(self, eris: Optional[Union[type[BaseERIs], NDArray[float]]]) -> Any:
        """Get the electron repulsion integrals.

        Args:
            eris: Input electron repulsion integrals.

        Returns:
            Electron repulsion integrals.
        """
        pass

    @abstractmethod
    def get_g(self, g: NDArray[float]) -> Any:
        """Get the blocks of the electron-boson coupling matrix.

        This matrix corresponds to the bosonic annihilation operator.

        Args:
            g: Electron-boson coupling matrix.

        Returns:
            Blocks of the electron-boson coupling matrix.
        """
        pass

    @abstractmethod
    def get_mean_field_G(self) -> Any:
        """Get the mean-field boson non-conserving term.

        Returns:
            Mean-field boson non-conserving term.
        """
        pass

    def const(self) -> float:
        """Get the shift in energy from moving to the polaritonic basis.

        Returns:
            Constant energy shift due to the polaritonic basis.
        """
        if self.options.shift:
            return util.einsum("I,I->", self.omega, self.xi**2)
        return 0.0

    @abstractmethod
    @property
    def xi(self) -> NDArray[float]:
        """Get the shift in the bosonic operators to diagonalise the photon Hamiltonian.

        Returns:
            Shift in the bosonic operators.
        """
        pass

    @property
    def mo_coeff(self) -> NDArray[float]:
        """Get the molecular orbital coefficients.

        Returns:
            Molecular orbital coefficients.
        """
        if self._mo_coeff is None:
            return np.asarray(self.mf.mo_coeff).astype(types[float])
        return self._mo_coeff

    @property
    def mo_occ(self) -> NDArray[float]:
        """Get the molecular orbital occupation numbers.

        Returns:
            Molecular orbital occupation numbers.
        """
        if self._mo_occ is None:
            return np.asarray(self.mf.mo_occ).astype(types[float])
        return self._mo_occ

    @abstractmethod
    @property
    def nmo(self) -> Any:
        """Get the number of molecular orbitals.

        Returns:
            Number of molecular orbitals.
        """
        pass

    @abstractmethod
    @property
    def nocc(self) -> Any:
        """Get the number of occupied molecular orbitals.

        Returns:
            Number of occupied molecular orbitals.
        """
        pass

    @abstractmethod
    @property
    def nvir(self) -> Any:
        """Get the number of virtual molecular orbitals.

        Returns:
            Number of virtual molecular orbitals.
        """
        pass

    @property
    def nbos(self) -> int:
        """Get the number of bosonic modes.

        Returns:
            Number of bosonic modes.
        """
        if self.omega is None:
            return 0
        return self.omega.shape[0]

    @property
    def e_tot(self) -> float:
        """Get the total energy (mean-field plus correlation).

        Returns:
            Total energy.
        """
        return types[float](self.mf.e_tot) + self.e_corr

    @property
    def t1(self) -> Any:
        """Get the T1 amplitudes."""
        return self.amplitudes["t1"]

    @property
    def t2(self) -> Any:
        """Get the T2 amplitudes."""
        return self.amplitudes["t2"]

    @property
    def t3(self) -> Any:
        """Get the T3 amplitudes."""
        return self.amplitudes["t3"]

    @property
    def l1(self) -> Any:
        """Get the L1 amplitudes."""
        return self.lambdas["l1"]

    @property
    def l2(self) -> Any:
        """Get the L2 amplitudes."""
        return self.lambdas["l2"]

    @property
    def l3(self) -> Any:
        """Get the L3 amplitudes."""
        return self.lambdas["l3"]
