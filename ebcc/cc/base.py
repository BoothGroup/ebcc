"""Base classes for `ebcc.cc`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ebcc import default_log, init_logging
from ebcc import numpy as np
from ebcc import util
from ebcc.ansatz import Ansatz
from ebcc.space import Space
from ebcc.logging import ANSI
from ebcc.precision import types

if TYPE_CHECKING:
    from logging import Logger
    from typing import Optional, Union, Any
    from dataclasses import dataclass

    from pyscf.scf import SCF


class EBCC(ABC):
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
    Options: dataclass
    ERIs: ERIs
    Fock: Fock
    CDERIs: ERIs
    Brueckner: BruecknerEBCC

    def __init__(
        self,
        mf: SCF,
        log: Optional[Logger] = None,
        ansatz: Optional[Union[Ansatz, str]] = "CCSD",
        options: Optional[dataclass] = None,
        space: Optional[Any] = None,
        omega: Optional[Any] = None,
        g: Optional[Any] = None,
        G: Optional[Any] = None,
        mo_coeff: Optional[Any] = None,
        mo_occ: Optional[Any] = None,
        fock: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize the EBCC object.

        Args:
            mf: PySCF mean-field object.
            log: Log to write output to. Default is the global logger, outputting to `stderr`.
            ansatz: Overall ansatz.
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
        self.e_corr = None
        self.amplitudes = None
        self.converged = False
        self.lambdas = None
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
