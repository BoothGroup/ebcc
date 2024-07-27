"""Base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ebcc import util

if TYPE_CHECKING:
    from logging import Logger
    from typing import Optional, Union, Any
    from dataclasses import dataclass

    from pyscf.scf import SCF

    from ebcc.ansatz import Ansatz
    from ebcc.space import Space


class EBCC(ABC):
    """Base class for electron-boson coupled cluster.

    """

    # Types
    Options: dataclass
    ERIs: ERIs
    Fock: Fock
    CDERIs: ERIs
    Brueckner: BruecknerEBCC

    # Metadata
    log: Logger
    options: dataclass

    # Mean-field
    mf: SCF
    _mo_coeff: Any
    _mo_occ: Any
    fock: Any

    # Ansatz
    ansatz: Ansatz
    _eqns: ModuleType
    space: Any

    # Bosons
    omega: Any
    bare_g: Any
    bare_G: Any

    # Results
    e_corr: float
    amplitudes: Any
    converged: bool

    def __init__(
        self,
        mf: SCF,
        log: Optional[Logger] = None,
        ansatz: Optional[Union[Ansatz, str]] = "CCSD",
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



class EOM(ABC):
    """Base class for equation-of-motion methods."""

    pass


class BruecknerEBCC(ABC):
    """Base class for Brueckner orbital methods."""

    pass


class ERIs(ABC, util.Namespace):
    """Base class for electronic repulsion integrals."""

    pass


class Fock(ABC, util.Namespace):
    """Base class for Fock matrices."""

    pass
