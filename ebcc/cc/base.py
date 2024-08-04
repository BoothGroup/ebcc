"""Base classes for `ebcc.cc`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ebcc import default_log, init_logging
from ebcc import numpy as np
from ebcc import util
from ebcc.core.ansatz import Ansatz
from ebcc.core.damping import DIIS
from ebcc.core.dump import Dump
from ebcc.core.logging import ANSI
from ebcc.core.precision import astype, types

if TYPE_CHECKING:
    from typing import Any, Callable, Literal, Optional, TypeVar, Union

    from pyscf.scf.hf import SCF  # type: ignore

    from ebcc.core.logging import Logger
    from ebcc.ham.base import BaseERIs, BaseFock
    from ebcc.numpy.typing import NDArray
    from ebcc.opt.base import BaseBruecknerEBCC
    from ebcc.util import Namespace

    ERIsInputType = Union[type[BaseERIs], NDArray[float]]
    AmplitudeType = TypeVar("AmplitudeType")
    SpaceType = TypeVar("SpaceType")


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
    Brueckner: type[BaseBruecknerEBCC]

    # Attributes
    space: SpaceType
    amplitudes: Namespace[AmplitudeType]
    lambdas: Namespace[AmplitudeType]

    def __init__(
        self,
        mf: SCF,
        log: Optional[Logger] = None,
        ansatz: Optional[Union[Ansatz, str]] = "CCSD",
        options: Optional[BaseOptions] = None,
        space: Optional[SpaceType] = None,
        omega: Optional[NDArray[float]] = None,
        g: Optional[NDArray[float]] = None,
        G: Optional[NDArray[float]] = None,
        mo_coeff: Optional[NDArray[float]] = None,
        mo_occ: Optional[NDArray[float]] = None,
        fock: Optional[BaseFock] = None,
        **kwargs: Any,
    ) -> None:
        r"""Initialise the EBCC object.

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
        self.e_corr = 0.0
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

    @property
    @abstractmethod
    def spin_type(self) -> str:
        """Get a string representation of the spin type."""
        pass

    @property
    def name(self) -> str:
        """Get the name of the method."""
        return self.spin_type + self.ansatz.name

    def kernel(self, eris: Optional[ERIsInputType] = None) -> float:
        """Run the coupled cluster calculation.

        Args:
            eris: Electron repulsion integrals.

        Returns:
            Correlation energy.
        """
        timer = util.Timer()

        # Get the ERIs:
        eris = self.get_eris(eris)

        # Get the amplitude guesses:
        amplitudes = self.amplitudes
        if not amplitudes:
            amplitudes = self.init_amps(eris=eris)

        # Get the initial energy:
        e_cc = self.energy(amplitudes=amplitudes, eris=eris)

        self.log.output("Solving for excitation amplitudes.")
        self.log.debug("")
        self.log.info(
            f"{ANSI.B}{'Iter':>4s} {'Energy (corr.)':>16s} {'Energy (tot.)':>18s} "
            f"{'Δ(Energy)':>13s} {'Δ(Ampl.)':>13s}{ANSI.R}"
        )
        self.log.info(f"{0:4d} {e_cc:16.10f} {e_cc + self.mf.e_tot:18.10f}")

        if not self.ansatz.is_one_shot:
            # Set up DIIS:
            diis = DIIS()
            diis.space = self.options.diis_space
            diis.damping = self.options.damping

            converged = False
            for niter in range(1, self.options.max_iter + 1):
                # Update the amplitudes, extrapolate with DIIS and calculate change:
                amplitudes_prev = amplitudes
                amplitudes = self.update_amps(amplitudes=amplitudes, eris=eris)
                vector = self.amplitudes_to_vector(amplitudes)
                vector = diis.update(vector)
                amplitudes = self.vector_to_amplitudes(vector)
                dt = np.linalg.norm(vector - self.amplitudes_to_vector(amplitudes_prev), ord=np.inf)

                # Update the energy and calculate change:
                e_prev = e_cc
                e_cc = self.energy(amplitudes=amplitudes, eris=eris)
                de = abs(e_prev - e_cc)

                # Log the iteration:
                converged_e = bool(de < self.options.e_tol)
                converged_t = bool(dt < self.options.t_tol)
                self.log.info(
                    f"{niter:4d} {e_cc:16.10f} {e_cc + self.mf.e_tot:18.10f}"
                    f" {[ANSI.r, ANSI.g][bool(converged_e)]}{de:13.3e}{ANSI.R}"
                    f" {[ANSI.r, ANSI.g][bool(converged_t)]}{dt:13.3e}{ANSI.R}"
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

            # Include perturbative correction if required:
            if self.ansatz.has_perturbative_correction:
                self.log.debug("")
                self.log.info("Computing perturbative energy correction.")
                e_pert = self.energy_perturbative(amplitudes=amplitudes, eris=eris)
                e_cc += e_pert
                self.log.info(f"E(pert) = {e_pert:.10f}")

        else:
            converged = True

        # Update attributes:
        self.e_corr = e_cc
        self.amplitudes = amplitudes
        self.converged = converged

        self.log.debug("")
        self.log.output(f"E(corr) = {self.e_corr:.10f}")
        self.log.output(f"E(tot)  = {self.e_tot:.10f}")
        self.log.debug("")
        self.log.debug("Time elapsed: %s", timer.format_time(timer()))
        self.log.debug("")

        return e_cc

    def solve_lambda(
        self,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        eris: Optional[ERIsInputType] = None,
    ) -> None:
        """Solve for the lambda amplitudes.

        Args:
            amplitudes: Cluster amplitudes.
            eris: Electron repulsion integrals.
        """
        timer = util.Timer()

        # Get the ERIs:
        eris = self.get_eris(eris)

        # Get the amplitudes:
        amplitudes = self.amplitudes
        if not amplitudes:
            amplitudes = self.init_amps(eris=eris)

        # If needed, get the perturbative part of the lambda amplitudes:
        lambdas_pert = None
        if self.ansatz.has_perturbative_correction:
            lambdas_pert = self.update_lams(eris=eris, amplitudes=amplitudes, perturbative=True)

        # Get the initial lambda amplitudes:
        lambdas = self.lambdas
        if not lambdas:
            lambdas = self.init_lams(amplitudes=amplitudes)

        self.log.output("Solving for de-excitation (lambda) amplitudes.")
        self.log.debug("")
        self.log.info(f"{ANSI.B}{'Iter':>4s} {'Δ(Ampl.)':>13s}{ANSI.R}")

        # Set up DIIS:
        diis = DIIS()
        diis.space = self.options.diis_space
        diis.damping = self.options.damping

        converged = False
        for niter in range(1, self.options.max_iter + 1):
            # Update the lambda amplitudes, extrapolate with DIIS and calculate change:
            lambdas_prev = lambdas
            lambdas = self.update_lams(
                amplitudes=amplitudes,
                lambdas=lambdas,
                lambdas_pert=lambdas_pert,
                eris=eris,
            )
            vector = self.lambdas_to_vector(lambdas)
            vector = diis.update(vector)
            lambdas = self.vector_to_lambdas(vector)
            dl = np.linalg.norm(vector - self.lambdas_to_vector(lambdas_prev), ord=np.inf)

            # Log the iteration:
            converged = bool(dl < self.options.t_tol)
            self.log.info(f"{niter:4d} {[ANSI.r, ANSI.g][converged]}{dl:13.3e}{ANSI.R}")

            # Check for convergence:
            if converged:
                self.log.debug("")
                self.log.output(f"{ANSI.g}Converged.{ANSI.R}")
                break
        else:
            self.log.debug("")
            self.log.warning(f"{ANSI.r}Failed to converge.{ANSI.R}")

        self.log.debug("")
        self.log.debug("Time elapsed: %s", timer.format_time(timer()))
        self.log.debug("")
        self.log.debug("")

        # Update attributes:
        self.lambdas = lambdas
        self.converged_lambda = converged

    @abstractmethod
    def ip_eom(self, options: Optional[BaseOptions] = None, **kwargs: Any) -> Any:
        """Get the IP-EOM object.

        Args:
            options: Options for the IP-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            IP-EOM object.
        """
        pass

    @abstractmethod
    def ea_eom(self, options: Optional[BaseOptions] = None, **kwargs: Any) -> Any:
        """Get the EA-EOM object.

        Args:
            options: Options for the EA-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            EA-EOM object.
        """
        pass

    @abstractmethod
    def ee_eom(self, options: Optional[BaseOptions] = None, **kwargs: Any) -> Any:
        """Get the EE-EOM object.

        Args:
            options: Options for the EE-EOM calculation.
            **kwargs: Additional keyword arguments.

        Returns:
            EE-EOM object.
        """
        pass

    def brueckner(self, *args: Any, **kwargs: Any) -> float:
        """Run a Brueckner orbital coupled cluster calculation.

        The coupled cluster object will be update in-place.

        Args:
            *args: Arguments to pass to the Brueckner object.
            **kwargs: Keyword arguments to pass to the Brueckner object.

        Returns:
            Correlation energy.
        """
        bcc = self.Brueckner(self, *args, **kwargs)
        return bcc.kernel()

    def write(self, file: str) -> None:
        """Write the EBCC object to a file.

        Args:
            file: File to write the object to.
        """
        writer = Dump(file)
        writer.write(self)

    @classmethod
    def read(cls, file: str, log: Optional[Logger] = None) -> BaseEBCC:
        """Read the EBCC object from a file.

        Args:
            file: File to read the object from.
            log: Logger to use for new object.

        Returns:
            EBCC object.
        """
        reader = Dump(file)
        return reader.read(cls=cls, log=log)

    @staticmethod
    @abstractmethod
    def _convert_mf(mf: SCF) -> SCF:
        """Convert the mean-field object to the appropriate type."""
        pass

    def _load_function(
        self,
        name: str,
        eris: Optional[Union[ERIsInputType, Literal[False]]] = False,
        amplitudes: Optional[Union[Namespace[AmplitudeType], Literal[False]]] = False,
        lambdas: Optional[Union[Namespace[AmplitudeType], Literal[False]]] = False,
        **kwargs: Any,
    ) -> tuple[Callable[..., Any], dict[str, Any]]:
        """Load a function from the generated code, and return the arguments."""
        dicts = []

        # Get the ERIs:
        if not (eris is False):
            eris = self.get_eris(eris)
        else:
            eris = None

        # Get the amplitudes:
        if not (amplitudes is False):
            if not amplitudes:
                amplitudes = self.amplitudes
            if not amplitudes:
                amplitudes = self.init_amps(eris=eris)
            dicts.append(dict(amplitudes))

        # Get the lambda amplitudes:
        if not (lambdas is False):
            if not lambdas:
                lambdas = self.lambdas
            if not lambdas:
                lambdas = self.init_lams(amplitudes=amplitudes if amplitudes else None)
            dicts.append(dict(lambdas))

        # Get the function:
        func = getattr(self._eqns, name, None)
        if func is None:
            raise util.ModelNotImplemented("%s for %s" % (name, self.name))

        # Get the arguments:
        if kwargs:
            dicts.append(kwargs)
        all_kwargs = self._pack_codegen_kwargs(*dicts, eris=eris)

        return func, all_kwargs

    @abstractmethod
    def _pack_codegen_kwargs(
        self, *extra_kwargs: dict[str, Any], eris: Optional[ERIsInputType] = None
    ) -> dict[str, Any]:
        """Pack all the keyword arguments for the generated code."""
        pass

    @abstractmethod
    def init_amps(self, eris: Optional[ERIsInputType] = None) -> Namespace[AmplitudeType]:
        """Initialise the cluster amplitudes.

        Args:
            eris: Electron repulsion integrals.

        Returns:
            Initial cluster amplitudes.
        """
        pass

    @abstractmethod
    def init_lams(
        self, amplitude: Optional[Namespace[AmplitudeType]] = None
    ) -> Namespace[AmplitudeType]:
        """Initialise the cluster lambda amplitudes.

        Args:
            amplitude: Cluster amplitudes.

        Returns:
            Initial cluster lambda amplitudes.
        """
        pass

    def energy(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
    ) -> float:
        """Calculate the correlation energy.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.

        Returns:
            Correlation energy.
        """
        func, kwargs = self._load_function(
            "energy",
            eris=eris,
            amplitudes=amplitudes,
        )
        return astype(func(**kwargs).real, float)

    def energy_perturbative(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
    ) -> float:
        """Calculate the perturbative correction to the correlation energy.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.

        Returns:
            Perturbative energy correction.
        """
        func, kwargs = self._load_function(
            "energy_perturbative",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        return astype(func(**kwargs).real, float)

    @abstractmethod
    def update_amps(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
    ) -> Namespace[AmplitudeType]:
        """Update the cluster amplitudes.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.

        Returns:
            Updated cluster amplitudes.
        """
        pass

    @abstractmethod
    def update_lams(
        self,
        eris: ERIsInputType = None,
        amplitude: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        lambdas_pert: Optional[Namespace[AmplitudeType]] = None,
        perturbative: bool = False,
    ) -> Namespace[AmplitudeType]:
        """Update the cluster lambda amplitudes.

        Args:
            eris: Electron repulsion integrals.
            amplitude: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            lambdas_pert: Perturbative cluster lambda amplitudes.
            perturbative: Flag to include perturbative correction.

        Returns:
            Updated cluster lambda amplitudes.
        """
        pass

    def make_sing_b_dm(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
    ) -> NDArray[float]:
        r"""Make the single boson density matrix :math:`\langle b \rangle`.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.

        Returns:
            Single boson density matrix.
        """
        func, kwargs = self._load_function(
            "make_sing_b_dm",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        return cast(NDArray[float], func(**kwargs))

    def make_rdm1_b(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        unshifted: bool = True,
        hermitise: bool = True,
    ) -> NDArray[float]:
        r"""Make the one-particle boson reduced density matrix :math:`\langle b^+ c \rangle`.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            unshifted: If `self.options.shift` is `True`, return the unshifted density matrix. Has
                no effect if `self.options.shift` is `False`.
            hermitise: Hermitise the density matrix.

        Returns:
            One-particle boson reduced density matrix.
        """
        func, kwargs = self._load_function(
            "make_rdm1_b",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        dm = cast(NDArray[float], func(**kwargs))

        if hermitise:
            dm = 0.5 * (dm + dm.T)

        if unshifted and self.options.shift:
            dm_cre, dm_ann = self.make_sing_b_dm()
            xi = self.xi
            dm[np.diag_indices_from(dm)] -= xi * (dm_cre + dm_ann) - xi**2

        return dm

    @abstractmethod
    def make_rdm1_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
    ) -> Any:
        r"""Make the one-particle fermionic reduced density matrix :math:`\langle i^+ j \rangle`.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            hermitise: Hermitise the density matrix.

        Returns:
            One-particle fermion reduced density matrix.
        """
        pass

    @abstractmethod
    def make_rdm2_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        hermitise: bool = True,
    ) -> Any:
        r"""Make the two-particle fermionic reduced density matrix :math:`\langle i^+j^+lk \rangle`.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            hermitise: Hermitise the density matrix.

        Returns:
            Two-particle fermion reduced density matrix.
        """
        pass

    @abstractmethod
    def make_eb_coup_rdm(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
        unshifted: bool = True,
        hermitise: bool = True,
    ) -> Any:
        r"""Make the electron-boson coupling reduced density matrix.

        .. math::
            \langle b^+ i^+ j \rangle

        and

        .. math::
            \langle b i^+ j \rangle

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            unshifted: If `self.options.shift` is `True`, return the unshifted density matrix. Has
                no effect if `self.options.shift` is `False`.
            hermitise: Hermitise the density matrix.

        Returns:
            Electron-boson coupling reduced density matrix.
        """
        pass

    def hbar_matvec_ip(
        self,
        r1: AmplitudeType,
        r2: AmplitudeType,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
    ) -> tuple[AmplitudeType, AmplitudeType]:
        """Compute the product between a state vector and the IP-EOM Hamiltonian.

        Args:
            r1: State vector (single excitations).
            r2: State vector (double excitations).
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.

        Returns:
            Products between the state vectors and the IP-EOM Hamiltonian for the singles and
            doubles.
        """
        func, kwargs = self._load_function(
            "hbar_matvec_ip",
            eris=eris,
            amplitudes=amplitudes,
            r1=r1,
            r2=r2,
        )
        return cast(tuple[AmplitudeType, AmplitudeType], func(**kwargs))

    def hbar_matvec_ea(
        self,
        r1: AmplitudeType,
        r2: AmplitudeType,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
    ) -> tuple[AmplitudeType, AmplitudeType]:
        """Compute the product between a state vector and the EA-EOM Hamiltonian.

        Args:
            r1: State vector (single excitations).
            r2: State vector (double excitations).
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.

        Returns:
            Products between the state vectors and the EA-EOM Hamiltonian for the singles and
            doubles.
        """
        func, kwargs = self._load_function(
            "hbar_matvec_ea",
            eris=eris,
            amplitudes=amplitudes,
            r1=r1,
            r2=r2,
        )
        return cast(tuple[AmplitudeType, AmplitudeType], func(**kwargs))

    def hbar_matvec_ee(
        self,
        r1: AmplitudeType,
        r2: AmplitudeType,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
    ) -> tuple[AmplitudeType, AmplitudeType]:
        """Compute the product between a state vector and the EE-EOM Hamiltonian.

        Args:
            r1: State vector (single excitations).
            r2: State vector (double excitations).
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.

        Returns:
            Products between the state vectors and the EE-EOM Hamiltonian for the singles and
            doubles.
        """
        func, kwargs = self._load_function(
            "hbar_matvec_ee",
            eris=eris,
            amplitudes=amplitudes,
            r1=r1,
            r2=r2,
        )
        return cast(tuple[AmplitudeType, AmplitudeType], func(**kwargs))

    def make_ip_mom_bras(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
    ) -> tuple[AmplitudeType, ...]:
        """Get the bra vectors to construct IP-EOM moments.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.

        Returns:
            Bra vectors for IP-EOM moments.
        """
        func, kwargs = self._load_function(
            "make_ip_mom_bras",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        return cast(tuple[AmplitudeType, ...], func(**kwargs))

    def make_ea_mom_bras(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
    ) -> tuple[AmplitudeType, ...]:
        """Get the bra vectors to construct EA-EOM moments.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.

        Returns:
            Bra vectors for EA-EOM moments.
        """
        func, kwargs = self._load_function(
            "make_ea_mom_bras",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        return cast(tuple[AmplitudeType, ...], func(**kwargs))

    def make_ee_mom_bras(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
    ) -> tuple[AmplitudeType, ...]:
        """Get the bra vectors to construct EE-EOM moments.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.

        Returns:
            Bra vectors for EE-EOM moments.
        """
        func, kwargs = self._load_function(
            "make_ee_mom_bras",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        return cast(tuple[AmplitudeType, ...], func(**kwargs))

    def make_ip_mom_kets(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
    ) -> tuple[AmplitudeType, ...]:
        """Get the ket vectors to construct IP-EOM moments.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.

        Returns:
            Ket vectors for IP-EOM moments.
        """
        func, kwargs = self._load_function(
            "make_ip_mom_kets",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        return cast(tuple[AmplitudeType, ...], func(**kwargs))

    def make_ea_mom_kets(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
    ) -> tuple[AmplitudeType, ...]:
        """Get the ket vectors to construct EA-EOM moments.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.

        Returns:
            Ket vectors for EA-EOM moments.
        """
        func, kwargs = self._load_function(
            "make_ea_mom_kets",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        return cast(tuple[AmplitudeType, ...], func(**kwargs))

    def make_ee_mom_kets(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[AmplitudeType]] = None,
        lambdas: Optional[Namespace[AmplitudeType]] = None,
    ) -> tuple[AmplitudeType, ...]:
        """Get the ket vectors to construct EE-EOM moments.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.

        Returns:
            Ket vectors for EE-EOM moments.
        """
        func, kwargs = self._load_function(
            "make_ee_mom_kets",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        return cast(tuple[AmplitudeType, ...], func(**kwargs))

    @abstractmethod
    def energy_sum(self, *args: str, signs_dict: Optional[dict[str, int]] = None) -> NDArray[float]:
        """Get a direct sum of energies.

        Args:
            *args: Energies to sum. Subclass should specify a subscript, and optionally spins.
            signs_dict: Signs of the energies in the sum. Default sets `("o", "O", "i")` to be
                positive, and `("v", "V", "a", "b")` to be negative.

        Returns:
            Sum of energies.
        """
        pass

    @abstractmethod
    def amplitudes_to_vector(self, amplitudes: Namespace[AmplitudeType]) -> NDArray[float]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
        pass

    @abstractmethod
    def vector_to_amplitudes(self, vector: NDArray[float]) -> Namespace[AmplitudeType]:
        """Construct a namespace of amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        pass

    @abstractmethod
    def lambdas_to_vector(self, lambdas: Namespace[AmplitudeType]) -> NDArray[float]:
        """Construct a vector containing all of the lambda amplitudes used in the given ansatz.

        Args:
            lambdas: Cluster lambda amplitudes.

        Returns:
            Cluster lambda amplitudes as a vector.
        """
        pass

    @abstractmethod
    def vector_to_lambdas(self, vector: NDArray[float]) -> Namespace[AmplitudeType]:
        """Construct a namespace of lambda amplitudes from a vector.

        Args:
            vector: Cluster lambda amplitudes as a vector.

        Returns:
            Cluster lambda amplitudes.
        """
        pass

    @abstractmethod
    def excitations_to_vector_ip(self, *excitations: Namespace[AmplitudeType]) -> NDArray[float]:
        """Construct a vector containing all of the IP-EOM excitations.

        Args:
            excitations: IP-EOM excitations.

        Returns:
            IP-EOM excitations as a vector.
        """
        pass

    @abstractmethod
    def excitations_to_vector_ea(self, *excitations: Namespace[AmplitudeType]) -> NDArray[float]:
        """Construct a vector containing all of the EA-EOM excitations.

        Args:
            excitations: EA-EOM excitations.

        Returns:
            EA-EOM excitations as a vector.
        """
        pass

    @abstractmethod
    def excitations_to_vector_ee(self, *excitations: Namespace[AmplitudeType]) -> NDArray[float]:
        """Construct a vector containing all of the EE-EOM excitations.

        Args:
            excitations: EE-EOM excitations.

        Returns:
            EE-EOM excitations as a vector.
        """
        pass

    @abstractmethod
    def vector_to_excitations_ip(
        self, vector: NDArray[float]
    ) -> tuple[Namespace[AmplitudeType], ...]:
        """Construct a namespace of IP-EOM excitations from a vector.

        Args:
            vector: IP-EOM excitations as a vector.

        Returns:
            IP-EOM excitations.
        """
        pass

    @abstractmethod
    def vector_to_excitations_ea(
        self, vector: NDArray[float]
    ) -> tuple[Namespace[AmplitudeType], ...]:
        """Construct a namespace of EA-EOM excitations from a vector.

        Args:
            vector: EA-EOM excitations as a vector.

        Returns:
            EA-EOM excitations.
        """
        pass

    @abstractmethod
    def vector_to_excitations_ee(
        self, vector: NDArray[float]
    ) -> tuple[Namespace[AmplitudeType], ...]:
        """Construct a namespace of EE-EOM excitations from a vector.

        Args:
            vector: EE-EOM excitations as a vector.

        Returns:
            EE-EOM excitations.
        """
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
            Fermionic space. All fermionic degrees of freedom are assumed to be correlated.
        """
        pass

    @abstractmethod
    def get_fock(self) -> BaseFock:
        """Get the Fock matrix.

        Returns:
            Fock matrix.
        """
        pass

    @abstractmethod
    def get_eris(self, eris: Optional[ERIsInputType] = None) -> BaseERIs:
        """Get the electron repulsion integrals.

        Args:
            eris: Input electron repulsion integrals.

        Returns:
            Electron repulsion integrals.
        """
        pass

    @abstractmethod
    def get_g(self, g: NDArray[float]) -> Namespace[Any]:
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

    @property
    @abstractmethod
    def bare_fock(self) -> Any:
        """Get the mean-field Fock matrix in the MO basis, including frozen parts.

        Returns an array and not a `BaseFock` object.

        Returns:
            Mean-field Fock matrix.
        """
        pass

    @property
    @abstractmethod
    def xi(self) -> NDArray[float]:
        """Get the shift in the bosonic operators to diagonalise the photon Hamiltonian.

        Returns:
            Shift in the bosonic operators.
        """
        pass

    @property
    def const(self) -> float:
        """Get the shift in energy from moving to the polaritonic basis.

        Returns:
            Constant energy shift due to the polaritonic basis.
        """
        if self.options.shift:
            return util.einsum("I,I->", self.omega, self.xi**2)
        return 0.0

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

    @property
    @abstractmethod
    def nmo(self) -> Any:
        """Get the number of molecular orbitals.

        Returns:
            Number of molecular orbitals.
        """
        pass

    @property
    @abstractmethod
    def nocc(self) -> Any:
        """Get the number of occupied molecular orbitals.

        Returns:
            Number of occupied molecular orbitals.
        """
        pass

    @property
    @abstractmethod
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
        return int(self.omega.shape[0])

    @property
    def e_tot(self) -> float:
        """Get the total energy (mean-field plus correlation).

        Returns:
            Total energy.
        """
        return astype(self.mf.e_tot + self.e_corr, float)

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
