"""Base classes for `ebcc.eom`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyscf import lib

from ebcc import numpy as np
from ebcc import util
from ebcc.core.logging import ANSI
from ebcc.core.precision import types

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Union

    from ebcc.cc.base import BaseEBCC, ERIsInputType, SpaceType, SpinArrayType
    from ebcc.core.ansatz import Ansatz
    from ebcc.numpy.typing import NDArray
    from ebcc.util import Namespace

    PickFunctionType = Callable[
        [NDArray[float], NDArray[float], int, dict[str, Any]],
        tuple[NDArray[float], NDArray[float], int],
    ]


@dataclass
class BaseOptions:
    """Options for EOM calculations.

    Args:
        nroots: Number of roots to solver for.
        e_tol: Threshold for convergence in the eigenvalues.
        max_iter: Maximum number of iterations.
        max_space: Maximum size of the Lanczos vector space.
        koopmans: Whether to use a Koopmans'-like guess.
    """

    nroots: int = 5
    e_tol: float = 1e-6
    max_iter: int = 100
    max_space: int = 12
    koopmans: bool = False


class BaseEOM(ABC):
    """Base class for equation-of-motion coupled cluster.

    Attributes:
        ebcc: Parent `EBCC` object.
        options: Options for the EBCC calculation.
    """

    # Types
    Options = BaseOptions

    # Attrbutes
    ebcc: BaseEBCC
    space: SpaceType
    ansatz: Ansatz

    def __init__(
        self,
        ebcc: BaseEBCC,
        options: Optional[BaseOptions] = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the EOM object.

        Args:
            ebcc: Parent `EBCC` object.
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
        self.ebcc = ebcc
        self.space = ebcc.space
        self.ansatz = ebcc.ansatz
        self.log = ebcc.log

        # Attributes:
        self.converged = False
        self.e: NDArray[float] = np.empty((0), dtype=types[float])
        self.v: NDArray[float] = np.empty((0, 0), dtype=types[float])

        # Logging:
        self.log.info(f"\n{ANSI.B}{ANSI.U}{self.name}{ANSI.R}")
        self.log.debug(f"{ANSI.B}{'*' * len(self.name)}{ANSI.R}")
        self.log.debug("")
        self.log.info(f"{ANSI.B}Options{ANSI.R}:")
        self.log.info(f" > nroots:  {ANSI.y}{self.options.nroots}{ANSI.R}")
        self.log.info(f" > e_tol:  {ANSI.y}{self.options.e_tol}{ANSI.R}")
        self.log.info(f" > max_iter:  {ANSI.y}{self.options.max_iter}{ANSI.R}")
        self.log.info(f" > max_space:  {ANSI.y}{self.options.max_space}{ANSI.R}")
        self.log.debug("")

    @property
    @abstractmethod
    def excitation_type(self) -> str:
        """Get the type of excitation."""
        pass

    @property
    def spin_type(self) -> str:
        """Get the spin type."""
        return self.ebcc.spin_type

    @property
    def name(self) -> str:
        """Get the name of the method."""
        return f"{self.excitation_type.upper()}-EOM-{self.spin_type}{self.ansatz.name}"

    @abstractmethod
    def amplitudes_to_vector(self, amplitudes: Namespace[SpinArrayType]) -> NDArray[float]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
        pass

    @abstractmethod
    def vector_to_amplitudes(self, vector: NDArray[float]) -> Namespace[SpinArrayType]:
        """Construct amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        pass

    @abstractmethod
    def matvec(
        self, vector: NDArray[float], eris: Optional[ERIsInputType] = None
    ) -> NDArray[float]:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: State vector to apply the Hamiltonian to.
            eris: Electronic repulsion integrals.

        Returns:
            Resulting vector.
        """
        pass

    @abstractmethod
    def diag(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the diagonal of the Hamiltonian.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Diagonal of the Hamiltonian.
        """
        pass

    @abstractmethod
    def bras(self, eris: Optional[ERIsInputType] = None) -> Namespace[SpinArrayType]:
        """Get the bra vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Bra vectors.
        """
        pass

    @abstractmethod
    def kets(self, eris: Optional[ERIsInputType] = None) -> Namespace[SpinArrayType]:
        """Get the ket vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Ket vectors.
        """
        pass

    def dot_braket(self, bra: NDArray[float], ket: NDArray[float]) -> Union[float, NDArray[float]]:
        """Compute the dot product of a bra and ket."""
        return np.dot(bra, ket)

    def get_pick(
        self, guesses: Optional[NDArray[float]] = None, real_system: bool = True
    ) -> PickFunctionType:
        """Get the function to pick the eigenvalues matching the criteria.

        Args:
            guesses: Initial guesses for the roots.
            real_system: Whether the system is real-valued.

        Returns:
            Function to pick the eigenvalues.
        """
        if self.options.koopmans:
            assert guesses is not None
            guesses_array = np.asarray(guesses)

            def pick(
                w: NDArray[float], v: NDArray[float], nroots: int, env: dict[str, Any]
            ) -> tuple[NDArray[float], NDArray[float], int]:
                """Pick the eigenvalues."""
                x0 = lib.linalg_helper._gen_x0(env["v"], env["xs"])
                x0 = np.asarray(x0)
                s = np.dot(guesses_array.conj(), x0.T)
                s = util.einsum("pi,qi->i", s.conj(), s)
                arg = np.argsort(-s)[:nroots]
                return lib.linalg_helper._eigs_cmplx2real(w, v, arg, real_system)  # type: ignore

        else:

            def pick(
                w: NDArray[float], v: NDArray[float], nroots: int, env: dict[str, Any]
            ) -> tuple[NDArray[float], NDArray[float], int]:
                """Pick the eigenvalues."""
                real_idx = np.where(abs(w.imag) < 1e-3)[0]
                w, v, idx = lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
                mask = np.argsort(np.abs(w))
                w, v = w[mask], v[:, mask]
                return w, v, 0

        return pick

    @abstractmethod
    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        pass

    @abstractmethod
    def _quasiparticle_weight(self, r1: SpinArrayType) -> float:
        """Get the quasiparticle weight."""
        pass

    def get_guesses(self, diag: Optional[NDArray[float]] = None) -> list[NDArray[float]]:
        """Get the initial guesses vectors.

        Args:
            diag: Diagonal of the Hamiltonian.

        Returns:
            Initial guesses.
        """
        if diag is None:
            diag = self.diag()
        arg = self._argsort_guesses(diag)

        nroots = min(self.options.nroots, diag.size)
        guesses = np.zeros((nroots, diag.size), dtype=diag.dtype)
        for root, guess in enumerate(arg[:nroots]):
            guesses[root, guess] = 1.0

        return list(guesses)

    def callback(self, envs: dict[str, Any]) -> None:  # noqa: B027
        """Callback function for the eigensolver."""  # noqa: D401
        pass

    def davidson(
        self, eris: Optional[ERIsInputType] = None, guesses: Optional[list[NDArray[float]]] = None
    ) -> NDArray[float]:
        """Solve the EOM Hamiltonian using the Davidson solver.

        Args:
            eris: Electronic repulsion integrals.
            guesses: Initial guesses for the roots.

        Returns:
            Energies of the roots.
        """
        timer = util.Timer()

        # Get the ERIs:
        eris = self.ebcc.get_eris(eris)

        self.log.output(
            "Solving for %s excitations using the Davidson solver.", self.excitation_type.upper()
        )

        # Get the matrix-vector products and the diagonal:
        matvecs = lambda vs: [self.matvec(v, eris=eris) for v in vs]
        diag = self.diag(eris=eris)

        # Get the guesses:
        if guesses is None:
            guesses = self.get_guesses(diag=diag)

        # Solve the EOM Hamiltonian:
        nroots = min(len(guesses), self.options.nroots)
        pick = self.get_pick(guesses=guesses)
        converged, e, v = lib.davidson_nosym1(
            matvecs,
            guesses,
            diag,
            tol=self.options.e_tol,
            nroots=nroots,
            pick=pick,
            max_cycle=self.options.max_iter,
            max_space=self.options.max_space,
            callback=self.callback,
            verbose=0,
        )

        # Check for convergence:
        if all(converged):
            self.log.debug("")
            self.log.output(f"{ANSI.g}Converged.{ANSI.R}")
        else:
            self.log.debug("")
            self.log.warning(
                f"{ANSI.r}Failed to converge {sum(not c for c in converged)} roots.{ANSI.R}"
            )

        # Update attributes:
        self.converged = converged
        self.e = e
        self.v = v

        self.log.debug("")
        self.log.output(
            f"{ANSI.B}{'Root':>4s} {'Energy':>16s} {'Weight':>13s} {'Conv.':>8s}{ANSI.R}"
        )
        for n, (en, vn, cn) in enumerate(zip(e, v, converged)):
            r1n = self.vector_to_amplitudes(vn)["r1"]
            qpwt = self._quasiparticle_weight(r1n)
            self.log.output(
                f"{n:>4d} {en:>16.10f} {qpwt:>13.5g} {[ANSI.r, ANSI.g][bool(cn)]}{cn!r:>8s}{ANSI.R}"
            )

        self.log.debug("")
        self.log.debug("Time elapsed: %s", timer.format_time(timer()))
        self.log.debug("")

        return e

    kernel = davidson

    @abstractmethod
    def moments(
        self,
        nmom: int,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        hermitise: bool = True,
    ) -> SpinArrayType:
        """Construct the moments of the EOM Hamiltonian.

        Args:
            nmom: Number of moments.
            eris: Electronic repulsion integrals.
            amplitudes: Cluster amplitudes.
            hermitise: Hermitise the moments.

        Returns:
            Moments of the EOM Hamiltonian.
        """
        pass

    @property
    def nmo(self) -> Any:
        """Get the number of MOs."""
        return self.ebcc.nmo

    @property
    def nocc(self) -> Any:
        """Get the number of occupied MOs."""
        return self.ebcc.nocc

    @property
    def nvir(self) -> Any:
        """Get the number of virtual MOs."""
        return self.ebcc.nvir


class BaseIP_EOM(BaseEOM):
    """Base class for ionisation-potential EOM-CC."""

    @property
    def excitation_type(self) -> str:
        """Get the type of excitation."""
        return "ip"

    def amplitudes_to_vector(self, amplitudes: Namespace[SpinArrayType]) -> NDArray[float]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
        return self.ebcc.excitations_to_vector_ip(amplitudes)

    def vector_to_amplitudes(self, vector: NDArray[float]) -> Namespace[SpinArrayType]:
        """Construct amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        return self.ebcc.vector_to_excitations_ip(vector)

    def matvec(
        self, vector: NDArray[float], eris: Optional[ERIsInputType] = None
    ) -> NDArray[float]:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: State vector to apply the Hamiltonian to.
            eris: Electronic repulsion integrals.

        Returns:
            Resulting vector.
        """
        amplitudes = self.vector_to_amplitudes(vector)
        result = self.ebcc.hbar_matvec_ip(amplitudes, eris=eris)
        return self.amplitudes_to_vector(result)


class BaseEA_EOM(BaseEOM):
    """Base class for electron-affinity EOM-CC."""

    @property
    def excitation_type(self) -> str:
        """Get the type of excitation."""
        return "ea"

    def amplitudes_to_vector(self, amplitudes: SpinArrayType) -> NDArray[float]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
        return self.ebcc.excitations_to_vector_ea(amplitudes)

    def vector_to_amplitudes(self, vector: NDArray[float]) -> Namespace[SpinArrayType]:
        """Construct amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        return self.ebcc.vector_to_excitations_ea(vector)

    def matvec(
        self, vector: NDArray[float], eris: Optional[ERIsInputType] = None
    ) -> NDArray[float]:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: State vector to apply the Hamiltonian to.
            eris: Electronic repulsion integrals.

        Returns:
            Resulting vector.
        """
        amplitudes = self.vector_to_amplitudes(vector)
        result = self.ebcc.hbar_matvec_ea(amplitudes, eris=eris)
        return self.amplitudes_to_vector(result)


class BaseEE_EOM(BaseEOM):
    """Base class for electron-electron EOM-CC."""

    @property
    def excitation_type(self) -> str:
        """Get the type of excitation."""
        return "ee"

    def amplitudes_to_vector(self, amplitudes: Namespace[SpinArrayType]) -> NDArray[float]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
        return self.ebcc.excitations_to_vector_ee(amplitudes)

    def vector_to_amplitudes(self, vector: NDArray[float]) -> Namespace[SpinArrayType]:
        """Construct amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        return self.ebcc.vector_to_excitations_ee(vector)

    def matvec(
        self, vector: NDArray[float], eris: Optional[ERIsInputType] = None
    ) -> NDArray[float]:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: State vector to apply the Hamiltonian to.
            eris: Electronic repulsion integrals.

        Returns:
            Resulting vector.
        """
        amplitudes = self.vector_to_amplitudes(vector)
        result = self.ebcc.hbar_matvec_ee(amplitudes, eris=eris)
        return self.amplitudes_to_vector(result)
