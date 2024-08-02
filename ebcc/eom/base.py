"""Base classes for `ebcc.eom`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyscf import lib

from ebcc import numpy as np
from ebcc import util
from ebcc.logging import ANSI
from ebcc.precision import types

if TYPE_CHECKING:
    from typing import Any, Optional, TypeVar, Union, Callable

    from ebcc.util import Namespace
    from ebcc.numpy.typing import NDArray
    from ebcc.cc.base import BaseEBCC

    ERIsInputType = Union[type[BaseERIs], NDArray[float]]
    AmplitudeType = TypeVar("AmplitudeType")
    PickFunctionType = Callable[
        [NDArray[float], NDArray[float], int, dict[str, Any]],
        tuple[NDArray[float], NDArray[float], int],
    ]


@dataclass
class Options:
    """Options for EOM calculations.

    Args:
        nroots: Number of roots to solver for.
        e_tol: Threshold for convergence in the eigenvalues.
        max_iter: Maximum number of iterations.
        max_space: Maximum size of the Lanczos vector space.
        koopmans: Whether to use a Koopmans'-like guess.
    """

    nroots: int = 5
    e_tol: float = util.Inherited * 1e2
    max_iter: int = util.Inherited
    max_space: int = 12


class BaseEOM(ABC):
    """Base class for equation-of-motion coupled cluster.

    Attributes:
        ebcc: Parent `EBCC` object.
        options: Options for the EBCC calculation.
    """

    # Types
    Options = Options

    def __init__(
        self,
        ebcc: BaseEBCC,
        options: Optional[Options] = None,
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
        self.ansatz = ebcc.anzatz
        self.log = ebcc.log

        # Attributes:
        self.converged = False
        self.e = None
        self.v = None

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
    def name(self) -> str:
        """Get the name of the method."""
        return f"{self.excitation_type.upper()}-EOM-{self.spin_type}{self.ansatz.name}"

    @abstractmethod
    def amplitudes_to_vector(self, *amplitudes: AmplitudesType) -> NDArray[float]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
        pass

    @abstractmethod
    def vector_to_amplitudes(self, vector: NDArray[float]) -> tuple[AmplitudesType, ...]:
        """Construct amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        pass

    @abstractmethod
    def matvec(self, vector: NDArray[float], eris: Optional[ERIsInputType] = None) -> NDArray[float]:
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
    def bras(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the bra vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Bra vectors.
        """
        pass

    @abstractmethod
    def kets(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the ket vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Ket vectors.
        """
        pass

    def dot_braket(self, bra: NDArray[float], ket: NDArray[float]) -> float:
        """Compute the dot product of a bra and ket."""
        return np.dot(bra, ket)

    def get_pick(self, guesses: Optional[NDArray[float]] = None, real_system: bool = True) -> PickFunctionType:
        """Get the function to pick the eigenvalues matching the criteria.

        Args:
            guesses: Initial guesses for the roots.
            real_system: Whether the system is real-valued.

        Returns:
            Function to pick the eigenvalues.
       """
        if self.options.koopmans:
            assert guesses is None
            guesses_array = np.asarray(guesses)

            def pick(w: NDArray[float], v: NDArray[float], nroots: int, env: dict[str, Any]) -> tuple[NDArray[float], NDArray[float], int]:
                """Pick the eigenvalues."""
                x0 = lib.linalg_helper._gen_x0(envs["v"], envs["xs"])
                x0 = np.asarray(x0)
                s = np.dot(g.conj(), x0.T)
                s = util.einsum("pi,qi->i", s.conj(), s)
                idx = np.argsort(-s)[:nroots]
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)

        else:

            def pick(w: NDArray[float], v: NDArray[float], nroots: int, env: dict[str, Any]) -> tuple[NDArray[float], NDArray[float], int]:
                """Pick the eigenvalues."""
                real_idx = np.where(abs(w.imag) < 1e-3)[0]
                w, v, idx = lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
                mask = np.argsort(np.abs(w))
                w, v = w[mask], v[:, mask]
                return w, v, 0

        return pick

    def _argsort_guesses(self, diag: NDArray[float]) -> NDArray[int]:
        """Sort the diagonal to inform the initial guesses."""
        if self.options.koopmans:
            r1 = self.vector_to_amplitudes(diag)[0]
            arg = np.argsort(diag[:r1.size])
        else:
            arg = np.argsort(diag)
        return arg

    def _quasiparticle_weights(self, r1: NDArray[float]) -> float:
        """Get the quasiparticle weight."""
        return np.linalg.norm(r1) ** 2

    def get_guesses(self, diag: NDArray[float]) -> list[NDArray[float]]:
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

    def callback(self, envs: dict[str, Any]) -> None:
        """Callback function for the eigensolver."""  # noqa: D401
        pass

    def davidson(self, eris: Optional[ERIsInputType] = None, guesses: Optional[list[NDArray[float]]] = None) -> NDArray[float]:
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
            r1n = self.vector_to_amplitudes(vn)[0]
            qpwt = self._quasiparticle_weight(r1n)
            self.log.output(
                f"{n:>4d} {en:>16.10f} {qpwt:>13.5g} " f"{[ANSI.r, ANSI.g][cn]}{cn!r:>8s}{ANSI.R}"
            )

        self.log.debug("")
        self.log.debug("Time elapsed: %s", timer.format_time(timer()))
        self.log.debug("")

        return e

    kernel = davidson

    def moments(self, nmom: int, eris: Optional[ERIsInputType] = None, amplitudes: Namespace[AmplitudeType] = None, hermitise: bool = True) -> NDArray[float]:
        """Construct the moments of the EOM Hamiltonian."""
        if eris is None:
            eris = self.ebcc.get_eris()
        if not amplitudes:
            amplitudes = self.ebcc.amplitudes

        bras = self.bras(eris=eris)
        kets = self.kets(eris=eris)

        moments = np.zeros((nmom, self.nmo, self.nmo), dtype=types[float])

        for j in range(self.nmo):
            ket = kets[j]
            for n in range(nmom):
                for i in range(self.nmo):
                    bra = bras[i]
                    moments[n, i, j] = self.dot_braket(bra, ket)
                if n != (nmom - 1):
                    ket = self.matvec(ket, eris=eris)

        if hermitise:
            moments = 0.5 * (moments + moments.swapaxes(1, 2))

        return moments

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


class BaseIPEOM(BaseEOM):
    """Base class for ionization-potential EOM-CC."""

    @property
    def excitation_type(self) -> str:
        """Get the type of excitation."""
        return "ip"

    def amplitudes_to_vector(self, *amplitudes: AmplitudesType) -> NDArray[float]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
        return self.ebcc.excitations_to_vector_ip(*amplitudes)

    def vector_to_amplitudes(self, vector: NDArray[float]) -> tuple[AmplitudesType, ...]:
        """Construct amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        return self.ebcc.vector_to_excitations_ip(vector)

    def matvec(self, vector: NDArray[float], eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: State vector to apply the Hamiltonian to.
            eris: Electronic repulsion integrals.

        Returns:
            Resulting vector.
        """
        amplitudes = self.vector_to_amplitudes(vector)
        result = self.ebcc.hbar_matvec_ip(*amplitudes, eris=eris)
        return self.amplitudes_to_vector(*result)

    def bras(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the bra vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Bra vectors.
        """
        bras_raw = list(self.ebcc.make_ip_mom_bras(eris=eris))
        bras = np.array(
            [self.amplitudes_to_vector(*[b[i] for b in bras_raw]) for i in range(self.nmo)]
        )
        return bras

    def kets(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the ket vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Ket vectors.
        """
        kets_raw = list(self.ebcc.make_ip_mom_kets(eris=eris))
        kets = np.array(
            [self.amplitudes_to_vector(*[k[..., i] for k in kets_raw]) for i in range(self.nmo)]
        )
        return kets


class BaseEAEOM(BaseEOM):
    """Base class for electron-affinity EOM-CC."""

    @property
    def excitation_type(self) -> str:
        """Get the type of excitation."""
        return "ea"

    def amplitudes_to_vector(self, *amplitudes: AmplitudesType) -> NDArray[float]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
        return self.ebcc.excitations_to_vector_ea(*amplitudes)

    def vector_to_amplitudes(self, vector: NDArray[float]) -> tuple[AmplitudesType, ...]:
        """Construct amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        return self.ebcc.vector_to_excitations_ea(vector)

    def matvec(self, vector: NDArray[float], eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: State vector to apply the Hamiltonian to.
            eris: Electronic repulsion integrals.

        Returns:
            Resulting vector.
        """
        amplitudes = self.vector_to_amplitudes(vector)
        result = self.ebcc.hbar_matvec_ea(*amplitudes, eris=eris)
        return self.amplitudes_to_vector(*result)

    def bras(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the bra vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Bra vectors.
        """
        bras_raw = list(self.ebcc.make_ea_mom_bras(eris=eris))
        bras = np.array(
            [self.amplitudes_to_vector(*[b[i] for b in bras_raw]) for i in range(self.nmo)]
        )
        return bras

    def kets(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the ket vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Ket vectors.
        """
        kets_raw = list(self.ebcc.make_ea_mom_kets(eris=eris))
        kets = np.array(
            [self.amplitudes_to_vector(*[k[..., i] for k in kets_raw]) for i in range(self.nmo)]
        )
        return kets


class BaseEEEOM(BaseEOM):
    """Base class for electron-electron EOM-CC."""

    @property
    def excitation_type(self) -> str:
        """Get the type of excitation."""
        return "ee"

    def amplitudes_to_vector(self, *amplitudes: AmplitudesType) -> NDArray[float]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
        return self.ebcc.excitations_to_vector_ee(*amplitudes)

    def vector_to_amplitudes(self, vector: NDArray[float]) -> tuple[AmplitudesType, ...]:
        """Construct amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        return self.ebcc.vector_to_excitations_ee(vector)

    def matvec(self, vector: NDArray[float], eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: State vector to apply the Hamiltonian to.
            eris: Electronic repulsion integrals.

        Returns:
            Resulting vector.
        """
        amplitudes = self.vector_to_amplitudes(vector)
        result = self.ebcc.hbar_matvec_ee(*amplitudes, eris=eris)
        return self.amplitudes_to_vector(*result)

    def bras(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the bra vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Bra vectors.
        """
        bras_raw = list(self.ebcc.make_ee_mom_bras(eris=eris))
        bras = np.array(
            [self.amplitudes_to_vector(*[b[i] for b in bras_raw]) for i in range(self.nmo)]
        )
        return bras

    def kets(self, eris: Optional[ERIsInputType] = None) -> NDArray[float]:
        """Get the ket vectors.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Ket vectors.
        """
        kets_raw = list(self.ebcc.make_ee_mom_kets(eris=eris))
