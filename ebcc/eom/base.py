"""Base classes for `ebcc.eom`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy
from pyscf import lib

from ebcc import numpy as np
from ebcc import util
from ebcc.core.logging import ANSI
from ebcc.core.precision import types

if TYPE_CHECKING:
    from typing import Any, Callable, Optional

    from numpy import float64, int64
    from numpy.typing import NDArray

    from ebcc.cc.base import BaseEBCC, ERIsInputType, SpaceType, SpinArrayType
    from ebcc.core.ansatz import Ansatz
    from ebcc.util import Namespace

    T = float64

    PickFunctionType = Callable[
        [NDArray[T], NDArray[T], int, dict[str, Any]],
        tuple[NDArray[T], NDArray[T], int],
    ]

# TODO Custom implementation


@dataclass
class BaseOptions:
    """Options for EOM calculations.

    Args:
        nroots: Number of roots to solver for.
        e_tol: Threshold for convergence in the eigenvalues.
        max_iter: Maximum number of iterations.
        max_space: Maximum size of the Lanczos vector space.
        koopmans: Whether to use a Koopmans'-like guess.
        left: Whether to apply the left-hand side of the Hamiltonian.
    """

    nroots: int = 5
    e_tol: float = 1e-6
    max_iter: int = 100
    max_space: int = 12
    koopmans: bool = False
    left: bool = False


class BaseEOM(ABC):
    """Base class for equation-of-motion coupled cluster."""

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
        self.e: NDArray[T] = np.zeros((0,), dtype=types[float])
        self.v: NDArray[T] = np.zeros((0, 0), dtype=types[float])

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
    def amplitudes_to_vector(self, amplitudes: Namespace[SpinArrayType]) -> NDArray[T]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
        pass

    @abstractmethod
    def vector_to_amplitudes(self, vector: NDArray[T]) -> Namespace[SpinArrayType]:
        """Construct amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        pass

    @abstractmethod
    def matvec(
        self,
        vector: NDArray[T],
        eris: Optional[ERIsInputType] = None,
        ints: Optional[Namespace[NDArray[T]]] = None,
        left: bool = False,
    ) -> NDArray[T]:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: State vector to apply the Hamiltonian to.
            eris: Electronic repulsion integrals.
            ints: Intermediate products.
            left: Whether to apply the left-hand side of the Hamiltonian.

        Returns:
            Resulting vector.
        """
        pass

    @abstractmethod
    def matvec_intermediates(
        self, eris: Optional[ERIsInputType] = None, left: bool = False
    ) -> Namespace[NDArray[T]]:
        """Get the intermediates for application of the Hamiltonian to a vector.

        Args:
            eris: Electronic repulsion integrals.
            left: Whether to apply the left-hand side of the Hamiltonian.

        Returns:
            Intermediate products.
        """
        pass

    @abstractmethod
    def diag(self, eris: Optional[ERIsInputType] = None) -> NDArray[T]:
        """Get the diagonal of the Hamiltonian.

        Args:
            eris: Electronic repulsion integrals.

        Returns:
            Diagonal of the Hamiltonian.
        """
        pass

    def get_pick(
        self, guesses: Optional[NDArray[T]] = None, real_system: bool = True
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

            def pick(
                w: NDArray[T], v: NDArray[T], nroots: int, env: dict[str, Any]
            ) -> tuple[NDArray[T], NDArray[T], int]:
                """Pick the eigenvalues."""
                x0 = numpy.asarray(lib.linalg_helper._gen_x0(env["v"], env["xs"]))
                s = numpy.asarray(guesses).conj() @ x0.T
                s = numpy.einsum("pi,pi->i", s.conj(), s)
                arg = numpy.argsort(-s)[:nroots]
                w, v, idx = lib.linalg_helper._eigs_cmplx2real(
                    numpy.asarray(w),
                    numpy.asarray(v),
                    arg,
                    real_system,
                )
                return w, v, idx

        else:

            def pick(
                w: NDArray[T], v: NDArray[T], nroots: int, env: dict[str, Any]
            ) -> tuple[NDArray[T], NDArray[T], int]:
                """Pick the eigenvalues."""
                real_idx = numpy.where(abs(w.imag) < 1e-3)[0]
                w, v, idx = lib.linalg_helper._eigs_cmplx2real(
                    numpy.asarray(w),
                    numpy.asarray(v),
                    real_idx,
                    real_system,
                )
                return w, v, idx

        return pick

    @abstractmethod
    def _argsort_guesses(self, diag: NDArray[T]) -> NDArray[int64]:
        """Sort the diagonal to inform the initial guesses."""
        pass

    @abstractmethod
    def _quasiparticle_weight(self, r1: SpinArrayType) -> T:
        """Get the quasiparticle weight."""
        pass

    def get_guesses(self, diag: Optional[NDArray[T]] = None) -> list[NDArray[T]]:
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
        guesses: list[NDArray[T]] = []
        for root, guess in enumerate(arg[:nroots]):
            guesses.append(np.eye(1, diag.size, guess, dtype=types[float])[0])

        return guesses

    def callback(self, envs: dict[str, Any]) -> None:  # noqa: B027
        """Callback function for the eigensolver."""  # noqa: D401
        pass

    def davidson(
        self,
        eris: Optional[ERIsInputType] = None,
        guesses: Optional[list[NDArray[T]]] = None,
    ) -> NDArray[T]:
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
        ints = self.matvec_intermediates(eris=eris, left=self.options.left)
        matvecs = lambda vs: [
            numpy.asarray(
                self.matvec(np.asarray(v), eris=eris, ints=ints, left=self.options.left)
            )
            for v in vs
        ]
        diag = self.diag(eris=eris)

        # Get the guesses:
        if guesses is None:
            guesses = self.get_guesses(diag=diag)

        # Solve the EOM Hamiltonian:
        nroots = min(len(guesses), self.options.nroots)
        pick = self.get_pick(guesses=np.stack(guesses))
        converged, e, v = lib.davidson_nosym1(
            matvecs,
            [numpy.asarray(g) for g in guesses],
            numpy.asarray(diag),
            tol=self.options.e_tol,
            nroots=nroots,
            pick=pick,
            max_cycle=self.options.max_iter,
            max_space=self.options.max_space,
            callback=self.callback,
            verbose=0,
        )
        e = np.asarray(e)
        v = np.asarray(v)

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
        self.e = e.astype(types[float])
        self.v = np.asarray(v).T.astype(types[float])

        self.log.debug("")
        self.log.output(
            f"{ANSI.B}{'Root':>4s} {'Energy':>16s} {'Weight':>13s} {'Conv.':>8s}{ANSI.R}"
        )
        for n, (en, vn, cn) in enumerate(zip(self.e, self.v.T, converged)):
            r1n = self.vector_to_amplitudes(vn)["r1"]
            qpwt = self._quasiparticle_weight(r1n)
            self.log.output(
                f"{n:>4d} {en.item():>16.10f} {qpwt:>13.5g} "
                f"{[ANSI.r, ANSI.g][bool(cn)]}{cn!r:>8s}{ANSI.R}"
            )

        self.log.debug("")
        self.log.debug("Time elapsed: %s", timer.format_time(timer()))
        self.log.debug("")

        return self.e

    kernel = davidson

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

    def matvec(
        self,
        vector: NDArray[T],
        eris: Optional[ERIsInputType] = None,
        ints: Optional[Namespace[NDArray[T]]] = None,
        left: bool = False,
    ) -> NDArray[T]:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: State vector to apply the Hamiltonian to.
            eris: Electronic repulsion integrals.
            ints: Intermediate products.
            left: Whether to apply the left-hand side of the Hamiltonian.

        Returns:
            Resulting vector.
        """
        if not ints:
            ints = self.matvec_intermediates(eris=eris, left=left)
        amplitudes = self.vector_to_amplitudes(vector)
        func, kwargs = self.ebcc._load_function(
            f"hbar_{'l' if left else ''}matvec_ip",
            eris=eris,
            ints=ints,
            amplitudes=self.ebcc.amplitudes,
            excitations=amplitudes,
        )
        res: Namespace[SpinArrayType] = func(**kwargs)
        res = util.Namespace(**{key.rstrip("new"): val for key, val in res.items()})
        return self.amplitudes_to_vector(res)

    def matvec_intermediates(
        self, eris: Optional[ERIsInputType] = None, left: bool = False
    ) -> Namespace[NDArray[T]]:
        """Get the intermediates for application of the Hamiltonian to a vector.

        Args:
            eris: Electronic repulsion integrals.
            left: Whether to apply the left-hand side of the Hamiltonian.

        Returns:
            Intermediate products.
        """
        func, kwargs = self.ebcc._load_function(
            f"hbar_{'l' if left else ''}matvec_ip_intermediates",
            eris=eris,
            amplitudes=self.ebcc.amplitudes,
        )
        res: Namespace[NDArray[T]] = util.Namespace(**func(**kwargs))
        return res


class BaseEA_EOM(BaseEOM):
    """Base class for electron-affinity EOM-CC."""

    @property
    def excitation_type(self) -> str:
        """Get the type of excitation."""
        return "ea"

    def matvec(
        self,
        vector: NDArray[T],
        eris: Optional[ERIsInputType] = None,
        ints: Optional[Namespace[NDArray[T]]] = None,
        left: bool = False,
    ) -> NDArray[T]:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: State vector to apply the Hamiltonian to.
            eris: Electronic repulsion integrals.
            ints: Intermediate products.
            left: Whether to apply the left-hand side of the Hamiltonian.

        Returns:
            Resulting vector.
        """
        if not ints:
            ints = self.matvec_intermediates(eris=eris, left=left)
        amplitudes = self.vector_to_amplitudes(vector)
        func, kwargs = self.ebcc._load_function(
            f"hbar_{'l' if left else ''}matvec_ea",
            eris=eris,
            ints=ints,
            amplitudes=self.ebcc.amplitudes,
            excitations=amplitudes,
        )
        res: Namespace[SpinArrayType] = func(**kwargs)
        res = util.Namespace(**{key.rstrip("new"): val for key, val in res.items()})
        return self.amplitudes_to_vector(res)

    def matvec_intermediates(
        self, eris: Optional[ERIsInputType] = None, left: bool = False
    ) -> Namespace[NDArray[T]]:
        """Get the intermediates for application of the Hamiltonian to a vector.

        Args:
            eris: Electronic repulsion integrals.
            left: Whether to apply the left-hand side of the Hamiltonian.

        Returns:
            Intermediate products.
        """
        func, kwargs = self.ebcc._load_function(
            f"hbar_{'l' if left else ''}matvec_ea_intermediates",
            eris=eris,
            amplitudes=self.ebcc.amplitudes,
        )
        res: Namespace[NDArray[T]] = util.Namespace(**func(**kwargs))
        return res


class BaseEE_EOM(BaseEOM):
    """Base class for electron-electron EOM-CC."""

    @property
    def excitation_type(self) -> str:
        """Get the type of excitation."""
        return "ee"

    def matvec(
        self,
        vector: NDArray[T],
        eris: Optional[ERIsInputType] = None,
        ints: Optional[Namespace[NDArray[T]]] = None,
        left: bool = False,
    ) -> NDArray[T]:
        """Apply the Hamiltonian to a vector.

        Args:
            vector: State vector to apply the Hamiltonian to.
            eris: Electronic repulsion integrals.
            ints: Intermediate products.
            left: Whether to apply the left-hand side of the Hamiltonian.

        Returns:
            Resulting vector.
        """
        if not ints:
            ints = self.matvec_intermediates(eris=eris, left=left)
        amplitudes = self.vector_to_amplitudes(vector)
        func, kwargs = self.ebcc._load_function(
            f"hbar_{'l' if left else ''}matvec_ee",
            eris=eris,
            ints=ints,
            amplitudes=self.ebcc.amplitudes,
            excitations=amplitudes,
        )
        res: Namespace[SpinArrayType] = func(**kwargs)
        res = util.Namespace(**{key.rstrip("new"): val for key, val in res.items()})
        return self.amplitudes_to_vector(res)

    def matvec_intermediates(
        self, eris: Optional[ERIsInputType] = None, left: bool = False
    ) -> Namespace[NDArray[T]]:
        """Get the intermediates for application of the Hamiltonian to a vector.

        Args:
            eris: Electronic repulsion integrals.
            left: Whether to apply the left-hand side of the Hamiltonian.

        Returns:
            Intermediate products.
        """
        func, kwargs = self.ebcc._load_function(
            f"hbar_{'l' if left else ''}matvec_ee_intermediates",
            eris=eris,
            amplitudes=self.ebcc.amplitudes,
        )
        res: Namespace[NDArray[T]] = util.Namespace(**func(**kwargs))
        return res
