"""Space definition."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, cast

from ebcc import pyscf
from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import types

if TYPE_CHECKING:
    from typing import Optional, Union

    from numpy import bool_, float64
    from numpy.typing import NDArray
    from pyscf.scf.hf import SCF

    from ebcc.cc.base import SpinArrayType
    from ebcc.util import Namespace

    T = float64
    B = bool_
    _slice = slice

# Development note: multiplication of boolean arrays is used in place of logical or bitwise
# AND functions. This is because backends are not guaranteed to support logical or bitwise
# operations via overloaded operators. Similarly, subtraction of boolean arrays is used in
# place of logical or bitwise NOT functions.


class Space:
    """Space class.

    .. code-block:: none

                ─┬─ ┌──────────┐
                 │  │  frozen  │
                 │  ├──────────┤ ─┬─
         virtual │  │  active  │  │
                 │  ├──────────┤  │ correlated
                 │  │ inactive │  │
                ─┼─ ├══════════┤ ─┼─
                 │  │ inactive │  │
                 │  ├──────────┤  │ correlated
        occupied │  │  active  │  │
                 │  ├──────────┤ ─┴─
                 │  │  frozen  │
                ─┴─ └──────────┘
    """

    def __init__(
        self,
        occupied: NDArray[B],
        frozen: NDArray[B],
        active: NDArray[B],
    ) -> None:
        """Initialise the space.

        Args:
            occupied: Array containing boolean flags indicating whether or not each orbital is
                occupied.
            frozen: Array containing boolean flags indicating whether or not each orbital is frozen.
            active: Array containing boolean flags indicating whether or not each orbital is active.
        """
        self._occupied = np.asarray(occupied, dtype=np.bool_)
        self._frozen = np.asarray(frozen, dtype=np.bool_)
        self._active = np.asarray(active, dtype=np.bool_)

        # Checks:
        if not (self._occupied.size == self._frozen.size == self._active.size):
            raise ValueError("The sizes of the space arrays must match.")
        if np.any(np.bitwise_and(self._frozen, self._active)):
            raise ValueError("Frozen and active orbitals must be mutually exclusive.")

    def __repr__(self) -> str:
        """Get a string representation of the space."""
        out = "(%do, %dv)" % (self.nocc, self.nvir)
        parts = []
        if self.nfroz:
            parts.append("(%do, %dv) frozen" % (self.nfocc, self.nfvir))
        if self.nact:
            parts.append("(%do, %dv) active" % (self.naocc, self.navir))
        if len(parts):
            out += " [" + ", ".join(parts) + "]"
        return out

    def size(self, char: str) -> int:
        """Convert a character corresponding to a space to the size of that space.

        Args:
            char: Character to convert.

        Returns:
            Size of the space.
        """
        return {
            "x": self.ncorr,
            "o": self.ncocc,
            "O": self.naocc,
            "i": self.niocc,
            "v": self.ncvir,
            "V": self.navir,
            "a": self.nivir,
        }[char]

    def mask(self, char: str) -> NDArray[B]:
        """Convert a character corresponding to a space to a mask of that space.

        Args:
            char: Character to convert.

        Returns:
            Mask of the space.
        """
        return {
            "x": self.correlated,
            "o": self.correlated_occupied,
            "O": self.active_occupied,
            "i": self.inactive_occupied,
            "v": self.correlated_virtual,
            "V": self.active_virtual,
            "a": self.inactive_virtual,
        }[char]

    @functools.lru_cache(maxsize=128)  # noqa: B019
    def slice(self, char: str) -> _slice:
        """Convert a character corresponding to a space to a slice of that space.

        Args:
            char: Character to convert.

        Returns:
            Slice of the space.
        """
        # Check that the respective mask is contiguous
        mask = self.mask(char)
        first = np.argmax(mask)
        size = self.size(char)
        if not np.all(mask[first : first + size]):
            raise ValueError(
                f"Space '{char}' is not contiguous. In order to slice into this space, "
                "the `mask` method must be used. If you see this error internally, it is "
                "likely that you have constructed a disjoint space. Please reorder the "
                "orbitals in the space."
            )
        return slice(first, first + size)

    def omask(self, char: str) -> NDArray[B]:
        """Like `mask`, but returns only a mask into only the occupied sector.

        Args:
            char: Character to convert.

        Returns:
            Mask of the space.
        """
        return self.mask(char)[self.occupied]

    def vmask(self, char: str) -> NDArray[B]:
        """Like `mask`, but returns only a mask into only the virtual sector.

        Args:
            char: Character to convert.

        Returns:
            Mask of the space.
        """
        return self.mask(char)[self.virtual]

    def oslice(self, char: str) -> _slice:
        """Like `slice`, but returns only a slice into only the occupied sector.

        Args:
            char: Character to convert.

        Returns:
            Slice of the space.
        """
        s = self.slice(char)
        nocc = self.nocc
        return slice(s.start, min(s.stop, nocc))

    def vslice(self, char: str) -> _slice:
        """Like `slice`, but returns only a slice into only the virtual sector.

        Args:
            char: Character to convert.

        Returns:
            Slice of the space.
        """
        s = self.slice(char)
        nocc = self.nocc
        return slice(max(s.start, nocc) - nocc, s.stop - nocc)

    # Full space:

    @property
    def occupied(self) -> NDArray[B]:
        """Get a boolean mask of occupied orbitals."""
        return self._occupied

    @functools.cached_property
    def virtual(self) -> NDArray[B]:
        """Get a boolean mask of virtual orbitals."""
        return np.bitwise_not(self.occupied)

    @property
    def nmo(self) -> int:
        """Get the number of orbitals."""
        return self.occupied.size

    @functools.cached_property
    def nocc(self) -> int:
        """Get the number of occupied orbitals."""
        return cast(int, np.sum(self.occupied))

    @functools.cached_property
    def nvir(self) -> int:
        """Get the number of virtual orbitals."""
        return cast(int, np.sum(self.virtual))

    # Correlated space:

    @functools.cached_property
    def correlated(self) -> NDArray[B]:
        """Get a boolean mask of correlated orbitals."""
        return np.bitwise_not(self.frozen)

    @functools.cached_property
    def correlated_occupied(self) -> NDArray[B]:
        """Get a boolean mask of occupied correlated orbitals."""
        return np.bitwise_and(self.correlated, self.occupied)

    @functools.cached_property
    def correlated_virtual(self) -> NDArray[B]:
        """Get a boolean mask of virtual correlated orbitals."""
        return np.bitwise_and(self.correlated, self.virtual)

    @functools.cached_property
    def ncorr(self) -> int:
        """Get the number of correlated orbitals."""
        return cast(int, np.sum(self.correlated))

    @functools.cached_property
    def ncocc(self) -> int:
        """Get the number of occupied correlated orbitals."""
        return cast(int, np.sum(self.correlated_occupied))

    @functools.cached_property
    def ncvir(self) -> int:
        """Get the number of virtual correlated orbitals."""
        return cast(int, np.sum(self.correlated_virtual))

    # Inactive space:

    @functools.cached_property
    def inactive(self) -> NDArray[B]:
        """Get a boolean mask of inactive orbitals."""
        return np.bitwise_not(self.active)

    @functools.cached_property
    def inactive_occupied(self) -> NDArray[B]:
        """Get a boolean mask of occupied inactive orbitals."""
        return np.bitwise_and(self.inactive, self.occupied)

    @functools.cached_property
    def inactive_virtual(self) -> NDArray[B]:
        """Get a boolean mask of virtual inactive orbitals."""
        return np.bitwise_and(self.inactive, self.virtual)

    @functools.cached_property
    def ninact(self) -> int:
        """Get the number of inactive orbitals."""
        return cast(int, np.sum(self.inactive))

    @functools.cached_property
    def niocc(self) -> int:
        """Get the number of occupied inactive orbitals."""
        return cast(int, np.sum(self.inactive_occupied))

    @functools.cached_property
    def nivir(self) -> int:
        """Get the number of virtual inactive orbitals."""
        return cast(int, np.sum(self.inactive_virtual))

    # Frozen space:

    @property
    def frozen(self) -> NDArray[B]:
        """Get a boolean mask of frozen orbitals."""
        return self._frozen

    @functools.cached_property
    def frozen_occupied(self) -> NDArray[B]:
        """Get a boolean mask of occupied frozen orbitals."""
        return np.bitwise_and(self.frozen, self.occupied)

    @functools.cached_property
    def frozen_virtual(self) -> NDArray[B]:
        """Get a boolean mask of virtual frozen orbitals."""
        return np.bitwise_and(self.frozen, self.virtual)

    @functools.cached_property
    def nfroz(self) -> int:
        """Get the number of frozen orbitals."""
        return cast(int, np.sum(self.frozen))

    @functools.cached_property
    def nfocc(self) -> int:
        """Get the number of occupied frozen orbitals."""
        return cast(int, np.sum(self.frozen_occupied))

    @functools.cached_property
    def nfvir(self) -> int:
        """Get the number of virtual frozen orbitals."""
        return cast(int, np.sum(self.frozen_virtual))

    # Active space:

    @property
    def active(self) -> NDArray[B]:
        """Get a boolean mask of active orbitals."""
        return self._active

    @functools.cached_property
    def active_occupied(self) -> NDArray[B]:
        """Get a boolean mask of occupied active orbitals."""
        return np.bitwise_and(self.active, self.occupied)

    @functools.cached_property
    def active_virtual(self) -> NDArray[B]:
        """Get a boolean mask of virtual active orbitals."""
        return np.bitwise_and(self.active, self.virtual)

    @functools.cached_property
    def nact(self) -> int:
        """Get the number of active orbitals."""
        return cast(int, np.sum(self.active))

    @functools.cached_property
    def naocc(self) -> int:
        """Get the number of occupied active orbitals."""
        return cast(int, np.sum(self.active_occupied))

    @functools.cached_property
    def navir(self) -> int:
        """Get the number of virtual active orbitals."""
        return cast(int, np.sum(self.active_virtual))


if TYPE_CHECKING:
    # Needs to be defined after Space
    RConstructSpaceReturnType = tuple[NDArray[T], NDArray[T], Space]
    UConstructSpaceReturnType = tuple[
        tuple[NDArray[T], NDArray[T]],
        tuple[NDArray[T], NDArray[T]],
        tuple[Space, Space],
    ]


def construct_default_space(mf: SCF) -> Union[RConstructSpaceReturnType, UConstructSpaceReturnType]:
    """Construct a default space.

    Args:
        mf: PySCF mean-field object.

    Returns:
        The molecular orbital coefficients, the molecular orbital occupation numbers, and the
        default space.
    """

    def _construct(mo_occ: NDArray[T]) -> Space:
        """Build the default space."""
        frozen = np.zeros(mo_occ.shape, dtype=np.bool_)
        active = np.zeros(mo_occ.shape, dtype=np.bool_)
        space = Space(
            occupied=mo_occ > 0,
            frozen=frozen,
            active=active,
        )
        return space

    # Construct the default space
    if mf.mo_occ.ndim == 2:
        space_a = _construct(mf.mo_occ[0])
        space_b = _construct(mf.mo_occ[1])
        return mf.mo_coeff, mf.mo_occ, (space_a, space_b)
    else:
        return mf.mo_coeff, mf.mo_occ, _construct(mf.mo_occ)


def construct_fno_space(
    mf: SCF,
    occ_tol: Optional[float] = 1e-5,
    occ_frac: Optional[float] = None,
    amplitudes: Optional[Namespace[SpinArrayType]] = None,
) -> Union[RConstructSpaceReturnType, UConstructSpaceReturnType]:
    """Construct a frozen natural orbital space.

    Args:
        mf: PySCF mean-field object.
        occ_tol: Threshold in the natural orbital occupation numbers.
        occ_frac: Fraction of the natural orbital occupation numbers to be retained. Overrides
            `occ_tol` if both are specified.
        amplitudes: Cluster amplitudes. If provided, use these amplitudes when calculating the MP2
            1RDM.

    Returns:
        The natural orbital coefficients, the natural orbital occupation numbers, and the frozen
        natural orbital space.
    """
    # Get the MP2 1RDM
    solver = pyscf.mp2.MP2(mf)
    dm1: NDArray[T]
    if not amplitudes:
        solver.kernel()
        dm1 = np.asarray(solver.make_rdm1(), dtype=types[float])
    else:
        if isinstance(amplitudes.t2, util.Namespace):
            t2 = (amplitudes.t2.aaaa, amplitudes.t2.abab, amplitudes.t2.bbbb)
            dm1 = np.asarray(solver.make_rdm1(t2=t2), dtype=types[float])
        else:
            dm1 = np.asarray(solver.make_rdm1(t2=amplitudes.t2), dtype=types[float])

    # def _construct(dm1, mo_energy, mo_coeff, mo_occ):
    def _construct(
        dm1: NDArray[T],
        mo_energy: NDArray[T],
        mo_coeff: NDArray[T],
        mo_occ: NDArray[T],
    ) -> RConstructSpaceReturnType:
        # Get the number of occupied orbitals
        nocc = cast(int, np.sum(mo_occ > 0))

        # Calculate the natural orbitals
        n, c = np.linalg.eigh(dm1[nocc:, nocc:])
        n, c = n[::-1], c[:, ::-1]

        # Truncate the natural orbitals
        if occ_frac is None:
            active_vir = n > occ_tol
        else:
            active_vir = np.cumsum(n / np.sum(n)) <= occ_frac
        num_active_vir = cast(int, np.sum(active_vir))

        # Canonicalise the natural orbitals
        fock_vv = np.diag(mo_energy[nocc:])
        fock_vv = util.einsum("ab,au,bv->uv", fock_vv, c, c)
        _, c_can = np.linalg.eigh(fock_vv[active_vir][:, active_vir])

        # Transform the natural orbitals
        no_coeff_avir = util.einsum(
            "pi,iq,qj->pj", mo_coeff[:, nocc:], c[:, :num_active_vir], c_can
        )
        no_coeff_fvir = mo_coeff[:, nocc:] @ c[:, num_active_vir:]
        no_coeff_occ = mo_coeff[:, :nocc]
        no_coeff = np.concatenate((no_coeff_occ, no_coeff_avir, no_coeff_fvir), axis=1)

        # Build the natural orbital space
        active = np.zeros(mo_occ.shape, dtype=np.bool_)
        frozen = np.concatenate(
            (
                np.zeros((nocc + num_active_vir,), dtype=np.bool_),
                np.ones((mo_occ.size - nocc - num_active_vir,), dtype=np.bool_),
            )
        )
        no_space = Space(
            occupied=mo_occ > 0,
            frozen=frozen,
            active=active,
        )

        return no_coeff, mo_occ, no_space

    # Construct the natural orbitals
    if mf.mo_occ.ndim == 2:
        coeff_a, occ_a, space_a = _construct(
            np.asarray(dm1[0], dtype=types[float]),
            np.asarray(mf.mo_energy[0], dtype=types[float]),
            np.asarray(mf.mo_coeff[0], dtype=types[float]),
            np.asarray(mf.mo_occ[0], dtype=types[float]),
        )
        coeff_b, occ_b, space_b = _construct(
            np.asarray(dm1[1], dtype=types[float]),
            np.asarray(mf.mo_energy[1], dtype=types[float]),
            np.asarray(mf.mo_coeff[1], dtype=types[float]),
            np.asarray(mf.mo_occ[1], dtype=types[float]),
        )
        return (coeff_a, coeff_b), (occ_a, occ_b), (space_a, space_b)
    else:
        return _construct(
            np.asarray(dm1, dtype=types[float]),
            np.asarray(mf.mo_energy, dtype=types[float]),
            np.asarray(mf.mo_coeff, dtype=types[float]),
            np.asarray(mf.mo_occ, dtype=types[float]),
        )
