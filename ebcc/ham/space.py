"""Space definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pyscf.mp import MP2

from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import types

if TYPE_CHECKING:
    from typing import Optional, Union

    from pyscf.scf.hf import SCF

    from ebcc.cc.base import SpinArrayType
    from ebcc.numpy.typing import NDArray
    from ebcc.util import Namespace


class Space:
    """Space class.

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

    Args:
        occupied: Array containing boolean flags indicating whether or not each orbital is occupied.
        frozen: Array containing boolean flags indicating whether or not each orbital is frozen.
        active: Array containing boolean flags indicating whether or not each orbital is active.
    """

    def __init__(
        self,
        occupied: NDArray[bool],
        frozen: NDArray[bool],
        active: NDArray[bool],
    ) -> None:
        """Initialise the space.

        Args:
            occupied: Array containing boolean flags indicating whether or not each orbital is
                occupied.
            frozen: Array containing boolean flags indicating whether or not each orbital is frozen.
            active: Array containing boolean flags indicating whether or not each orbital is active.
        """
        self._occupied = np.asarray(occupied, dtype=bool)
        self._frozen = np.asarray(frozen, dtype=bool)
        self._active = np.asarray(active, dtype=bool)

        # Checks:
        if not (self._occupied.size == self._frozen.size == self._active.size):
            raise ValueError("The sizes of the space arrays must match.")
        if np.any(np.logical_and(self._frozen, self._active)):
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

    def mask(self, char: str) -> NDArray[bool]:
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

    def omask(self, char: str) -> NDArray[bool]:
        """Like `mask`, but returns only a mask into only the occupied sector.

        Args:
            char: Character to convert.

        Returns:
            Mask of the space.
        """
        return self.mask(char)[self.occupied]

    def vmask(self, char: str) -> NDArray[bool]:
        """Like `mask`, but returns only a mask into only the virtual sector.

        Args:
            char: Character to convert.

        Returns:
            Mask of the space.
        """
        return self.mask(char)[self.virtual]

    # Full space:

    @property
    def occupied(self) -> NDArray[bool]:
        """Get a boolean mask of occupied orbitals."""
        return self._occupied

    @property
    def virtual(self) -> NDArray[bool]:
        """Get a boolean mask of virtual orbitals."""
        return ~self.occupied

    @property
    def nmo(self) -> int:
        """Get the number of orbitals."""
        return cast(int, self.occupied.size)

    @property
    def nocc(self) -> int:
        """Get the number of occupied orbitals."""
        return cast(int, np.sum(self.occupied))

    @property
    def nvir(self) -> int:
        """Get the number of virtual orbitals."""
        return cast(int, np.sum(self.virtual))

    # Correlated space:

    @property
    def correlated(self) -> NDArray[bool]:
        """Get a boolean mask of correlated orbitals."""
        return ~self.frozen

    @property
    def correlated_occupied(self) -> NDArray[bool]:
        """Get a boolean mask of occupied correlated orbitals."""
        return np.logical_and(self.correlated, self.occupied)

    @property
    def correlated_virtual(self) -> NDArray[bool]:
        """Get a boolean mask of virtual correlated orbitals."""
        return np.logical_and(self.correlated, self.virtual)

    @property
    def ncorr(self) -> int:
        """Get the number of correlated orbitals."""
        return cast(int, np.sum(self.correlated))

    @property
    def ncocc(self) -> int:
        """Get the number of occupied correlated orbitals."""
        return cast(int, np.sum(self.correlated_occupied))

    @property
    def ncvir(self) -> int:
        """Get the number of virtual correlated orbitals."""
        return cast(int, np.sum(self.correlated_virtual))

    # Inactive space:

    @property
    def inactive(self) -> NDArray[bool]:
        """Get a boolean mask of inactive orbitals."""
        return ~self.active

    @property
    def inactive_occupied(self) -> NDArray[bool]:
        """Get a boolean mask of occupied inactive orbitals."""
        return np.logical_and(self.inactive, self.occupied)

    @property
    def inactive_virtual(self) -> NDArray[bool]:
        """Get a boolean mask of virtual inactive orbitals."""
        return np.logical_and(self.inactive, self.virtual)

    @property
    def ninact(self) -> int:
        """Get the number of inactive orbitals."""
        return cast(int, np.sum(self.inactive))

    @property
    def niocc(self) -> int:
        """Get the number of occupied inactive orbitals."""
        return cast(int, np.sum(self.inactive_occupied))

    @property
    def nivir(self) -> int:
        """Get the number of virtual inactive orbitals."""
        return cast(int, np.sum(self.inactive_virtual))

    # Frozen space:

    @property
    def frozen(self) -> NDArray[bool]:
        """Get a boolean mask of frozen orbitals."""
        return self._frozen

    @property
    def frozen_occupied(self) -> NDArray[bool]:
        """Get a boolean mask of occupied frozen orbitals."""
        return np.logical_and(self.frozen, self.occupied)

    @property
    def frozen_virtual(self) -> NDArray[bool]:
        """Get a boolean mask of virtual frozen orbitals."""
        return np.logical_and(self.frozen, self.virtual)

    @property
    def nfroz(self) -> int:
        """Get the number of frozen orbitals."""
        return cast(int, np.sum(self.frozen))

    @property
    def nfocc(self) -> int:
        """Get the number of occupied frozen orbitals."""
        return cast(int, np.sum(self.frozen_occupied))

    @property
    def nfvir(self) -> int:
        """Get the number of virtual frozen orbitals."""
        return cast(int, np.sum(self.frozen_virtual))

    # Active space:

    @property
    def active(self) -> NDArray[bool]:
        """Get a boolean mask of active orbitals."""
        return self._active

    @property
    def active_occupied(self) -> NDArray[bool]:
        """Get a boolean mask of occupied active orbitals."""
        return np.logical_and(self.active, self.occupied)

    @property
    def active_virtual(self) -> NDArray[bool]:
        """Get a boolean mask of virtual active orbitals."""
        return np.logical_and(self.active, self.virtual)

    @property
    def nact(self) -> int:
        """Get the number of active orbitals."""
        return cast(int, np.sum(self.active))

    @property
    def naocc(self) -> int:
        """Get the number of occupied active orbitals."""
        return cast(int, np.sum(self.active_occupied))

    @property
    def navir(self) -> int:
        """Get the number of virtual active orbitals."""
        return cast(int, np.sum(self.active_virtual))


if TYPE_CHECKING:
    # Needs to be defined after Space
    RConstructSpaceReturnType = tuple[NDArray[float], NDArray[float], Space]
    UConstructSpaceReturnType = tuple[
        tuple[NDArray[float], NDArray[float]],
        tuple[NDArray[float], NDArray[float]],
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

    def _construct(mo_occ: NDArray[float]) -> Space:
        """Build the default space."""
        frozen = np.zeros_like(mo_occ, dtype=bool)
        active = np.zeros_like(mo_occ, dtype=bool)
        space = Space(
            occupied=mo_occ > 0,
            frozen=frozen,
            active=active,
        )
        return space

    # Construct the default space
    if np.ndim(mf.mo_occ) == 2:
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
    solver = MP2(mf)
    if not amplitudes:
        solver.kernel()
        dm1 = solver.make_rdm1()
    else:
        if isinstance(amplitudes.t2, util.Namespace):
            t2 = (amplitudes.t2.aaaa, amplitudes.t2.abab, amplitudes.t2.bbbb)
            dm1 = solver.make_rdm1(t2=t2)
        else:
            dm1 = solver.make_rdm1(t2=amplitudes.t2)

    # def _construct(dm1, mo_energy, mo_coeff, mo_occ):
    def _construct(
        dm1: NDArray[float],
        mo_energy: NDArray[float],
        mo_coeff: NDArray[float],
        mo_occ: NDArray[float],
    ) -> RConstructSpaceReturnType:
        # Get the number of occupied orbitals
        nocc = int(np.sum(mo_occ > 0))

        # Calculate the natural orbitals
        n, c = np.linalg.eigh(dm1[nocc:, nocc:])
        n, c = n[::-1], c[:, ::-1]

        # Truncate the natural orbitals
        if occ_frac is None:
            active_vir = n > occ_tol
        else:
            active_vir = np.cumsum(n / np.sum(n)) <= occ_frac
        num_active_vir = int(np.sum(active_vir))

        # Canonicalise the natural orbitals
        fock_vv = np.diag(mo_energy[nocc:])
        fock_vv = util.einsum("ab,au,bv->uv", fock_vv, c, c)
        _, c_can = np.linalg.eigh(fock_vv[active_vir][:, active_vir])

        # Transform the natural orbitals
        no_coeff_avir = np.linalg.multi_dot((mo_coeff[:, nocc:], c[:, :num_active_vir], c_can))
        no_coeff_fvir = np.dot(mo_coeff[:, nocc:], c[:, num_active_vir:])
        no_coeff_occ = mo_coeff[:, :nocc]
        no_coeff: NDArray[float] = np.hstack((no_coeff_occ, no_coeff_avir, no_coeff_fvir))

        # Build the natural orbital space
        frozen = np.zeros_like(mo_occ, dtype=bool)
        frozen[nocc + num_active_vir :] = True
        no_space = Space(
            occupied=mo_occ > 0,
            frozen=frozen,
            active=np.zeros_like(mo_occ, dtype=bool),
        )

        return no_coeff, mo_occ, no_space

    # Construct the natural orbitals
    if np.ndim(mf.mo_occ) == 2:
        coeff_a, occ_a, space_a = _construct(
            dm1[0].astype(types[float]),
            mf.mo_energy[0].astype(types[float]),
            mf.mo_coeff[0].astype(types[float]),
            mf.mo_occ[0].astype(types[float]),
        )
        coeff_b, occ_b, space_b = _construct(
            dm1[1].astype(types[float]),
            mf.mo_energy[1].astype(types[float]),
            mf.mo_coeff[1].astype(types[float]),
            mf.mo_occ[1].astype(types[float]),
        )
        return (coeff_a, coeff_b), (occ_a, occ_b), (space_a, space_b)
    else:
        return _construct(
            dm1.astype(types[float]),
            mf.mo_energy.astype(types[float]),
            mf.mo_coeff.astype(types[float]),
            mf.mo_occ.astype(types[float]),
        )
