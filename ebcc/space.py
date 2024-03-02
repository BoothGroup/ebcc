"""Space definition."""

from pyscf.mp import MP2

from ebcc import numpy as np
from ebcc import util
from ebcc.precision import types


class Space:
    """
    Space class.

             -  +----------+
             |  |  frozen  |
             |  +----------+  -
    occupied |  |  active  |  |
             |  +----------+  | correlated
             |  | inactive |  |
             -  #==========#  -
             |  | inactive |  |
             |  +----------+  | correlated
     virtual |  |  active  |  |
             |  +----------+  -
             |  |  frozen  |
             -  +----------+

    Parameters
    ----------
    occupied : np.ndarray
        Array containing boolean flags indicating whether or not each
        orbital is occupied.
    frozen : np.ndarray
        Array containing boolean flags indicating whether or not each
        orbital is frozen.
    active : np.ndarray
        Array containing boolean flags indicating whether or not each
        orbital is active.
    """

    def __init__(
        self,
        occupied: np.ndarray,
        frozen: np.ndarray,
        active: np.ndarray,
    ):
        self._occupied = np.asarray(occupied, dtype=bool)
        self._frozen = np.asarray(frozen, dtype=bool)
        self._active = np.asarray(active, dtype=bool)

        assert self._occupied.size == self._frozen.size == self._active.size
        assert not np.any(np.logical_and(self._frozen, self._active))

    def __repr__(self):
        """Return a string representation of the space."""
        out = "(%do, %dv)" % (self.nocc, self.nvir)
        parts = []
        if self.nfroz:
            parts.append("(%do, %dv) frozen" % (self.nfocc, self.nfvir))
        if self.nact:
            parts.append("(%do, %dv) active" % (self.naocc, self.navir))
        if len(parts):
            out += " [" + ", ".join(parts) + "]"
        return out

    def size(self, char):
        """
        Convert a character in the standard `ebcc` notation to the size
        corresponding to this space.  See `ebcc.eris` for details on the
        default slices.

        Parameters
        ----------
        char : str
            The character to convert.

        Returns
        -------
        n : int
            The size of the space.
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

    def mask(self, char):
        """
        Convert a character in the standard `ebcc` notation to the mask
        corresponding to this space.  See `ebcc.eris` for details on the
        default slices.

        Parameters
        ----------
        char : str
            The character to convert.

        Returns
        -------
        mask : np.ndarray
            The mask corresponding to the space.
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

    def slice(self, char):
        """
        Convert a character in the standard `ebcc` notation to the slice
        corresponding to this space.  See `ebcc.eris` for details on the
        default slices.

        Parameters
        ----------
        char : str
            The character to convert.

        Returns
        -------
        slice : slice
            The slice corresponding to the space.

        Raises
        ------
        ValueError
            If the space is not a single contiguous block, and therefore
            cannot be represented as a slice.
        """
        return mask_to_slice(self.mask(char))

    def indexer(self, char):
        """
        Convert a character in the standard `ebcc` notation to an
        indexing object corresponding to this space.  If the space can
        be represented as a slice, then a slice is returned, otherwise
        the array mask.  See `ebcc.eris` for details on the default
        slices.

        Parameters
        ----------
        char : str
            The character to convert.

        Returns
        -------
        indexer : slice or np.ndarray
            The slice or mask corresponding to the space.
        """
        try:
            return self.slice(char)
        except ValueError:
            return self.mask(char)

    def omask(self, char):
        """
        Like `mask`, but returns only a mask into only the occupied sector.

        Parameters
        ----------
        char : str
            The character to convert.

        Returns
        -------
        mask : np.ndarray
            The mask corresponding to the space.
        """
        return self.mask(char)[self.occupied]

    def vmask(self, char):
        """
        Like `mask`, but returns only a mask into only the virtual sector.

        Parameters
        ----------
        char : str
            The character to convert.

        Returns
        -------
        mask : np.ndarray
            The mask corresponding to the space.
        """
        return self.mask(char)[self.virtual]

    # Full space:

    @property
    def occupied(self):
        """Get a boolean mask of occupied orbitals."""
        return self._occupied

    @property
    def virtual(self):
        """Get a boolean mask of virtual orbitals."""
        return ~self.occupied

    @property
    def nmo(self):
        """Get the number of orbitals."""
        return self.occupied.size

    @property
    def nocc(self):
        """Get the number of occupied orbitals."""
        return np.sum(self.occupied)

    @property
    def nvir(self):
        """Get the number of virtual orbitals."""
        return np.sum(self.virtual)

    # Correlated space:

    @property
    def correlated(self):
        """Get a boolean mask of correlated orbitals."""
        return ~self.frozen

    @property
    def correlated_occupied(self):
        """Get a boolean mask of occupied correlated orbitals."""
        return np.logical_and(self.correlated, self.occupied)

    @property
    def correlated_virtual(self):
        """Get a boolean mask of virtual correlated orbitals."""
        return np.logical_and(self.correlated, self.virtual)

    @property
    def ncorr(self):
        """Get the number of correlated orbitals."""
        return np.sum(self.correlated)

    @property
    def ncocc(self):
        """Get the number of occupied correlated orbitals."""
        return np.sum(self.correlated_occupied)

    @property
    def ncvir(self):
        """Get the number of virtual correlated orbitals."""
        return np.sum(self.correlated_virtual)

    # Inactive space:

    @property
    def inactive(self):
        """Get a boolean mask of inactive orbitals."""
        return ~self.active

    @property
    def inactive_occupied(self):
        """Get a boolean mask of occupied inactive orbitals."""
        return np.logical_and(self.inactive, self.occupied)

    @property
    def inactive_virtual(self):
        """Get a boolean mask of virtual inactive orbitals."""
        return np.logical_and(self.inactive, self.virtual)

    @property
    def ninact(self):
        """Get the number of inactive orbitals."""
        return np.sum(self.inactive)

    @property
    def niocc(self):
        """Get the number of occupied inactive orbitals."""
        return np.sum(self.inactive_occupied)

    @property
    def nivir(self):
        """Get the number of virtual inactive orbitals."""
        return np.sum(self.inactive_virtual)

    # Frozen space:

    @property
    def frozen(self):
        """Get a boolean mask of frozen orbitals."""
        return self._frozen

    @property
    def frozen_occupied(self):
        """Get a boolean mask of occupied frozen orbitals."""
        return np.logical_and(self.frozen, self.occupied)

    @property
    def frozen_virtual(self):
        """Get a boolean mask of virtual frozen orbitals."""
        return np.logical_and(self.frozen, self.virtual)

    @property
    def nfroz(self):
        """Get the number of frozen orbitals."""
        return np.sum(self.frozen)

    @property
    def nfocc(self):
        """Get the number of occupied frozen orbitals."""
        return np.sum(self.frozen_occupied)

    @property
    def nfvir(self):
        """Get the number of virtual frozen orbitals."""
        return np.sum(self.frozen_virtual)

    # Active space:

    @property
    def active(self):
        """Get a boolean mask of active orbitals."""
        return self._active

    @property
    def active_occupied(self):
        """Get a boolean mask of occupied active orbitals."""
        return np.logical_and(self.active, self.occupied)

    @property
    def active_virtual(self):
        """Get a boolean mask of virtual active orbitals."""
        return np.logical_and(self.active, self.virtual)

    @property
    def nact(self):
        """Get the number of active orbitals."""
        return np.sum(self.active)

    @property
    def naocc(self):
        """Get the number of occupied active orbitals."""
        return np.sum(self.active_occupied)

    @property
    def navir(self):
        """Get the number of virtual active orbitals."""
        return np.sum(self.active_virtual)


def mask_to_slice(mask):
    """
    Convert a boolean mask to a slice. If not possible, then an
    exception is raised.
    """

    if isinstance(mask, slice):
        return mask

    indices = np.where(mask > 0)[0]
    differences = np.diff(indices)

    if np.any(differences != 1):
        raise ValueError

    return slice(min(indices), max(indices) + 1)


def construct_default_space(mf):
    """
    Construct a default space.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        PySCF mean-field object.

    Returns
    -------
    mo_coeff : np.ndarray
        The molecular orbital coefficients.
    mo_occ : np.ndarray
        The molecular orbital occupation numbers.
    space : Space
        The default space.
    """

    def _construct(mo_occ):
        # Build the default space
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
        space = (space_a, space_b)
    else:
        space = _construct(mf.mo_occ)

    return mf.mo_coeff, mf.mo_occ, space


def construct_fno_space(mf, occ_tol=1e-5, occ_frac=None, amplitudes=None):
    """
    Construct a frozen natural orbital space.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        PySCF mean-field object.
    occ_tol : float, optional
        Threshold in the natural orbital occupation numbers. Default
        value is `1e-5`.
    occ_frac : float, optional
        Fraction of the natural orbital occupation numbers to be
        retained. Overrides `occ_tol` if both are specified. Default
        value is `None`.
    amplitudes : Namespace, optional
        Cluster amplitudes. If provided, use these amplitudes when
        calculating the MP2 1RDM. Default value is `None`.

    Returns
    -------
    no_coeff : np.ndarray
        The natural orbital coefficients.
    no_occ : np.ndarray
        The natural orbital occupation numbers.
    no_space : Space
        The frozen natural orbital space.
    """

    # Get the MP2 1RDM
    solver = MP2(mf)
    if amplitudes is None:
        solver.kernel()
        dm1 = solver.make_rdm1()
    else:
        if isinstance(amplitudes.t2, util.Namespace):
            t2 = (amplitudes.t2.aaaa, amplitudes.t2.abab, amplitudes.t2.bbbb)
            dm1 = solver.make_rdm1(t2=t2)
        else:
            dm1 = solver.make_rdm1(t2=amplitudes.t2)

    def _construct(dm1, mo_energy, mo_coeff, mo_occ):
        # Get the number of occupied orbitals
        nocc = np.sum(mo_occ > 0)

        # Calculate the natural orbitals
        n, c = np.linalg.eigh(dm1[nocc:, nocc:])
        n, c = n[::-1], c[:, ::-1]

        # Truncate the natural orbitals
        if occ_frac is None:
            active_vir = n > occ_tol
        else:
            active_vir = np.cumsum(n / np.sum(n)) <= occ_frac
        num_active_vir = np.sum(active_vir)

        # Canonicalise the natural orbitals
        fock_vv = np.diag(mo_energy[nocc:]).astype(types[float])
        fock_vv = util.einsum("ab,au,bv->uv", fock_vv, c, c)
        _, c_can = np.linalg.eigh(fock_vv[active_vir][:, active_vir])

        # Transform the natural orbitals
        no_coeff_avir = np.linalg.multi_dot((mo_coeff[:, nocc:], c[:, :num_active_vir], c_can))
        no_coeff_fvir = np.dot(mo_coeff[:, nocc:], c[:, num_active_vir:])
        no_coeff_occ = mo_coeff[:, :nocc]
        no_coeff = np.hstack((no_coeff_occ, no_coeff_avir, no_coeff_fvir)).astype(types[float])

        # Build the natural orbital space
        frozen = np.zeros_like(mo_occ, dtype=bool)
        frozen[nocc + num_active_vir :] = True
        no_space = Space(
            occupied=mo_occ > 0,
            frozen=frozen,
            active=np.zeros_like(mo_occ, dtype=bool),
        )

        return no_coeff, no_space

    # Construct the natural orbitals
    if np.ndim(mf.mo_occ) == 2:
        no_coeff_a, no_space_a = _construct(dm1[0], mf.mo_energy[0], mf.mo_coeff[0], mf.mo_occ[0])
        no_coeff_b, no_space_b = _construct(dm1[1], mf.mo_energy[1], mf.mo_coeff[1], mf.mo_occ[1])
        no_coeff = (no_coeff_a, no_coeff_b)
        no_space = (no_space_a, no_space_b)
    else:
        no_coeff, no_space = _construct(dm1, mf.mo_energy, mf.mo_coeff, mf.mo_occ)

    return no_coeff, mf.mo_occ, no_space
