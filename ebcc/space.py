"""Space definition."""

from ebcc import numpy as np


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
            "o": self.correlated_occupied,
            "O": self.active_occupied,
            "i": self.inactive_occupied,
            "v": self.correlated_virtual,
            "V": self.active_virtual,
            "a": self.inactive_virtual,
        }[char]

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
