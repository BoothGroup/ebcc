"""Space definition.
"""

import numpy as np


class Space:
    """Space class.

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
        """Convert a character in the standard `ebcc` notation to the
        size corresponding to this space.  See `ebcc.eris` for details
        on the default slices.
        """

        return {
            "o": self.ncocc,
            "O": self.naocc,
            "i": self.niocc,
            "v": self.ncvir,
            "V": self.navir,
            "a": self.nivir,
        }[char]

    # Full space:

    @property
    def occupied(self):
        return self._occupied

    @property
    def virtual(self):
        return ~self.occupied

    @property
    def nmo(self):
        return self.occupied.size

    @property
    def nocc(self):
        return np.sum(self.occupied)

    @property
    def nvir(self):
        return np.sum(self.virtual)

    # Correlated space:

    @property
    def correlated(self):
        return ~self.frozen

    @property
    def correlated_occupied(self):
        return np.logical_and(self.correlated, self.occupied)

    @property
    def correlated_virtual(self):
        return np.logical_and(self.correlated, self.virtual)

    @property
    def ncorr(self):
        return np.sum(self.correlated)

    @property
    def ncocc(self):
        return np.sum(self.correlated_occupied)

    @property
    def ncvir(self):
        return np.sum(self.correlated_virtual)

    # Inactive space:

    @property
    def inactive(self):
        return ~self.active

    @property
    def inactive_occupied(self):
        return np.logical_and(self.inactive, self.occupied)

    @property
    def inactive_virtual(self):
        return np.logical_and(self.inactive, self.virtual)

    @property
    def ninact(self):
        return np.sum(self.inactive)

    @property
    def niocc(self):
        return np.sum(self.inactive_occupied)

    @property
    def nivir(self):
        return np.sum(self.inactive_virtual)

    # Frozen space:

    @property
    def frozen(self):
        return self._frozen

    @property
    def frozen_occupied(self):
        return np.logical_and(self.frozen, self.occupied)

    @property
    def frozen_virtual(self):
        return np.logical_and(self.frozen, self.virtual)

    @property
    def nfroz(self):
        return np.sum(self.frozen)

    @property
    def nfocc(self):
        return np.sum(self.frozen_occupied)

    @property
    def nfvir(self):
        return np.sum(self.frozen_virtual)

    # Active space:

    @property
    def active(self):
        return self._active

    @property
    def active_occupied(self):
        return np.logical_and(self.active, self.occupied)

    @property
    def active_virtual(self):
        return np.logical_and(self.active, self.virtual)

    @property
    def nact(self):
        return np.sum(self.active)

    @property
    def naocc(self):
        return np.sum(self.active_occupied)

    @property
    def navir(self):
        return np.sum(self.active_virtual)
