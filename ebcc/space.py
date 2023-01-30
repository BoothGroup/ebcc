"""Space definition.
"""

import numpy as np


class Space:
    """Space class.

    Note: orbitals are not either active or frozen, in the notation of
    `ebcc` active orbitals are active in the correlated sense i.e. the
    triples of CCSDt.

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
        self.occupied = np.asarray(occupied, dtype=bool)
        self.frozen = np.asarray(frozen, dtype=bool)
        self.active = np.asarray(active, dtype=bool)

        assert self.occupied.size == self.frozen.size == self.active.size
        assert not np.any((self.frozen.astype(int) + self.correlated.astype(int)) > 1)

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

    @property
    def virtual(self):
        return ~self.occupied

    @property
    def correlated(self):
        return ~self.frozen

    @property
    def frozen_occupied(self):
        return np.logical_and(self.frozen, self.occupied)

    @property
    def correlated_occupied(self):
        return np.logical_and(self.correlated, self.occupied)

    @property
    def active_occupied(self):
        return np.logical_and(self.active, self.occupied)

    @property
    def frozen_virtual(self):
        return np.logical_and(self.frozen, self.virtual)

    @property
    def correlated_virtual(self):
        return np.logical_and(self.correlated, self.virtual)

    @property
    def active_virtual(self):
        return np.logical_and(self.active, self.virtual)

    @property
    def nmo(self):
        return self.occupied.size

    @property
    def nocc(self):
        return np.sum(self.occupied)

    @property
    def nvir(self):
        return np.sum(self.virtual)

    @property
    def nfroz(self):
        return np.sum(self.frozen)

    @property
    def ncorr(self):
        return np.sum(self.correlated)

    @property
    def nact(self):
        return np.sum(self.active)

    @property
    def nfocc(self):
        return np.sum(self.frozen_occupied)

    @property
    def ncocc(self):
        return np.sum(self.correlated_occupied)

    @property
    def naocc(self):
        return np.sum(self.active_occupied)

    @property
    def nfvir(self):
        return np.sum(self.frozen_virtual)

    @property
    def ncvir(self):
        return np.sum(self.correlated_virtual)

    @property
    def navir(self):
        return np.sum(self.active_virtual)
