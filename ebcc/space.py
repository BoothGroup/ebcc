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

    Note that each space indicated above must be contiguous. For
    non-contiguous spaces in any of these categories, the MO space
    should be reordered to make them contiguous.

    Parameters
    ----------
    occupied : np.ndarray or slice
        Array containing boolean flags indicating whether or not each
        orbital is occupied.
    frozen : np.ndarray or slice
        Array containing boolean flags indicating whether or not each
        orbital is frozen.
    active : np.ndarray or slice
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

        self._set_slices(self._occupied, self._frozen, self._active)

    def _set_slices(self, occupied, frozen, active):
        """Converts the array masks into slices and sets them."""

        assert occupied.size == frozen.size == active.size
        assert not np.any(np.logical_and(frozen, active))

        # Check contiguity of each space.
        def check_contiguous(mask, name):
            where = np.where(mask)[0]
            if len(where):
                if not np.all(where == np.arange(where[0], where[-1] + 1)):
                    raise ValueError("%s space is not contiguous." % name)

        check_contiguous(occupied, "Occupied")
        check_contiguous(np.logical_and(frozen, occupied), "Frozen occupied")
        check_contiguous(np.logical_and(active, occupied), "Active occupied")
        check_contiguous(~occupied, "Virtual")
        check_contiguous(np.logical_and(frozen, ~occupied), "Frozen virtual")
        check_contiguous(np.logical_and(active, ~occupied), "Active virtual")

        self.nfocc = np.sum(np.logical_and(frozen, occupied))
        self.naocc = np.sum(np.logical_and(active, occupied))
        self.niocc = np.sum(np.logical_and.reduce((~active, ~frozen, occupied)))
        self.nfvir = np.sum(np.logical_and(frozen, ~occupied))
        self.navir = np.sum(np.logical_and(active, ~occupied))
        self.nivir = np.sum(np.logical_and.reduce((~active, ~frozen, ~occupied)))

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

    # Full space:

    @property
    def nocc(self):
        return self.nfocc + self.naocc + self.niocc

    @property
    def nvir(self):
        return self.nfvir + self.navir + self.nivir

    @property
    def nmo(self):
        return self.nocc + self.nvir

    @property
    def occ(self):
        return slice(None, self.nocc)

    @property
    def vir(self):
        return slice(self.nocc, None)

    occupied = occ
    virtual = vir

    # Correlated:

    @property
    def ncocc(self):
        return self.naocc + self.niocc

    @property
    def ncvir(self):
        return self.navir + self.nivir

    @property
    def ncorr(self):
        return self.ncocc + self.ncvir

    @property
    def cocc(self):
        return slice(self.nfocc, self.nocc)

    @property
    def cvir(self):
        return slice(self.nocc, self.nocc + self.ncvir)

    @property
    def corr(self):
        return slice(self.nfocc, self.nocc + self.ncvir)

    correlated = corr
    correlated_occupied = cocc
    correlated_virtual = cvir

    # Frozen:

    @property
    def nfroz(self):
        return self.nfocc + self.nfvir

    @property
    def focc(self):
        return slice(None, self.nfocc)

    @property
    def fvir(self):
        return slice(self.nocc + self.ncvir, None)

    @property
    def froz(self):
        raise ValueError("Total frozen space is not contiguous, and has no slice representation.")

    frozen = froz
    frozen_occupied = focc
    frozen_virtual = fvir

    # Active:

    @property
    def nact(self):
        return self.naocc + self.navir

    @property
    def aocc(self):
        return slice(self.nfocc, self.nfocc + self.naocc)

    @property
    def avir(self):
        return slice(self.nocc + self.nivir, self.nocc + self.nivir + self.navir)

    @property
    def act(self):
        raise ValueError("Total active space is not contiguous, and has no slice representation.")

    active = act
    active_occupied = aocc
    active_virtual = avir

    # Inactive:

    @property
    def ninact(self):
        return self.niocc + self.nivir

    @property
    def iocc(self):
        return slice(self.nfocc + self.naocc, self.nocc)

    @property
    def ivir(self):
        return slice(self.nocc, self.nocc + self.nivir)

    @property
    def inact(self):
        return slice(self.nfocc + self.naocc, self.nocc + self.nivir)

    inactive = inact
    inactive_occupied = iocc
    inactive_virtual = ivir
