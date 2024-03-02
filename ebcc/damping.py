"""Damping and DIIS control.

Adapted from `pyscf.lib.diis.DIIS`.
"""

import scipy.linalg

from ebcc import numpy as np


class DIIS:
    """Direct inversion of the iterative subspace.

    Attributes
    ----------
    space : int
        DIIS subspace size, the maximum number of vectors to be stored.
        Default value is `6`.
    min_space : int
        The minimal size of subspace before DIIS extrapolation. Default
        value is `1`.
    damping : float
        The damping factor too apply. Default value is `0.0`.
    """

    def __init__(self):
        """Initialize the DIIS object."""

        # Options:
        self.space = 6
        self.min_space = 1

        # Attributes:
        self._buffer = {}
        self._bookkeep = []  # keep the ordering of input vectors
        self._head = 0
        self._H = None
        self._xprev = None
        self._err_vec_touched = False

    def _store(self, key, value):
        """Store the value in the buffer."""
        self._buffer[key] = value

    def push_err_vec(self, xerr):
        """Push the error vector into the buffer."""
        self._err_vec_touched = True
        if self._head >= self.space:
            self._head = 0
        self._store(f"e{self._head}", xerr.ravel())

    def push_vec(self, x):
        """Push the vector into the buffer."""

        # Check if the space is full
        if len(self._bookkeep) >= self.space:
            self._bookkeep = self._bookkeep[1 - self.space :]

        # Push the vector into the buffer
        if self._err_vec_touched:
            self._bookkeep.append(self._head)
            self._store(f"x{self._head}", x.ravel())
            self._head += 1
        elif self._xprev is None:
            self._xprev = x
            self._store("xprev", x.ravel())
        else:
            if self._head >= self.space:
                self._head = 0
            self._bookkeep.append(self._head)
            self._store(f"x{self._head}", x.ravel())
            self._store(f"e{self._head}", x.ravel() - self._xprev)
            self._head += 1

    def get_err_vec(self, idx):
        """Get the error vector from the buffer."""
        return self._buffer[f"e{idx}"]

    def get_vec(self, idx):
        """Get the vector from the buffer."""
        return self._buffer[f"x{idx}"]

    def get_num_vec(self):
        """Get the number of vectors in the buffer."""
        return len(self._bookkeep)

    def update(self, x, xerr=None):
        """
        Update the DIIS matrix using a new vector and its corresponding
        error vector.
        """

        # Push the vector and error vector into the buffer
        if xerr is not None:
            self.push_err_vec(xerr)
        self.push_vec(x)

        # Get the number of vectors in the buffer
        nd = self.get_num_vec()
        if nd < self.min_space:
            return x

        # Update the DIIS matrix
        dt = self.get_err_vec(self._head - 1)
        if self._H is None:
            self._H = np.zeros((self.space + 1, self.space + 1), dt.dtype)
            self._H[0, 1:] = self._H[1:, 0] = 1
        for i in range(nd):
            self._H[self._head, i + 1] = dt.conj().dot(self.get_err_vec(i))
            self._H[i + 1, self._head] = self._H[self._head, i + 1].conj()

        # Perform the DIIS extrapolation
        if self._xprev is None:
            xnew = self.extrapolate(nd)
        else:
            self._xprev = xnew = self.extrapolate(nd)
            self._store("xprev", xnew)

        return xnew.reshape(x.shape)

    def extrapolate(self, nd=None):
        """Perform the DIIS extrapolation for a given number of vectors."""

        # Get the number of vectors
        if nd is None:
            nd = self.get_num_vec()
        if nd == 0:
            raise RuntimeError("No vector found in DIIS object.")

        # Get the DIIS matrix for the given number of vectors
        h = self._H[: nd + 1, : nd + 1]
        g = np.zeros(nd + 1, h.dtype)
        g[0] = 1

        # Solve the linear equation
        w, v = scipy.linalg.eigh(h)
        mask = np.abs(w) > 1e-14
        h_inv = np.dot(v[:, mask] / w[mask], v[:, mask].T.conj())
        c = np.dot(h_inv, g)

        # Perform the DIIS extrapolation
        xnew = None
        for i, ci in enumerate(c[1:]):
            xpart = self.get_vec(i) * ci
            if xnew is None:
                xnew = xpart
            else:
                xnew += xpart

        # Apply damping
        if self.damping > 0:
            xnew = self.damp(nd)

        return xnew

    def damp(self, nd=None):
        """
        Apply damping for a given vector number and its previous
        vector.
        """

        # Get the number of vectors
        if nd is None:
            nd = self.get_num_vec()
        if nd == 0:
            raise RuntimeError("No vector found in DIIS object.")
        elif nd == 1:
            return self.get_vec(0)

        # Get the vectors
        x = self.get_vec(nd - 1)
        xprev = self.get_vec(nd - 2)

        # Apply damping
        xnew = (1.0 - self.damping) * x + self.damping * xprev

        return xnew
