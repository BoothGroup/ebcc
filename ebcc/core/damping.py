"""Damping and DIIS control."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util

if TYPE_CHECKING:
    from typing import Optional

    from numpy import float64
    from numpy.typing import NDArray

    T = float64


class DIIS:
    """Direct inversion in the iterative subspace.

    Notes:
        This code is adapted from PySCF.
    """

    # Intermediates
    _index: int
    _indices: list[int]
    _arrays: dict[int, NDArray[T]]
    _errors: dict[int, NDArray[T]]
    _matrix: Optional[NDArray[T]]

    def __init__(self, space: int = 6, min_space: int = 1, damping: float = 0.0) -> None:
        """Initialize the DIIS object.

        Args:
            space: The number of vectors to store in the DIIS space.
            min_space: The minimum number of vectors to store in the DIIS space.
            damping: The damping factor to apply to the extrapolated vector.
        """
        # Options
        self.space = space
        self.min_space = min_space
        self.damping = damping

        # Intermediates
        self._index = 0
        self._indices = []
        self._arrays = {}
        self._errors = {}
        self._matrix = None

    def push(self, x: NDArray[T], xerr: Optional[NDArray[T]] = None) -> None:
        """Push the vectors and error vectors into the DIIS subspace.

        Args:
            x: The vector to push into the DIIS subspace.
            xerr: The error vector to push into the DIIS subspace.
        """
        if len(self._indices) >= self.space:
            self._indices = self._indices[1 - self.space :]

        if xerr is not None:
            if self._index >= self.space:
                self._index = 0
            self._errors[self._index] = xerr
            self._indices.append(self._index)
            self._arrays[self._index] = x
            self._index += 1
        elif -1 not in self._arrays:
            self._arrays[-1] = x
        else:
            if self._index >= self.space:
                self._index = 0
            self._indices.append(self._index)
            self._arrays[self._index] = x
            self._errors[self._index] = x - self._arrays[-1]
            self._index += 1

    @property
    def narrays(self) -> int:
        """Get the number of arrays stored in the DIIS object."""
        return len(self._indices)

    def update(self, x: NDArray[T], xerr: Optional[NDArray[T]] = None) -> NDArray[T]:
        """Extrapolate a vector.

        Args:
            x: The vector to extrapolate.
            xerr: The error vector to extrapolate.

        Returns:
            The extrapolated vector.
        """
        # Push the vector and error vector into the DIIS subspace
        self.push(x, xerr)

        # Check if the DIIS space is less than the minimum space
        nd = self.narrays
        if nd < self.min_space:
            return x

        # Build the error matrix
        x1 = self._errors[self._index - 1]
        if self._matrix is None:
            self._matrix = np.block(
                [
                    [np.zeros((1, 1)), np.ones((1, self.space))],
                    [np.ones((self.space, 1)), np.zeros((self.space, self.space))],
                ]
            )
        # this looks crazy, but it's just updating the `self._index`th row and
        # column with the new errors, it's just done this way to avoid using
        # calls to `__setitem__` in immutable backends
        m_i = np.array([
            np.ravel(np.dot(np.conj(np.ravel(x1)), np.ravel(self._errors[i])))[0]
            for i in range(nd)
        ])
        m_i = np.concatenate([np.array([1.0]), m_i, np.zeros(self.space - nd)])
        m_i = np.reshape(m_i, (-1, 1))
        m_j = np.conj(np.transpose(m_i))
        pre = slice(0, self._index)
        pos = slice(self._index + 1, self.space + 1)
        self._matrix = np.block(
            [
                [self._matrix[pre, pre], m_i[pre, :], self._matrix[pre, pos]],
                [m_j[:, pre], np.reshape(m_i[self._index, :], (1, 1)), m_j[:, pos]],
                [self._matrix[pos, pre], m_i[pos, :], self._matrix[pos, pos]],
            ]
        )

        xnew = self.extrapolate(nd)
        self._arrays[-1] = xnew

        # Apply damping
        if self.damping:
            nd = self.narrays
            if nd > 1:
                xprev = self._arrays[self.narrays - 1]
                xnew = (1.0 - self.damping) * xnew + self.damping * xprev

        return xnew

    def extrapolate(self, nd: Optional[int] = None) -> NDArray[T]:
        """Extrapolate the next vector.

        Args:
            nd: The number of arrays to use in the extrapolation.

        Returns:
            The extrapolated vector.
        """
        if nd is None:
            nd = self.narrays
        if nd == 0:
            raise RuntimeError("No vector found in DIIS object.")

        # Get the linear problem to solve
        if self._matrix is None:
            raise RuntimeError("DIIS object not initialised.")
        h = self._matrix[: nd + 1, : nd + 1]
        g = np.concatenate([np.ones((1,), h.dtype), np.zeros((nd,), h.dtype)])

        # Solve the linear problem
        w, v = np.linalg.eigh(h)
        if np.any(np.abs(w) < 1e-14):
            mask = np.abs(w) > 1e-14
            w, v = w[mask], v[:, mask]
        c = util.einsum("pi,qi,i,q->p", v, np.conj(v), w ** -1, g)

        # Construct the new vector
        xnew: NDArray[T] = np.zeros_like(self._arrays[0])
        for i in range(nd):
            xi = self._arrays[i]
            xnew += xi * c[i + 1]

        return xnew
