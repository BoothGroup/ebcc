"""Damping and DIIS control."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf.lib import diis

from ebcc import numpy as np
from ebcc import util

if TYPE_CHECKING:
    from typing import Optional

    from numpy import float64
    from numpy.typing import NDArray

    T = float64


class DIIS(diis.DIIS):
    """Direct inversion in the iterative subspace.

    Adapted from PySCF.
    """

    _head: int
    _buffer: dict[str, NDArray[T]]
    _bookkeep: list[int]
    _err_vec_touched: bool
    _H: Optional[NDArray[T]]
    _xprev: Optional[NDArray[T]]

    def __init__(self, space: int = 6, min_space: int = 1, damping: float = 0.0) -> None:
        """Initialize the DIIS object.

        Args:
            space: The number of vectors to store in the DIIS space.
            min_space: The minimum number of vectors to store in the DIIS space.
            damping: The damping factor to apply to the extrapolated vector.
        """
        super().__init__(incore=True)
        self.verbose = 0
        self.space = space
        self.min_space = min_space
        self.damping = damping

    def _store(self, key: str, value: NDArray[T]) -> None:
        """Store the given values in the DIIS buffer."""
        self._buffer[key] = value

    def push_err_vec(self, xerr: NDArray[T]) -> None:
        """Push the error vectors into the DIIS subspace."""
        self._err_vec_touched = True
        if self._head >= self.space:
            self._head = 0
        self._store(f"e{self._head}", xerr)

    def push_vec(self, x: NDArray[T]) -> None:
        """Push the vectors into the DIIS subspace."""
        if len(self._bookkeep) >= self.space:
            self._bookkeep = self._bookkeep[1 - self.space :]

        if self._err_vec_touched:
            self._bookkeep.append(self._head)
            self._store(f"x{self._head}", x)
            self._head += 1
        elif self._xprev is None:
            self._xprev = x
            self._store("xprev", x)
        else:
            if self._head >= self.space:
                self._head = 0
            self._bookkeep.append(self._head)
            self._store(f"x{self._head}", x)
            self._store(f"e{self._head}", x - self._xprev)
            self._head += 1

    def get_err_vec(self, idx: int) -> NDArray[T]:
        """Get the error vectors at the given index."""
        return self._buffer[f"e{idx}"]

    def get_vec(self, idx: int) -> NDArray[T]:
        """Get the vectors at the given index."""
        return self._buffer[f"x{idx}"]

    def get_num_vec(self) -> int:
        """Get the number of vector groups stored in the DIIS object."""
        return len(self._bookkeep)

    def update(self, x: NDArray[T], xerr: Optional[NDArray[T]] = None) -> NDArray[T]:
        """Extrapolate a vector."""
        # Push the vector and error vector into the DIIS subspace
        if xerr is not None:
            self.push_err_vec(xerr)
        self.push_vec(x)

        # Check if the DIIS space is less than the minimum space
        nd = self.get_num_vec()
        if nd < self.min_space:
            return x

        # Build the error matrix
        x1 = self.get_err_vec(self._head - 1)
        if self._H is None:
            self._H = np.block(
                [
                    [np.zeros((1, 1)), np.ones((1, self.space))],
                    [np.ones((self.space, 1)), np.zeros((self.space, self.space))],
                ]
            )
        # this looks crazy, but it's just updating the `self._head`th row and
        # column with the new errors, it's just done this way to avoid using
        # calls to `__setitem__` in immutable backends
        Hi = np.array([np.dot(x1.ravel().conj(), self.get_err_vec(i).ravel()) for i in range(nd)])
        Hi = np.concatenate([np.array([1.0]), Hi, np.zeros(self.space - nd)])
        Hi = Hi.reshape(-1, 1)
        Hj = Hi.T.conj()
        pre = slice(0, self._head)
        pos = slice(self._head + 1, self.space + 1)
        self._H = np.block(
            [
                [self._H[pre, pre], Hi[pre, :], self._H[pre, pos]],
                [Hj[:, pre], Hi[[self._head]].reshape(1, 1), Hj[:, pos]],
                [self._H[pos, pre], Hi[pos, :], self._H[pos, pos]],
            ]
        )

        if self._xprev is None:
            xnew = self.extrapolate(nd)
        else:
            self._xprev = None  # release memory first
            self._xprev = xnew = self.extrapolate(nd)
            self._store("xprev", xnew)

        # Apply damping
        if self.damping:
            nd = self.get_num_vec()
            if nd > 1:
                xprev = self.get_vec(self.get_num_vec() - 1)
                xnew = (1.0 - self.damping) * xnew + self.damping * xprev

        return xnew

    def extrapolate(self, nd: Optional[int] = None) -> NDArray[T]:
        """Extrapolate the next vector."""
        if nd is None:
            nd = self.get_num_vec()
        if nd == 0:
            raise RuntimeError("No vector found in DIIS object.")

        # Get the linear problem to solve
        if self._H is None:
            raise RuntimeError("DIIS object not initialised.")
        h = self._H[: nd + 1, : nd + 1]
        g = np.concatenate([np.ones((1,), h.dtype), np.zeros((nd,), h.dtype)])

        # Solve the linear problem
        w, v = np.linalg.eigh(h)
        mask = np.abs(w) > 1e-14
        c = util.einsum("pi,qi,i,q->p", v[:, mask], v[:, mask].conj(), 1 / w[mask], g)

        # Construct the new vector
        xnew: NDArray[T] = np.zeros_like(self.get_vec(0))
        for i, ci in enumerate(c[1:]):
            xi = self.get_vec(i)
            xnew += xi * ci

        return xnew
