"""Damping and DIIS control."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from pyscf.lib import diis

from ebcc import util
from ebcc import numpy as np
from ebcc.core.tensor import Tensor, einsum, zeros_like
from ebcc.util import Namespace

if TYPE_CHECKING:
    from typing import Optional

NDArray = np.typing.NDArray  # FIXME 
T = TypeVar("T", NDArray[float], Tensor[float])


class DIIS(diis.DIIS, Generic[T]):
    """Direct inversion in the iterative subspace.

    Adapted from PySCF.
    """

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

    def _store(self, key: str, value: tuple[T]) -> None:
        """Store the given values in the DIIS buffer."""
        self._buffer[key] = value

    def push_err_vec(self, xerr: tuple[T]) -> None:
        """Push the error vectors into the DIIS subspace."""
        self._err_vec_touched = True
        if self._head >= self.space:
            self._head = 0
        self._store(f"e{self._head}", xerr)

    def push_vec(self, x: tuple[T]) -> None:
        """Push the vectors into the DIIS subspace."""
        if len(self._bookkeep) >= self.space:
            self._bookkeep = self._bookkeep[1-self.space:]

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
            self._store(f"e{self._head}", tuple(i - j for i, j in zip(x, self._xprev)))
            self._head += 1

    def get_err_vec(self, idx: int) -> tuple[T]:
        """Get the error vectors at the given index."""
        return self._buffer[f"e{idx}"]

    def get_vec(self, idx: int) -> tuple[T]:
        """Get the vectors at the given index."""
        return self._buffer[f"x{idx}"]

    def get_num_vec(self) -> int:
        """Get the number of vector groups stored in the DIIS object."""
        return len(self._bookkeep)

    def update(self, x: tuple[T], xerr: Optional[tuple[T]] = None) -> tuple[T]:
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
            self._H = np.zeros((self.space+1, self.space+1), x1[0].dtype)
            self._H[0, 1:] = self._H[1:, 0] = 1
        for i in range(nd):
            x2 = self.get_err_vec(i)
            tmp = sum((x1i.conj() * x2i).sum() for x1i, x2i in zip(x1, x2))
            self._H[self._head, i+1] = tmp
            self._H[i+1, self._head] = tmp.conj()
        dt = None

        if self._xprev is None:
            xnew = self.extrapolate(nd)
        else:
            self._xprev = None # release memory first
            self._xprev = xnew = self.extrapolate(nd)
            self._store("xprev", xnew)

        # Apply damping  # TODO use _xprev
        if self.damping:
            nd = self.get_num_vec()
            if nd > 1:
                xprev = self.get_vec(self.get_num_vec() - 1)
                x = (1.0 - self.damping) * x + self.damping * xprev

        return xnew

    def extrapolate(self, nd: int = None) -> tuple[T]:
        """Extrapolate the next vector."""
        if nd is None:
            nd = self.get_num_vec()
        if nd == 0:
            raise RuntimeError('No vector found in DIIS object.')

        # Get the linear problem to solve
        h = self._H[:nd+1, :nd+1]
        g = np.zeros(nd+1, h.dtype)
        g[0] = 1

        # Solve the linear problem
        w, v = np.linalg.eigh(h)
        mask = np.abs(w) > 1e-14
        c = util.einsum("pi,qi,i,q->p", v[:, mask], v[:, mask].conj(), 1 / w[mask], g)

        # Construct the new vector
        xnew: list[T] = list(self._zeros_like(self.get_vec(0)))
        for i, ci in enumerate(c[1:]):
            xi = self.get_vec(i)
            for j in range(len(xnew)):
                xnew[j] += xi[j] * ci

        return tuple(xnew)

    def _zeros_like(self, x: tuple[T]) -> tuple[T]:
        """Create a new array with the same shape as the given array."""
        if isinstance(x[0], np.ndarray):
            return tuple(np.zeros_like(xi) for xi in x)
        elif isinstance(x[0], Tensor):
            return tuple(zeros_like(xi) for xi in x)
        else:
            raise TypeError("Invalid type for DIIS object.")
