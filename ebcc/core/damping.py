"""Damping and DIIS control."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf.lib import diis

if TYPE_CHECKING:
    from typing import Optional

    from numpy import float64
    from numpy.typing import NDArray

    T = float64

# TODO Custom version


class DIIS(diis.DIIS):
    """Direct inversion in the iterative subspace."""

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

    def update(self, x: NDArray[T], xerr: Optional[NDArray[T]] = None) -> NDArray[T]:
        """Extrapolate a vector."""
        x: NDArray[T] = super().update(x, xerr=xerr)

        # Apply damping
        if self.damping:
            nd = self.get_num_vec()
            if nd > 1:
                xprev = self.get_vec(self.get_num_vec() - 1)
                x = (1.0 - self.damping) * x + self.damping * xprev

        return x
