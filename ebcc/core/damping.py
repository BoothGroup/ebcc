"""Damping and DIIS control."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf.lib import diis

if TYPE_CHECKING:
    from ebcc.numpy.typing import NDArray


class DIIS(diis.DIIS):
    """Direct inversion in the iterative subspace.

    Attributes:
        space: The number of vectors to store in the DIIS space.
        min_space: The minimum number of vectors to store in the DIIS space.
        damping: The damping factor to apply to the extrapolated vector.
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

    def update(self, x: NDArray[float], xerr: NDArray[float] = None) -> NDArray[float]:
        """Extrapolate a vector."""
        x = super().update(x, xerr=xerr)

        # Apply damping
        if self.damping:
            nd = self.get_num_vec()
            if nd > 1:
                xprev = self.get_vec(self.get_num_vec() - 1)
                x = (1.0 - self.damping) * x + self.damping * xprev

        return x
