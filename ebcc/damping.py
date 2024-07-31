"""Damping and DIIS control."""

from pyscf.lib import diis


class DIIS(diis.DIIS):
    """Direct inversion in the iterative subspace."""

    def __init__(self, space=6, min_space=1, damping=0.0):
        super().__init__(incore=True)
        self.verbose = 0
        self.space = space
        self.min_space = min_space
        self.damping = damping

    def update(self, x, xerr=None):
        """Extrapolate a vector."""

        # Extrapolate the vector
        x = super().update(x, xerr=xerr)

        # Apply damping
        if self.damping:
            nd = self.get_num_vec()
            if nd > 1:
                xprev = self.get_vec(self.get_num_vec() - 1)
                x = (1.0 - self.damping) * x + self.damping * xprev

        return x
