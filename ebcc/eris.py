"""Electronic repulsion integral containers.
"""

import types
from typing import Sequence

import numpy as np
from pyscf import ao2mo

from ebcc import util


class RERIs(types.SimpleNamespace):
    """Electronic repulsion integral container class. Consists of a
    just-in-time namespace containing blocks of the integrals, with
    keys that are length-4 strings of `"o"` or `"v"` signifying
    whether the corresponding dimension is occupied or virtual.
    Additionally, capital letters letter `"O"` or `"V"` signify
    active orbitals, whereas lowercase correlated plus active.
    """

    def __init__(
        self,
        ebcc: util.AbstractEBCC,
        array: np.ndarray = None,
        slices: Sequence[slice] = None,
        mo_coeff: np.ndarray = None,
    ):
        self.mf = ebcc.mf
        self.space = ebcc.space
        self.slices = slices
        self.mo_coeff = mo_coeff
        self.array = array

        if self.mo_coeff is None:
            self.mo_coeff = ebcc.mo_coeff
        if not (isinstance(self.mo_coeff, (tuple, list)) or self.mo_coeff.ndim == 3):
            self.mo_coeff = [self.mo_coeff] * 4

        if self.slices is None:
            self.slices = {
                "x": self.space.correlated,
                "o": self.space.correlated_occupied,
                "v": self.space.correlated_virtual,
                "X": self.space.active,
                "O": self.space.active_occupied,
                "V": self.space.active_virtual,
            }
        if not isinstance(self.slices, (tuple, list)):
            self.slices = [self.slices] * 4

    def __getattr__(self, key: str) -> np.ndarray:
        """Just-in-time attribute getter."""

        if self.array is None:
            if key not in self.__dict__.keys():
                coeffs = []
                for i, k in enumerate(key):
                    coeffs.append(self.mo_coeff[i][:, self.slices[i][k]])
                block = ao2mo.incore.general(self.mf._eri, coeffs, compact=False)
                block = block.reshape([c.shape[-1] for c in coeffs])
                self.__dict__[key] = block
            return self.__dict__[key]
        else:
            slices = []
            for i, k in enumerate(key):
                slices.append(self.slices[i][k])
            si, sj, sk, sl = slices
            block = self.array[si][:, sj][:, :, sk][:, :, :, sl]
            return block


class UERIs(types.SimpleNamespace):
    """Electronic repulsion integral container class. Consists of a
    namespace with keys that are length-4 string of `"a"` or `"b"`
    signifying whether the corresponding dimension is alpha or beta
    spin, and values are of type `rebcc.ERIs`.
    """

    def __init__(
        self,
        ebcc: util.AbstractEBCC,
        array: Sequence[np.ndarray] = None,
        mo_coeff: Sequence[np.ndarray] = None,
    ):
        self.mf = ebcc.mf
        self.space = ebcc.space
        self.mo_coeff = mo_coeff
        slices = [
            {
                "x": space.correlated,
                "o": space.correlated_occupied,
                "v": space.correlated_virtual,
                "X": space.active,
                "O": space.active_occupied,
                "V": space.active_virtual,
            }
            for space in self.space
        ]

        if self.mo_coeff is None:
            self.mo_coeff = ebcc.mo_coeff

        if array is not None:
            arrays = (array[0], array[1], array[1].transpose((2, 3, 0, 1)), array[2])
        elif isinstance(self.mf._eri, tuple):
            # Have spin-dependent coulomb interaction; precalculate required arrays for simplicity.
            arrays_aabb = ao2mo.incore.general(
                self.mf._eri[1], [self.mo_coeff[i] for i in (0, 0, 1, 1)], compact=False
            )
            arrays = (
                ao2mo.incore.general(
                    self.mf._eri[0], [self.mo_coeff[i] for i in (0, 0, 0, 0)], compact=False
                ),
                arrays_aabb,
                arrays_aabb.transpose(2, 3, 0, 1),
                ao2mo.incore.general(
                    self.mf._eri[2], [self.mo_coeff[i] for i in (1, 1, 1, 1)], compact=False
                ),
            )
        else:
            arrays = (None, None, None, None)

        self.aaaa = RERIs(
            ebcc,
            arrays[0],
            slices=[slices[i] for i in (0, 0, 0, 0)],
            mo_coeff=[self.mo_coeff[i] for i in (0, 0, 0, 0)],
        )
        self.aabb = RERIs(
            ebcc,
            arrays[1],
            slices=[slices[i] for i in (0, 0, 1, 1)],
            mo_coeff=[self.mo_coeff[i] for i in (0, 0, 1, 1)],
        )
        self.bbaa = RERIs(
            ebcc,
            arrays[2],
            slices=[slices[i] for i in (1, 1, 0, 0)],
            mo_coeff=[self.mo_coeff[i] for i in (1, 1, 0, 0)],
        )
        self.bbbb = RERIs(
            ebcc,
            arrays[3],
            slices=[slices[i] for i in (1, 1, 1, 1)],
            mo_coeff=[self.mo_coeff[i] for i in (1, 1, 1, 1)],
        )


class GERIs(RERIs):
    """Electronic repulsion integral container class. Consists of a
    namespace containing blocks of the integrals, with keys that are
    length-4 strings of `"o"` or `"v"` signifying whether the
    corresponding dimension is occupied or virtual.
    """

    def __init__(
        self,
        ebcc: util.AbstractEBCC,
        array: np.ndarray = None,
        slices: Sequence[slice] = None,
        mo_coeff: np.ndarray = None,
    ):
        if array is None:
            RERIs.__init__(self, ebcc, slices=slices, mo_coeff=mo_coeff)

            mo_a = [mo[: self.mf.mol.nao] for mo in self.mo_coeff]
            mo_b = [mo[self.mf.mol.nao :] for mo in self.mo_coeff]

            eri = ao2mo.kernel(self.mf._eri, mo_a)
            eri += ao2mo.kernel(self.mf._eri, mo_b)
            eri += ao2mo.kernel(self.mf._eri, mo_a[:2] + mo_b[2:])
            eri += ao2mo.kernel(self.mf._eri, mo_b[:2] + mo_a[2:])

            eri = ao2mo.addons.restore(1, eri, ebcc.nmo)
            eri = eri.reshape((ebcc.nmo,) * 4)
            eri = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
        else:
            eri = array

        self.eri = eri

    def __getattr__(self, key: str) -> np.ndarray:
        i, j, k, l = (self.slices[i][k] for i, k in enumerate(key))
        return self.eri[i][:, j][:, :, k][:, :, :, l]
