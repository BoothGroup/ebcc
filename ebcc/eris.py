"""Electronic repulsion integral containers."""

from pyscf import ao2mo

from ebcc import numpy as np
from ebcc import util
from ebcc.precision import types


class ERIs(util.Namespace):
    """Base class for electronic repulsion integrals."""

    pass


class RERIs(ERIs):
    """
    Electronic repulsion integral container class for `REBCC`. Consists
    of a just-in-time namespace containing blocks of the integrals.

    The default slices are:
        * `"x"`: correlated
        * `"o"`: correlated occupied
        * `"v"`: correlated virtual
        * `"O"`: active occupied
        * `"V"`: active virtual
        * `"i"`: inactive occupied
        * `"a"`: inactive virtual

    Attributes
    ----------
    ebcc : REBCC
        The EBCC object.
    array : np.ndarray, optional
        The array of integrals in the MO basis. If provided, do not
        perform just-in-time transformations but instead slice the
        array.  Default value is `None`.
    slices : iterable of slice, optional
        The slices to use for each dimension. If provided, the default
        slices outlined above are used.
    mo_coeff : np.ndarray, optional
        The MO coefficients. If not provided, the MO coefficients from
        `ebcc` are used.  Default value is `None`.
    """

    def __init__(self, ebcc, array=None, slices=None, mo_coeff=None):
        util.Namespace.__init__(self)

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
                "O": self.space.active_occupied,
                "V": self.space.active_virtual,
                "i": self.space.inactive_occupied,
                "a": self.space.inactive_virtual,
            }
        if not isinstance(self.slices, (tuple, list)):
            self.slices = [self.slices] * 4

    def __getattr__(self, key):
        """Just-in-time attribute getter."""

        if self.array is None:
            if key not in self.__dict__.keys():
                coeffs = []
                for i, k in enumerate(key):
                    coeffs.append(self.mo_coeff[i][:, self.slices[i][k]].astype(np.float64))
                block = ao2mo.incore.general(
                    self.mf._eri,
                    coeffs,
                    compact=False,
                )
                block = block.reshape([c.shape[-1] for c in coeffs])
                self.__dict__[key] = block.astype(types[float])
            return self.__dict__[key]
        else:
            slices = []
            for i, k in enumerate(key):
                slices.append(self.slices[i][k])
            si, sj, sk, sl = slices
            block = self.array[si][:, sj][:, :, sk][:, :, :, sl]
            return block

    __getitem__ = __getattr__


@util.has_docstring
class UERIs(ERIs):
    """
    Electronic repulsion integral container class for `UEBCC`. Consists
    of a namespace of `REBCC` objects, one for each spin signature.

    Attributes
    ----------
    ebcc : UEBCC
        The EBCC object.
    array : iterable of np.ndarray, optional
        The array of integrals in the MO basis. If provided, do not
        perform just-in-time transformations but instead slice the
        array.  Default value is `None`.
    slices : iterable of iterable of slice, optional
        The slices to use for each spin and each dimension therein.
        If provided, the default slices outlined above are used.
    mo_coeff : iterable of np.ndarray, optional
        The MO coefficients for each spin. If not provided, the MO
        coefficients from `ebcc` are used.  Default value is `None`.
    """

    def __init__(self, ebcc, array=None, slices=None, mo_coeff=None):
        util.Namespace.__init__(self)

        self.mf = ebcc.mf
        self.space = ebcc.space
        self.slices = slices
        self.mo_coeff = mo_coeff

        if self.slices is None:
            self.slices = [
                {
                    "x": space.correlated,
                    "o": space.correlated_occupied,
                    "v": space.correlated_virtual,
                    "O": space.active_occupied,
                    "V": space.active_virtual,
                    "i": space.inactive_occupied,
                    "a": space.inactive_virtual,
                }
                for space in self.space
            ]

        if self.mo_coeff is None:
            self.mo_coeff = ebcc.mo_coeff

        if array is not None:
            arrays = (array[0], array[1], array[1].transpose((2, 3, 0, 1)), array[2])
        elif isinstance(self.mf._eri, tuple):
            # Have spin-dependent coulomb interaction; precalculate
            # required arrays for simplicity.
            arrays_aabb = ao2mo.incore.general(
                self.mf._eri[1],
                [self.mo_coeff[i].astype(np.float64) for i in (0, 0, 1, 1)],
                compact=False,
            )
            arrays = (
                ao2mo.incore.general(
                    self.mf._eri[0],
                    [self.mo_coeff[i].astype(np.float64) for i in (0, 0, 0, 0)],
                    compact=False,
                ),
                arrays_aabb,
                arrays_aabb.transpose(2, 3, 0, 1),
                ao2mo.incore.general(
                    self.mf._eri[2],
                    [self.mo_coeff[i].astype(np.float64) for i in (1, 1, 1, 1)],
                    compact=False,
                ),
            )
            arrays = tuple(array.astype(types[float]) for array in arrays)
        else:
            arrays = (None, None, None, None)

        self.aaaa = RERIs(
            ebcc,
            arrays[0],
            slices=[self.slices[i] for i in (0, 0, 0, 0)],
            mo_coeff=[self.mo_coeff[i] for i in (0, 0, 0, 0)],
        )
        self.aabb = RERIs(
            ebcc,
            arrays[1],
            slices=[self.slices[i] for i in (0, 0, 1, 1)],
            mo_coeff=[self.mo_coeff[i] for i in (0, 0, 1, 1)],
        )
        self.bbaa = RERIs(
            ebcc,
            arrays[2],
            slices=[self.slices[i] for i in (1, 1, 0, 0)],
            mo_coeff=[self.mo_coeff[i] for i in (1, 1, 0, 0)],
        )
        self.bbbb = RERIs(
            ebcc,
            arrays[3],
            slices=[self.slices[i] for i in (1, 1, 1, 1)],
            mo_coeff=[self.mo_coeff[i] for i in (1, 1, 1, 1)],
        )


@util.has_docstring
class GERIs(RERIs):
    __doc__ = __doc__.replace("REBCC", "GEBCC")

    def __init__(self, ebcc, array=None, slices=None, mo_coeff=None):
        util.Namespace.__init__(self)

        if mo_coeff is None:
            mo_coeff = ebcc.mo_coeff
        if not (isinstance(mo_coeff, (tuple, list)) or mo_coeff.ndim == 3):
            mo_coeff = [mo_coeff] * 4

        if array is None:
            mo_a = [mo[: ebcc.mf.mol.nao].astype(np.float64) for mo in mo_coeff]
            mo_b = [mo[ebcc.mf.mol.nao :].astype(np.float64) for mo in mo_coeff]

            array = ao2mo.kernel(ebcc.mf._eri, mo_a)
            array += ao2mo.kernel(ebcc.mf._eri, mo_b)
            array += ao2mo.kernel(ebcc.mf._eri, mo_a[:2] + mo_b[2:])
            array += ao2mo.kernel(ebcc.mf._eri, mo_b[:2] + mo_a[2:])

            array = ao2mo.addons.restore(1, array, ebcc.nmo)
            array = array.astype(types[float])
            array = array.reshape((ebcc.nmo,) * 4)
            array = array.transpose(0, 2, 1, 3) - array.transpose(0, 2, 3, 1)

        RERIs.__init__(self, ebcc, slices=slices, mo_coeff=mo_coeff, array=array)
