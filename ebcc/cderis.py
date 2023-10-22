"""Cholesky decomposed ERI containers."""

import numpy as np
from pyscf import ao2mo

from ebcc import util


class CDERIs(util.Namespace):
    """Base class for Cholesky decomposed ERIs."""

    pass


class RCDERIs(CDERIs):
    """
    Cholesky decomposed ERI container class for `REBCC`. Consists of a
    just-in-time namespace containing blocks of the integrals.

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
            self.mo_coeff = [self.mo_coeff] * 2

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
            self.slices = [self.slices] * 2

    def __getattr__(self, key):
        """Just-in-time attribute getter."""

        if len(key) == 4:
            e1 = getattr(self, key[:2])
            e2 = getattr(self, key[2:])
            return util.einsum("Qij,Qkl->ijkl", e1, e2)
        elif len(key) == 3:
            key = key[1:]

        if self.array is None:
            if key not in self.__dict__.keys():
                coeffs = []
                for i, k in enumerate(key):
                    coeffs.append(self.mo_coeff[i][:, self.slices[i][k]])
                ijslice = (
                    0,
                    coeffs[0].shape[-1],
                    coeffs[0].shape[-1],
                    coeffs[0].shape[-1] + coeffs[1].shape[-1],
                )
                coeffs = np.concatenate(coeffs, axis=1)
                block = ao2mo._ao2mo.nr_e2(
                    self.mf.with_df._cderi, coeffs, ijslice, aosym="s2", mosym="s1"
                )
                block = block.reshape(-1, ijslice[1] - ijslice[0], ijslice[3] - ijslice[2])
                self.__dict__[key] = block
            return self.__dict__[key]
        else:
            slices = []
            for i, k in enumerate(key):
                slices.append(self.slices[i][k])
            si, sj = slices
            block = self.array[:, si][:, :, sj]
            return block

    __getitem__ = __getattr__


@util.has_docstring
class UCDERIs(CDERIs):
    """
    Cholesky decomposed ERI container class for `UEBCC`. Consists of a
    just-in-time namespace containing blocks of the integrals.

    Attributes
    ----------
    ebcc : UEBCC
        The EBCC object.
    array : iterable of np.ndarray, optional
        The array of integrals in the MO basis for each spin. If
        provided, do not perform just-in-time transformations but
        instead slice the array.  Default value is `None`.
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

        if self.mo_coeff is None:
            self.mo_coeff = ebcc.mo_coeff

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

        if array is not None:
            arrays = array
        else:
            arrays = (None, None)

        self.aa = RCDERIs(
            ebcc,
            arrays[0],
            slices=[self.slices[0], self.slices[0]],
            mo_coeff=[self.mo_coeff[0], self.mo_coeff[0]],
        )
        self.bb = RCDERIs(
            ebcc,
            arrays[1],
            slices=[self.slices[1], self.slices[1]],
            mo_coeff=[self.mo_coeff[1], self.mo_coeff[1]],
        )

    def __getattr__(self, key):
        """Just-in-time attribute getter."""

        if len(key) == 4:
            # Hacks in support for i.e. `UCDERIs.aaaa.ovov`
            v1 = getattr(self, key[:2])
            v2 = getattr(self, key[2:])

            class FakeCDERIs:
                def __getattr__(self, key):
                    e1 = getattr(v1, key[:2])
                    e2 = getattr(v2, key[2:])
                    return util.einsum("Qij,Qkl->ijkl", e1, e2)

                __getitem__ = __getattr__

            return FakeCDERIs()

        elif len(key) == 3:
            key = key[1:]
            return getattr(self, key)

    __getitem__ = __getattr__
