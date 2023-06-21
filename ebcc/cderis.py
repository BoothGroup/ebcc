"""Density fitted electronic repulsion integral containers.
"""

import types
from typing import Sequence

import numpy as np
from pyscf.ao2mo import _ao2mo

from ebcc import util


class RCDERIs(util.Namespace):
    """
    Density fitted electronic repulsion integral container class for
    `REBCC`. Consists of a just-in-time namespace containing blocks of
    the integrals.

    The default slices are:
        * `"x"`: correlated
        * `"o"`: correlated occupied
        * `"v"`: correlated virtual
        * `"O"`: active occupied
        * `"V"`: active virtual
        * `"i"`: inactive occupied
        * `"a"`: inactive virtual

    Parameters
    ----------
    ebcc : AbstractEBCC
        The EBCC object.
    array : np.ndarray, optional
        The array of integrals in the MO basis. If provided, do not
        perform just-in-time transformations but instead slice the
        array.  Default value is `None`.
    slices : Sequence[slice], optional
        The slices to use for each dimension. If provided, the default
        slices outlined above are used.
    mo_coeff : np.ndarray, optional
        The MO coefficients. If not provided, the MO coefficients from
        `ebcc` are used.  Default value is `None`.
    """

    def __init__(
        self,
        ebcc: util.AbstractEBCC,
        array: np.ndarray = None,
        slices: Sequence[slice] = None,
        mo_coeff: np.ndarray = None,
    ):
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

    def __getattr__(self, key: str) -> np.ndarray:
        """Just-in-time attribute getter."""

        if self.array is None:
            if key not in self.__dict__.keys():
                coeffs = []
                for i, k in enumerate(key):
                    coeffs.append(self.mo_coeff[i][:, self.slices[i][k]])
                mo = np.stack(coeffs, axis=-1)
                ijslice = (0, coeffs[0].shape[-1], coeffs[0].shape[-1], mo.shape[-1])
                block = _ao2mo.nr_e2(self.mf.with_df._cderi, mo, ijslice, aosym="s2", mosym="s1")
                block = block.reshape([-1] + [c.shape[-1] for c in coeffs])
                self.__dict__[key] = block
            return self.__dict__[key]
        else:
            slices = []
            for i, k in enumerate(key):
                slices.append(self.slices[i][k])
            si, sj = slices
            block = self.array[:, si, sj]
            return block


class UCDERIs(util.Namespace):
    """
    Density fitted electronic repulsion integral container class for
    `UEBCC`. Consists of a just-in-time namespace containing blocks of
    the integrals.

    Parameters
    ----------
    ebcc : AbstractEBCC
        The EBCC object.
    array : Sequence[np.ndarray], optional
        The array of integrals in the MO basis. If provided, do not
        perform just-in-time transformations but instead slice the
        array.  Default value is `None`.
    slices : Sequence[Sequence[slice]], optional
        The slices to use for each spin and each dimension therein.
        If provided, the default slices outlined above are used.
    mo_coeff : Sequence[np.ndarray], optional
        The MO coefficients for each spin. If not provided, the MO
        coefficients from `ebcc` are used.  Default value is `None`.
    """

    def __init__(
        self,
        ebcc: util.AbstractEBCC,
        array: Sequence[np.ndarray] = None,
        mo_coeff: Sequence[np.ndarray] = None,
        slices: Sequence[Sequence[slice]] = None,
    ):
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
            arrays = (array[0], array[1])
        elif isinstance(self.mf.with_df._cderi, tuple):
            # Have spin-dependent coulomb interaction; precalculate required arrays for simplicity.
            raise NotImplementedError
        else:
            arrays = (None, None)

        self.aa = RERIs(
            ebcc,
            arrays[0],
            slices=[self.slices[i] for i in (0, 0)],
            mo_coeff=[self.mo_coeff[i] for i in (0, 0)],
        )
        self.bb = RERIs(
            ebcc,
            arrays[1],
            slices=[self.slices[i] for i in (1, 1)],
            mo_coeff=[self.mo_coeff[i] for i in (1, 1)],
        )
