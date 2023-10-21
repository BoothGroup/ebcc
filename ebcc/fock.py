"""Fock matrix containers."""

from ebcc import numpy as np
from ebcc import util
from ebcc.precision import types


class Fock(util.Namespace):
    """Base class for Fock matrices."""

    pass


class RFock(Fock):
    """
    Fock matrix container class for `REBCC`.

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
    ebcc : REBCC
        The EBCC object.
    array : np.ndarray, optional
        The array of the Fock matrix in the MO basis. If provided, do
        not perform just-in-time transformations but instead slice the
        array.  Default value is `None`.
    slices : iterable of slice, optional
        The slices to use for each dimension. If provided, the default
        slices outlined above are used.
    mo_coeff : np.ndarray, optional
        The MO coefficients. If not provided, the MO coefficients from
        `ebcc` are used.  Default value is `None`.
    g : Namespace, optional
        Namespace containing blocks of the electron-boson coupling
        matrix.  Default value is `None`.
    """

    def __init__(self, ebcc, array=None, slices=None, mo_coeff=None, g=None):
        util.Namespace.__init__(self)

        self.mf = ebcc.mf
        self.space = ebcc.space
        self.slices = slices
        self.mo_coeff = mo_coeff
        self.array = array

        self.shift = ebcc.options.shift
        self.xi = ebcc.xi
        self.g = g
        if self.g is None:
            self.g = ebcc.g

        if self.mo_coeff is None:
            self.mo_coeff = ebcc.mo_coeff
        if not (isinstance(self.mo_coeff, (tuple, list)) or self.mo_coeff.ndim == 3):
            self.mo_coeff = [self.mo_coeff] * 2

        if self.array is None:
            fock_ao = self.mf.get_fock().astype(types[float])
            self.array = util.einsum("pq,pi,qj->ij", fock_ao, *self.mo_coeff)

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

        if key not in self.__dict__.keys():
            ki, kj = key
            i = self.slices[0][ki]
            j = self.slices[1][kj]
            self.__dict__[key] = self.array[i][:, j].copy()

            if self.shift:
                xi = self.xi
                g = self.g.__getattr__(f"b{ki}{kj}")
                g += self.g.__getattr__(f"b{kj}{ki}").transpose(0, 2, 1)
                self.__dict__[key] -= util.einsum("I,Ipq->pq", xi, g)

        return self.__dict__[key]

    __getitem__ = __getattr__


class UFock(Fock, metaclass=util.InheritDocstrings):
    """
    Fock matrix container class for `UEBCC`. Consists of a namespace of
    `RFock` objects, on for each spin signature.

    Parameters
    ----------
    ebcc : UEBCC
        The EBCC object.
    array : iterable of np.ndarray, optional
        The array of the Fock matrix in the MO basis. If provided, do
        not perform just-in-time transformations but instead slice the
        array.  Default value is `None`.
    slices : iterable of iterable of slice, optional
        The slices to use for each dimension. If provided, the default
        slices outlined above are used.
    mo_coeff : iterable of np.ndarray, optional
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

        self.shift = ebcc.options.shift
        self.xi = ebcc.xi
        self.g = ebcc.g

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

        if self.array is None:
            fock_ao = np.asarray(self.mf.get_fock()).astype(types[float])
            self.array = (
                util.einsum("pq,pi,qj->ij", fock_ao[0], self.mo_coeff[0], self.mo_coeff[0]),
                util.einsum("pq,pi,qj->ij", fock_ao[1], self.mo_coeff[1], self.mo_coeff[1]),
            )

        self.aa = RFock(
            ebcc,
            array=self.array[0],
            slices=[self.slices[0], self.slices[0]],
            mo_coeff=[self.mo_coeff[0], self.mo_coeff[0]],
            g=self.g.aa if self.g is not None else None,
        )
        self.bb = RFock(
            ebcc,
            array=self.array[1],
            slices=[self.slices[1], self.slices[1]],
            mo_coeff=[self.mo_coeff[1], self.mo_coeff[1]],
            g=self.g.bb if self.g is not None else None,
        )


@util.has_docstring
class GFock(RFock):
    __doc__ = RFock.__doc__.replace("REBCC", "GEBCC")
