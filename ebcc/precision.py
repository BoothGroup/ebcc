"""Floating point precision control."""

from contextlib import contextmanager

import numpy  # Use standard NumPy dtypes for any backend

types = {
    bool: bool,
    float: numpy.float64,
    complex: numpy.complex128,
}


def set_precision(**kwargs):
    """Set the floating point type.

    Parameters
    ----------
    float : type, optional
        The floating point type to use.
    complex : type, optional
        The complex type to use.
    """
    types[float] = kwargs.get("float", types[float])
    types[complex] = kwargs.get("complex", types[complex])


@contextmanager
def precision(**kwargs):
    """Context manager for setting the floating point precision.

    Parameters
    ----------
    float : type, optional
        The floating point type to use.
    complex : type, optional
        The complex type to use.
    """
    old = {
        "float": types[float],
        "complex": types[complex],
    }
    set_precision(**kwargs)
    yield
    set_precision(**old)


@contextmanager
def single_precision():
    """
    Context manager for setting the floating point precision to single
    precision.
    """
    with precision(float=numpy.float32, complex=numpy.complex64):
        yield
