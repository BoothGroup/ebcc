"""Floating point precision control."""

from contextlib import contextmanager

from ebcc import numpy as np

types = {
    float: np.float64,
    complex: np.complex128,
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
    with precision(float=np.float32, complex=np.complex64):
        yield
