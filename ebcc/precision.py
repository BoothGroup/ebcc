"""Floating point precision control."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from ebcc import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal, Type, TypeVar, Union

    T = TypeVar("T", float, complex)


types: dict[type, type]
if TYPE_CHECKING:
    types = {
        float: float,
        complex: complex,
    }
else:
    types = {
        float: np.float64,
        complex: np.complex128,
    }


def cast(value: Any, dtype: Type[T]) -> T:
    """Cast a value to the current floating point type.

    Args:
        value: The value to cast.
        dtype: The type to cast to.

    Returns:
        The value cast to the current floating point type.
    """
    return types[dtype](value)


def set_precision(**kwargs: type) -> None:
    """Set the floating point type.

    Args:
        float: The floating point type to use.
        complex: The complex type to use.
    """
    types[float] = kwargs.get("float", types[float])
    types[complex] = kwargs.get("complex", types[complex])


@contextmanager
def precision(**kwargs: type) -> Iterator[None]:
    """Context manager for setting the floating point precision.

    Args:
        float: The floating point type to use.
        complex: The complex type to use.
    """
    old = {
        "float": types[float],
        "complex": types[complex],
    }
    set_precision(**kwargs)
    yield
    set_precision(**old)


@contextmanager
def single_precision() -> Iterator[None]:
    """Context manager for setting the floating point precision to single precision."""
    with precision(float=np.float32, complex=np.complex64):
        yield
