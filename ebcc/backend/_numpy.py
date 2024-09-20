"""NumPy backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from numpy import generic
    from numpy.typing import DType, NDArray


def __getattr__(name: str) -> object:
    """Get the attribute from NumPy."""
    return getattr(numpy, name)


def astype(obj: NDArray[generic], dtype: DType) -> NDArray[generic]:
    """Cast the array to the specified type.

    Args:
        obj: The array to cast.
        dtype: The type to cast the array to.

    Returns:
        The array cast to the specified type.

    Note:
        This function is part of the array API in NumPy 2.1.0, and this function is for backward
        compatibility.
    """
    return obj.astype(dtype)
