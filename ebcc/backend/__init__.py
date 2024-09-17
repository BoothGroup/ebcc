"""Backend for NumPy operations.

Notes:
    Currently, the following backends are supported:
        - NumPy
        - CuPy
        - TensorFlow
        - JAX

    Non-NumPy backends are only lightly supported. Some functionality may not be available, and only
    minimal tests are performed. Some operations that require interaction with NumPy such as the
    PySCF interfaces may not be efficient, due to the need to convert between NumPy and the backend
    array types.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ebcc import BACKEND

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Union, TypeVar

    from numpy import int64, generic
    from numpy.typing import NDArray

    T = TypeVar("T", bound=generic)

if BACKEND == "numpy":
    import numpy as np
elif BACKEND == "cupy":
    import cupy as np  # type: ignore[no-redef]
elif BACKEND == "tensorflow":
    import tensorflow as tf
    import tensorflow.experimental.numpy as np  # type: ignore[no-redef]
elif BACKEND == "jax":
    import jax
    import jax.numpy as np  # type: ignore[no-redef]


def __getattr__(name: str) -> ModuleType:
    """Get the backend module."""
    return importlib.import_module(f"ebcc.backend._{BACKEND.lower()}")


def _put(
    array: NDArray[T],
    indices: Union[NDArray[int64], tuple[NDArray[int64], ...]],
    values: NDArray[T],
) -> NDArray[T]:
    """Put values into an array at specified indices.

    Args:
        array: Array to put values into.
        indices: Indices to put values at.
        values: Values to put into the array.

    Returns:
        Array with values put at specified indices.

    Notes:
        This function does not guarantee a copy of the array.
    """
    if BACKEND == "numpy" or BACKEND == "cupy":
        if isinstance(indices, tuple):
            indices_flat = np.ravel_multi_index(indices, array.shape)
            np.put(array, indices_flat, values)
        else:
            np.put(array, indices, values)
        return array
    elif BACKEND == "jax":
        if isinstance(indices, tuple):
            indices_flat = np.ravel_multi_index(indices, array.shape)
            array = np.put(array, indices_flat, values, inplace=True)  # type: ignore
        else:
            array = np.put(array, indices, values, inplace=True)  # type: ignore
        return array
    elif BACKEND == "tensorflow":
        if isinstance(indices, (tuple, list)):
            indices_grid = tf.meshgrid(*indices, indexing="ij")
            indices = tf.stack([tf.cast(idx, tf.int32).ravel() for idx in indices_grid], axis=1)
        else:
            indices = tf.cast(tf.convert_to_tensor(indices), tf.int32)
            indices = tf.expand_dims(indices, axis=-1)
        values = tf.convert_to_tensor(values, dtype=array.dtype).ravel()
        return tf.tensor_scatter_nd_update(array, indices, values)  # type: ignore
    else:
        raise NotImplementedError(f"Backend {BACKEND} _put not implemented.")
