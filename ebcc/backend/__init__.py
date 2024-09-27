"""Backend for NumPy operations.

Notes:
    Currently, the following backends are supported:
        - NumPy
        - CuPy
        - TensorFlow
        - JAX
        - CTF (Cyclops Tensor Framework)

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
    from typing import Union, TypeVar, Optional

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
elif BACKEND in ("ctf", "cyclops"):
    import ctf


def __getattr__(name: str) -> ModuleType:
    """Get the backend module."""
    return importlib.import_module(f"ebcc.backend._{BACKEND.lower()}")


def ensure_scalar(obj: Union[T, NDArray[T]]) -> T:
    """Ensure that an object is a scalar.

    Args:
        obj: Object to ensure is a scalar.

    Returns:
        Scalar object.
    """
    if BACKEND in ("numpy", "cupy", "jax"):
        return np.asarray(obj).item()  # type: ignore
    elif BACKEND == "tensorflow":
        if isinstance(obj, tf.Tensor):
            return obj.numpy().item()  # type: ignore
        return obj  # type: ignore
    elif BACKEND in ("ctf", "cyclops"):
        if isinstance(obj, ctf.tensor):
            return obj.to_nparray().item()  # type: ignore
        return obj  # type: ignore
    else:
        raise NotImplementedError(f"`ensure_scalar` not implemented for backend {BACKEND}.")


def to_numpy(array: NDArray[T], dtype: Optional[type[generic]] = None) -> NDArray[T]:
    """Convert an array to NumPy.

    Args:
        array: Array to convert.
        dtype: Data type to convert to.

    Returns:
        Array in NumPy format.

    Notes:
        This function does not guarantee a copy of the array.
    """
    if BACKEND == "numpy":
        ndarray = array
    elif BACKEND == "cupy":
        ndarray = np.asnumpy(array)  # type: ignore
    elif BACKEND == "jax":
        ndarray = np.array(array)  # type: ignore
    elif BACKEND == "tensorflow":
        ndarray = array.numpy()  # type: ignore
    elif BACKEND in ("ctf", "cyclops"):
        ndarray = array.to_nparray()  # type: ignore
    else:
        raise NotImplementedError(f"`to_numpy` not implemented for backend {BACKEND}.")
    if dtype is not None and ndarray.dtype != dtype:
        ndarray = ndarray.astype(dtype)
    return ndarray


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
            array = np.put(array, indices_flat, values, inplace=False)  # type: ignore
        else:
            array = np.put(array, indices, values, inplace=False)  # type: ignore
        return array
    elif BACKEND == "tensorflow":
        if isinstance(indices, (tuple, list)):
            indices_grid = tf.meshgrid(*indices, indexing="ij")
            indices = tf.stack([np.ravel(tf.cast(idx, tf.int32)) for idx in indices_grid], axis=1)
        else:
            indices = tf.cast(tf.convert_to_tensor(indices), tf.int32)
            indices = tf.expand_dims(indices, axis=-1)
        values = np.ravel(tf.convert_to_tensor(values, dtype=array.dtype))
        return tf.tensor_scatter_nd_update(array, indices, values)  # type: ignore
    elif BACKEND in ("ctf", "cyclops"):
        # TODO MPI has to be manually managed here
        if isinstance(indices, tuple):
            indices_flat = np.ravel_multi_index(indices, array.shape)
            array.write(indices_flat, values)  # type: ignore
        else:
            array.write(indices, values)  # type: ignore
        return array
    else:
        raise NotImplementedError(f"`_put` not implemented for backend {BACKEND}.")


def _inflate(
    shape: tuple[int, ...],
    indices: Union[NDArray[int64], tuple[NDArray[int64], ...]],
    values: NDArray[T],
) -> NDArray[T]:
    """Inflate values into an array at specified indices.

    Args:
        shape: Shape of the array.
        indices: Indices to inflate values at.
        values: Values to inflate into the array.

    Returns:
        Array with values inflated at specified indices.
    """
    array = np.zeros(shape, dtype=values.dtype)
    return _put(array, indices, values)
