"""TensorFlow backend."""

from __future__ import annotations

import opt_einsum
import tensorflow
import tensorflow.experimental.numpy
tensorflow.experimental.numpy.experimental_enable_numpy_behavior()


def __getattr__(name: str) -> object:
    """Get the attribute from the NumPy drop-in."""
    return getattr(tensorflow.experimental.numpy, name)


def einsum_path(*args, **kwargs):  # type: ignore
    """Evaluate the lowest cost contraction order for an einsum expression."""
    kwargs = dict(kwargs)
    if kwargs.get("optimize", True) is True:
        kwargs["optimize"] = "optimal"
    return opt_einsum.contract_path(*args, **kwargs)
