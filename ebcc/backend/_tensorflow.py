# type: ignore
"""TensorFlow backend."""

import opt_einsum
import tensorflow as tf
import tensorflow.experimental.numpy

tensorflow.experimental.numpy.experimental_enable_numpy_behavior()


def __getattr__(name):
    """Get the attribute from the NumPy drop-in."""
    if name == "linalg":
        return tf.linalg
    return getattr(tensorflow.experimental.numpy, name)


tf.Tensor.item = lambda self: self.numpy().item()
tf.Tensor.copy = lambda self: tf.identity(self)
tf.Tensor.real = property(lambda self: tensorflow.experimental.numpy.real(self))
tf.Tensor.imag = property(lambda self: tensorflow.experimental.numpy.imag(self))
tf.Tensor.conj = lambda self: tensorflow.experimental.numpy.conj(self)


def _argsort(strings, **kwargs):
    if not isinstance(strings, tf.Tensor):
        return tf.convert_to_tensor(
            sorted(range(len(strings)), key=lambda i: strings[i]), dtype=tf.int32
        )
    return _tf_argsort(strings, **kwargs)


_tf_argsort = tf.experimental.numpy.argsort
tf.experimental.numpy.argsort = _argsort


def _block(arrays):
    # FIXME Can't get this to work
    import numpy

    return tensorflow.experimental.numpy.asarray(numpy.block(arrays))


tf.experimental.numpy.block = _block


def _ravel_multi_index(multi_index, dims, mode="raise", order="C"):
    if mode != "raise":
        raise NotImplementedError("Only 'raise' mode is implemented")
    if order != "C":
        raise NotImplementedError("Only 'C' order is implemented")

    # Calculate the strides for each dimension
    strides = tf.math.cumprod([1] + list(dims[:-1]), exclusive=True, reverse=True)

    # Compute the flat index
    flat_index = tf.reduce_sum([idx * stride for idx, stride in zip(multi_index, strides)], axis=0)

    return flat_index


tf.experimental.numpy.ravel_multi_index = _ravel_multi_index


def _indices(dimensions, dtype=tf.int32, sparse=False):
    # Generate a range of indices for each dimension
    ranges = [tf.range(dim, dtype=dtype) for dim in dimensions]

    if sparse:
        # Create sparse representation by reshaping ranges
        grids = [
            tf.reshape(r, [-1 if i == j else 1 for j in range(len(dimensions))])
            for i, r in enumerate(ranges)
        ]
        return grids
    else:
        # Create a dense meshgrid of indices for each dimension
        grids = tf.meshgrid(*ranges, indexing="ij")
        # Stack the grids together to form the final result
        indices = tf.stack(grids, axis=0)
        return indices


tf.experimental.numpy.indices = _indices


def _transpose(tensor, *axes):
    # If axes are provided as separate arguments, convert them to a tuple
    if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        axes = axes[0]
    if len(axes) == 0:
        axes = tuple(reversed(range(tf.rank(tensor))))
    return tf.transpose(tensor, perm=axes)


tf.Tensor.transpose = _transpose


def _swapaxes(tensor, axis1, axis2):
    # Get the current shape of the tensor
    shape = tf.range(tf.rank(tensor))

    # Swap the specified axes
    perm = tf.tensor_scatter_nd_update(shape, [[axis1], [axis2]], [axis2, axis1])

    # Transpose the tensor with the new permutation
    return tf.transpose(tensor, perm)


tf.Tensor.swapaxes = _swapaxes


def einsum_path(*args, **kwargs):  # type: ignore
    """Evaluate the lowest cost contraction order for an einsum expression."""
    kwargs = dict(kwargs)
    if kwargs.get("optimize", True) is True:
        kwargs["optimize"] = "optimal"
    return opt_einsum.contract_path(*args, **kwargs)
