# type: ignore
"""Cyclops Tensor Framework backend."""

import ctf
import numpy
import opt_einsum


def __getattr__(name):
    """Get the attribute from CTF."""
    return getattr(ctf, name)


class FakeLinalg:
    """Fake linalg module for CTF."""

    def __getattr__(self, name):
        """Get the attribute from CTF's linalg module."""
        return getattr(ctf.linalg, name)

    def eigh(self, a):  # noqa: D102
        # TODO Need to determine if SCALAPACK is available
        w, v = numpy.linalg.eigh(a.to_nparray())
        w = ctf.astensor(w)
        v = ctf.astensor(v)
        return w, v

    def norm(self, a, ord=None):  # noqa: D102
        return ctf.norm(a, ord=ord)


linalg = FakeLinalg()


bool_ = numpy.bool_
inf = numpy.inf
asarray = ctf.astensor


_array = ctf.array


def array(obj, **kwargs):  # noqa: D103
    if isinstance(obj, ctf.tensor):
        return obj
    return _array(numpy.asarray(obj), **kwargs)


def zeros_like(obj):  # noqa: D103
    return ctf.zeros(obj.shape).astype(obj.dtype)


def ones_like(obj):  # noqa: D103
    return ctf.ones(obj.shape).astype(obj.dtype)


_real = ctf.real
_imag = ctf.imag
_conj = ctf.conj


def real(obj):  # noqa: D103
    if isinstance(obj, ctf.tensor):
        return _real(obj)
    return numpy.real(obj)


def imag(obj):  # noqa: D103
    if isinstance(obj, ctf.tensor):
        return _imag(obj)
    return numpy.imag(obj)


def conj(obj):  # noqa: D103
    if isinstance(obj, ctf.tensor):
        return _conj(obj)
    return numpy.conj(obj)


def arange(start, stop=None, step=1, dtype=None):  # noqa: D103
    if stop is None:
        stop = start
        start = 0
    return ctf.arange(start, stop, step=step, dtype=dtype)


def argmin(obj):  # noqa: D103
    return ctf.to_nparray(obj).argmin()


def argmax(obj):  # noqa: D103
    return ctf.to_nparray(obj).argmax()


def bitwise_and(a, b):  # noqa: D103
    return a * b


def bitwise_not(a):  # noqa: D103
    return ones_like(a) - a


def count_nonzero(a):  # noqa: D103
    return ctf.sum((a != 0).astype(int))


def concatenate(arrays, axis=None):  # noqa: D103
    if axis is None:
        axis = 0
    if axis < 0:
        axis += arrays[0].ndim
    shape = list(arrays[0].shape)
    for arr in arrays[1:]:
        for i, (a, b) in enumerate(zip(shape, arr.shape)):
            if i == axis:
                shape[i] += b
            elif a != b:
                raise ValueError("All arrays must have the same shape")

    result = ctf.zeros(shape, dtype=arrays[0].dtype)
    start = 0
    for arr in arrays:
        end = start + arr.shape[axis]
        slices = [slice(None)] * result.ndim
        slices[axis] = slice(start, end)
        result[tuple(slices)] = arr
        start = end

    return result


def stack(arrays, axis=0):  # noqa: D103
    arrays_expanded = []
    for array in arrays:
        shape = list(array.shape)
        shape.insert(axis, 1)
        arrays_expanded.append(array.reshape(shape))
    return concatenate(arrays_expanded, axis=axis)


def _block_recursive(arrays, max_depth, depth=0):  # noqa: D103
    if depth < max_depth:
        arrs = [_block_recursive(arr, max_depth, depth + 1) for arr in arrays]
        return concatenate(arrs, axis=-(max_depth - depth))
    else:
        return arrays


def block(arrays):  # noqa: D103
    def _get_max_depth(arrays):
        if isinstance(arrays, list):
            return 1 + max([_get_max_depth(arr) for arr in arrays])
        return 0

    return _block_recursive(arrays, _get_max_depth(arrays))


def einsum(*args, optimize=True, **kwargs):
    """Evaluate an einsum expression."""
    # FIXME This shouldn't be called, except via `util.einsum`, which should have already
    #       optimised the expression. We should check if this contraction has more than
    #       two tensors and if so, raise an error.
    return ctf.einsum(*args, **kwargs)


def einsum_path(*args, **kwargs):
    """Evaluate the lowest cost contraction order for an einsum expression."""
    kwargs = dict(kwargs)
    if kwargs.get("optimize", True) is True:
        kwargs["optimize"] = "optimal"
    return opt_einsum.contract_path(*args, **kwargs)
