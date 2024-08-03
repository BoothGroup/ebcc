"""Einstein summation convention."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from pyscf.lib import direct_sum, dot  # type: ignore  # noqa: F401
from pyscf.lib import einsum as pyscf_einsum  # type: ignore  # noqa: F401

from ebcc import numpy as np

if TYPE_CHECKING:
    from typing import Any, TypeVar, Union

    from ebcc.numpy.typing import NDArray  # type: ignore

    T = TypeVar("T")

# Try to import TBLIS
try:
    try:
        import tblis_einsum  # type: ignore
    except ImportError:
        from pyscf.tblis_einsum import tblis_einsum  # type: ignore
    FOUND_TBLIS = True
except ImportError:
    FOUND_TBLIS = False

# Define the size of problem to fall back on NumPy
NUMPY_EINSUM_SIZE = 2000


class EinsumOperandError(ValueError):
    """Exception for invalid inputs to `einsum`."""

    pass


def _fallback_einsum(*operands: Any, **kwargs: Any) -> NDArray[T]:
    """Handle the fallback to `numpy.einsum`."""
    # Parse the kwargs
    kwargs = kwargs.copy()
    alpha = kwargs.pop("alpha", 1.0)
    beta = kwargs.pop("beta", 0.0)
    out = kwargs.pop("out", None)
    kwargs["optimize"] = True

    # Perform the contraction
    res = np.einsum(*operands, **kwargs)
    res *= alpha

    # Scale the output
    if out is not None:
        res += beta * out

    return res


def contract(subscript: str, *args: Any, **kwargs: Any) -> NDArray[T]:
    """Contraction a pair of terms in an Einstein summation.

    Supports additional keyword arguments `alpha` and `beta` which operate as `pyscf.lib.dot`.
    In some cases this will still require copying, but it minimises the memory overhead in simple
    cases.

    Args:
        subscript: Subscript representation of the contraction.
        args: Arrays to contract.
        alpha: Scaling factor for contraction.
        beta: Scaling factor for the output.
        out: Output array. If `None`, a new array is created.
        kwargs: Additional keyword arguments to `numpy.einsum`.

    Returns:
        Result of the contraction.
    """
    alpha = kwargs.get("alpha", 1.0)
    beta = kwargs.get("beta", 0.0)
    buf = kwargs.get("out", None)

    # If this is called for more than 2 arguments, fall back
    if len(args) > 2:
        return _fallback_einsum(subscript, *args, **kwargs)

    # Make sure that the input are numpy arrays
    a, b = args
    a = np.asarray(a)
    b = np.asarray(b)

    # Check if we should use NumPy
    if min(a.size, b.size) < NUMPY_EINSUM_SIZE:
        return _fallback_einsum(subscript, *args, **kwargs)

    # Make sure it can be done via DGEMM
    indices = subscript.replace(",", "").replace("->", "")
    if any(indices.count(x) != 2 for x in set(indices)):
        return _fallback_einsum(subscript, *args, **kwargs)

    # Get the characters for each input and output
    inp, out, args = np.core.einsumfunc._parse_einsum_input((subscript, a, b))  # type: ignore
    inp_a, inp_b = inps = inp.split(",")
    assert len(inps) == len(args) == 2
    assert all(len(inp) == arg.ndim for inp, arg in zip(inps, args))

    # If there is an internal trace, consume it:
    if any(len(inp) != len(set(inp)) for inp in inps):
        # FIXME
        return _fallback_einsum(subscript, *args, **kwargs)

    # Find the dummy indices
    dummy = set(inp_a).intersection(set(inp_b))
    if not dummy or inp_a == dummy or inp_b == dummy:
        return _fallback_einsum(subscript, *args, **kwargs)

    # Find the sizes of the indices
    ranges: dict[int, int] = {}
    for inp, arg in zip(inps, args):
        for i, s in zip(inp, arg.shape):
            if i in ranges:
                if ranges[i] != s:
                    raise EinsumOperandError(
                        "Incompatible shapes for einsum: {} with A={}, B={}".format(
                            subscript, a.shape, b.shape
                        )
                    )
            ranges[i] = s

    if not FOUND_TBLIS:
        # Reorder the indices appropriately
        inp_at = list(inp_a)
        inp_bt = list(inp_b)
        inner_shape = 1
        for i, n in enumerate(sorted(dummy)):
            j = len(inp_at) - 1
            inp_at.insert(j, inp_at.pop(inp_at.index(n)))
            inp_bt.insert(i, inp_bt.pop(inp_bt.index(n)))
            inner_shape *= ranges[n]

        # Find transposes
        order_a = [inp_a.index(idx) for idx in inp_at]
        order_b = [inp_b.index(idx) for idx in inp_bt]

        # Get shape and transpose for the output
        shape_ct = []
        inp_ct = []
        for idx in inp_at:
            if idx in dummy:
                break
            shape_ct.append(ranges[idx])
            inp_ct.append(idx)
        for idx in inp_bt:
            if idx in dummy:
                continue
            shape_ct.append(ranges[idx])
            inp_ct.append(idx)
        order_ct = [inp_ct.index(idx) for idx in out]

        # If any dimension has size zero, return here
        if a.size == 0 or b.size == 0:
            shape_c = [shape_ct[i] for i in order_ct]
            if buf is not None:
                return buf.reshape(shape_c) * beta if beta != 1.0 else buf.reshape(shape_c)
            else:
                return np.zeros(shape_c, dtype=np.result_type(a, b))

        # Apply transposes
        at = a.transpose(order_a)
        bt = b.transpose(order_b)

        # Find the optimal memory alignment
        at = np.asarray(at.reshape(-1, inner_shape), order="F" if at.flags.f_contiguous else "C")
        bt = np.asarray(bt.reshape(inner_shape, -1), order="F" if bt.flags.f_contiguous else "C")

        # Get the output buffer
        if buf is not None:
            shape_ct_flat = (at.shape[0], bt.shape[1])
            order_c = [out.index(idx) for idx in inp_ct]
            buf = buf.transpose(order_c)
            buf = np.asarray(
                buf.reshape(shape_ct_flat), order="F" if buf.flags.f_contiguous else "C"
            )

        # Perform the contraction
        ct = dot(at, bt, alpha=alpha, beta=beta, c=buf)

        # Reshape and transpose
        ct = ct.reshape(shape_ct, order="A")
        c = ct.transpose(order_ct)

    else:
        # Cast the data types
        dtype = np.result_type(a, b, alpha, beta)
        alpha = np.asarray(alpha, dtype=dtype)
        beta = np.asarray(beta, dtype=dtype)
        a = np.asarray(a, dtype=dtype)
        b = np.asarray(b, dtype=dtype)
        tblis_dtype = tblis_einsum.tblis_dtype[dtype]

        # Get the shapes
        shape_a = a.shape
        shape_b = b.shape
        shape_c = [ranges[x] for x in out]

        # If any dimension has size zero, return here
        if a.size == 0 or b.size == 0:
            if buf is not None:
                return buf.reshape(shape_c) * beta
            else:
                return np.zeros(shape_c, dtype=dtype)

        # Get the output buffer
        if buf is None:
            order = kwargs.get("order", "C")
            c = np.empty(shape_c, dtype=dtype, order=order)
        else:
            assert buf.dtype == dtype
            assert buf.size == np.prod(shape_c)
            c = buf.reshape(shape_c)

        # Get the C types
        shape_a = (ctypes.c_size_t * a.ndim)(*shape_a)
        shape_b = (ctypes.c_size_t * b.ndim)(*shape_b)
        shape_c = (ctypes.c_size_t * c.ndim)(*shape_c)
        strides_a = (ctypes.c_size_t * a.ndim)(*[x // dtype.itemsize for x in a.strides])
        strides_b = (ctypes.c_size_t * b.ndim)(*[x // dtype.itemsize for x in b.strides])
        strides_c = (ctypes.c_size_t * c.ndim)(*[x // dtype.itemsize for x in c.strides])

        # Perform the contraction
        tblis_einsum.libtblis.as_einsum(
            a,
            a.ndim,
            shape_a,
            strides_a,
            inp_a.encode("ascii"),
            b,
            b.ndim,
            shape_b,
            strides_b,
            inp_b.encode("ascii"),
            c,
            c.ndim,
            shape_c,
            strides_c,
            out.encode("ascii"),
            tblis_dtype,
            alpha,
            beta,
        )

    return c


def einsum(*operands: Any, **kwargs: Any) -> Union[T, NDArray[T]]:
    """Evaluate an Einstein summation convention on the operands.

    Using the Einstein summation convention, many common
    multi-dimensional, linear algebraic array operations can be
    represented in a simple fashion. In *implicit* mode `einsum`
    computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical
    Einstein summation operations, by disabling, or forcing summation
    over specified subscript labels.

    See the `numpy.einsum` documentation for clarification.

    Args:
        operands: Any valid input to `numpy.einsum`.
        alpha: Scaling factor for the contraction.
        beta: Scaling factor for the output.
        out: If provided, the calculation is done into this array.
        contract: The function to use for contraction.
        optimize: If `True`, use the `numpy.einsum_path` to optimize the contraction.

    Returns:
        The calculation based on the Einstein summation convention.

    Notes:
        This function may use `numpy.einsum`, `pyscf.lib.einsum`, or `tblis_einsum` as a backend,
        depending on the problem size and the modules available.
    """
    # Parse the kwargs
    inp, out, args = np.core.einsumfunc._parse_einsum_input(operands)  # type: ignore
    subscript = "%s->%s" % (inp, out)
    _contract = kwargs.get("contract", contract)

    # Perform the contraction
    if len(args) < 2:
        # If it's just a transpose, use the fallback
        out = _fallback_einsum(subscript, *args, **kwargs)
    elif len(args) < 3:
        # If it's a single contraction, call the backend directly
        out = _contract(subscript, *args, **kwargs)
    else:
        # If it's a chain of contractions, use the path optimizer
        optimize = kwargs.pop("optimize", True)
        args = list(args)
        path_kwargs = dict(optimize=optimize, einsum_call=True)
        contractions = np.einsum_path(subscript, *args, **path_kwargs)[1]  # type: ignore
        for contraction in contractions:
            inds, idx_rm, einsum_str, remain = list(contraction[:4])
            contraction_args = [args.pop(x) for x in inds]
            if kwargs.get("alpha", 1.0) != 1.0 or kwargs.get("beta", 0.0) != 0.0:
                raise NotImplementedError("Scaling factors not supported for >2 arguments")
            out = _contract(einsum_str, *contraction_args, **kwargs)
            args.append(out)

    return out
