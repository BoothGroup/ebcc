"""Einstein summation convention."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

from pyscf.lib import direct_sum, dot  # noqa: F401
from pyscf.lib import einsum as pyscf_einsum  # noqa: F401

from ebcc import numpy as np

if TYPE_CHECKING:
    from typing import Any, Callable, Optional, Sequence, TypeVar, Union

    from numpy import float64
    from numpy.typing import NDArray

    T = float64

    OperandType = Union[str, tuple[int, ...], NDArray[T]]

# Try to import TBLIS
try:
    try:
        import tblis_einsum  # type: ignore
    except ImportError:
        from pyscf.tblis_einsum import tblis_einsum  # type: ignore
    FOUND_TBLIS = True
except ImportError:
    FOUND_TBLIS = not TYPE_CHECKING

"""The size of the problem to fall back on NumPy."""
NUMPY_EINSUM_SIZE = 2000

"""Symbols used in einsum-like functions."""
EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
EINSUM_SYMBOLS_SET = set(EINSUM_SYMBOLS)


class EinsumOperandError(ValueError):
    """Exception for invalid inputs to `einsum`."""

    pass


def _parse_einsum_input(operands: list[OperandType]) -> tuple[str, str, list[NDArray[T]]]:
    """Parse the input for an einsum contraction.

    Args:
        operands: The input operands.

    Returns:
        The parsed input.
    """
    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = operands[1:]

        # Ensure all characters are valid
        for s in subscripts:
            if s in ".,->":
                continue
            if s not in EINSUM_SYMBOLS:
                raise ValueError(f"Character {s} is not a valid symbol.")

    else:
        if not all(isinstance(op, np.ndarray) for op in operands[::2]):
            raise EinsumOperandError("Invalid operands for einsum")
        if not all(isinstance(op, tuple) for op in operands[1::2]):
            raise EinsumOperandError("Invalid subscripts for einsum")
        operand_list: list[NDArray[T]] = operands[:-1:2]  # type: ignore
        subscript_list: list[tuple[int, ...]] = operands[1::2]  # type: ignore
        output_list: Optional[tuple[int, ...]] = operands[-1] if len(operands) % 2 else None  # type: ignore

        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for i in sub:
                subscripts += EINSUM_SYMBOLS[i]
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for i in output_list:
                subscripts += EINSUM_SYMBOLS[i]

    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(EINSUM_SYMBOLS_SET - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(",")
            out_sub = False

        for num, ssub in enumerate(split_subscripts):
            if "." in ssub:
                if (ssub.count(".") != 3) or (ssub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operand_list[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operand_list[num].ndim, 1)
                    ellipse_count -= len(ssub) - 3

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = ssub.replace("...", "")
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = ssub.replace("...", rep_inds)

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (EINSUM_SYMBOLS):
                    raise ValueError(f"Character {s} is not a valid symbol.")
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = "".join(sorted(set(output_subscript) - set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in EINSUM_SYMBOLS:
                raise ValueError(f"Character {s} is not a valid symbol.")
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if output_subscript.count(char) != 1:
            raise ValueError(f"Output character {s} appeared more than once in the output.")
        if char not in input_subscripts:
            raise ValueError(f"Output character {s} did not appear in the input")

    return (input_subscripts, output_subscript, operand_list)


def _fallback_einsum(
    subscript: str,
    *args: NDArray[T],
    alpha: T = 1.0,  # type: ignore[assignment]
    beta: T = 0.0,  # type: ignore[assignment]
    out: Optional[NDArray[T]] = None,
    optimize: bool = True,
    **kwargs: Any,
) -> NDArray[T]:
    """Handle the fallback to `numpy.einsum`."""
    # Perform the contraction
    res: NDArray[T] = np.einsum(subscript, *args, optimize=optimize, **kwargs)
    res *= alpha

    # Scale the output
    if out is not None:
        res += beta * out

    return res


def contract(
    subscript: str,
    *args: NDArray[T],
    alpha: T = 1.0,  # type: ignore[assignment]
    beta: T = 0.0,  # type: ignore[assignment]
    out: Optional[NDArray[T]] = None,
    **kwargs: Any,
) -> NDArray[T]:
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
    inp, outs, operands = _parse_einsum_input([subscript, a, b])
    inp_a, inp_b = inps = inp.split(",")
    assert len(inps) == len(operands) == 2
    assert all(len(inp) == arg.ndim for inp, arg in zip(inps, operands))

    # If there is an internal trace, consume it:
    if any(len(inp) != len(set(inp)) for inp in inps):
        # FIXME
        return _fallback_einsum(subscript, *args, **kwargs)

    # Find the dummy indices
    dummy = set(inp_a).intersection(set(inp_b))
    if not dummy or set(inp_a) == dummy or set(inp_b) == dummy:
        return _fallback_einsum(subscript, *args, **kwargs)

    # Find the sizes of the indices
    ranges: dict[str, int] = {}
    for inp, arg in zip(inps, operands):
        for ind, s in zip(inp, arg.shape):
            if ind in ranges:
                if ranges[ind] != s:
                    raise EinsumOperandError(
                        "Incompatible shapes for einsum: {} with A={}, B={}".format(
                            subscript, a.shape, b.shape
                        )
                    )
            ranges[ind] = s

    if not FOUND_TBLIS:
        # Reorder the indices appropriately
        inp_at = list(inp_a)
        inp_bt = list(inp_b)
        inner_shape = 1
        for i, ind in enumerate(sorted(dummy)):
            j = len(inp_at) - 1
            inp_at.insert(j, inp_at.pop(inp_at.index(ind)))
            inp_bt.insert(i, inp_bt.pop(inp_bt.index(ind)))
            inner_shape *= ranges[ind]

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
        order_ct = [inp_ct.index(idx) for idx in outs]

        # If any dimension has size zero, return here
        if a.size == 0 or b.size == 0:
            shape_c = [shape_ct[i] for i in order_ct]
            if out is not None:
                return out.reshape(shape_c) * beta if beta != 1.0 else out.reshape(shape_c)
            else:
                return np.zeros(shape_c, dtype=np.result_type(a, b))

        # Apply transposes
        at = a.transpose(order_a)
        bt = b.transpose(order_b)

        # Find the optimal memory alignment
        at = np.asarray(at.reshape(-1, inner_shape), order="F" if at.flags.f_contiguous else "C")
        bt = np.asarray(bt.reshape(inner_shape, -1), order="F" if bt.flags.f_contiguous else "C")

        # Get the output buffer
        if out is not None:
            shape_ct_flat = (at.shape[0], bt.shape[1])
            order_c = [outs.index(idx) for idx in inp_ct]
            out = out.transpose(order_c)
            out = np.asarray(
                out.reshape(shape_ct_flat), order="F" if out.flags.f_contiguous else "C"
            )

        # Perform the contraction
        ct: NDArray[T] = dot(at, bt, alpha=alpha, beta=beta, c=out)

        # Reshape and transpose
        ct = ct.reshape(shape_ct, order="A")
        c = ct.transpose(order_ct)

    else:
        # Cast the data types
        dtype = np.result_type(a, b, alpha, beta)
        a = np.asarray(a, dtype=dtype)
        b = np.asarray(b, dtype=dtype)
        tblis_dtype = tblis_einsum.tblis_dtype[dtype]

        # Get the shapes
        shape_a = a.shape
        shape_b = b.shape
        shape_c = [ranges[x] for x in outs]

        # If any dimension has size zero, return here
        if a.size == 0 or b.size == 0:
            if out is not None:
                return out.reshape(shape_c) * beta
            else:
                return np.zeros(shape_c, dtype=dtype)

        # Get the output buffer
        if out is None:
            order = kwargs.get("order", "C")
            c = np.empty(shape_c, dtype=dtype, order=order)
        else:
            assert out.dtype == dtype
            assert out.size == np.prod(shape_c)
            c = out.reshape(shape_c)

        # Perform the contraction
        tblis_einsum.libtblis.as_einsum(
            a,
            a.ndim,
            (ctypes.c_size_t * a.ndim)(*shape_a),
            (ctypes.c_size_t * a.ndim)(*[x // dtype.itemsize for x in a.strides]),
            inp_a.encode("ascii"),
            b,
            b.ndim,
            (ctypes.c_size_t * b.ndim)(*shape_b),
            (ctypes.c_size_t * b.ndim)(*[x // dtype.itemsize for x in b.strides]),
            inp_b.encode("ascii"),
            c,
            c.ndim,
            (ctypes.c_size_t * c.ndim)(*shape_c),
            (ctypes.c_size_t * c.ndim)(*[x // dtype.itemsize for x in c.strides]),
            outs.encode("ascii"),
            tblis_dtype,
            np.asarray(alpha, dtype=dtype),
            np.asarray(beta, dtype=dtype),
        )

    return c


def einsum(
    *operands: OperandType,
    alpha: T = 1.0,  # type: ignore[assignment]
    beta: T = 0.0,  # type: ignore[assignment]
    out: Optional[NDArray[T]] = None,
    contract: Callable[..., NDArray[T]] = contract,
    optimize: bool = True,
    **kwargs: Any,
) -> NDArray[T]:
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
    all_kwargs = dict(alpha=alpha, beta=beta, out=out, optimize=optimize, **kwargs)

    # Perform the contraction
    if not len(args):
        raise ValueError("No input operands")
    elif len(args) < 2:
        # If it's just a transpose, use the fallback
        res = _fallback_einsum(subscript, *args, **all_kwargs)
    elif len(args) < 3:
        # If it's a single contraction, call the backend directly
        res = _contract(subscript, *args, **all_kwargs)
    else:
        # If it's a chain of contractions, use the path optimizer
        args = list(args)
        path_kwargs = dict(optimize=optimize, einsum_call=True)
        contractions = np.einsum_path(subscript, *args, **path_kwargs)[1]
        for contraction in contractions:
            inds, idx_rm, einsum_str, remain = list(contraction[:4])
            contraction_args = [args.pop(x) for x in inds]
            if alpha != 1.0 or beta != 0.0:
                raise NotImplementedError("Scaling factors not supported for >2 arguments")
            res = _contract(einsum_str, *contraction_args, **all_kwargs)
            args.append(res)

    return res


def dirsum(*operands: Union[str, tuple[int, ...], NDArray[T]]) -> NDArray[T]:
    """Direct sum of arrays.

    Follows the `numpy.einsum` input conventions.

    Args:
        operands: Any valid input to `numpy.einsum`.

    Returns:
        The direct sum of the arrays.
    """
    # Parse the input
    input_str, output_str, input_arrays = _parse_einsum_input(list(operands))
    input_chars = input_str.split(",")

    for i, (chars, array) in enumerate(zip(input_chars, input_arrays)):
        if len(chars) != array.ndim:
            raise ValueError(f"Dimension mismatch for array {i}.")
        if len(set(chars)) != len(chars):
            unique_chars = "".join(set(chars))
            array = einsum(f"{chars}->{unique_chars}", array)
            input_chars[i] = unique_chars
        if i == 0:
            res = array
        else:
            shape = res.shape + (1,) * array.ndim
            res = res.reshape(shape) + array

    # Reshape the output
    res = einsum(f"{''.join(input_chars)}->{output_str}", res)
    res.flags.writeable = True

    return res
