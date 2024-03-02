"""Einstein summation convention."""

import ctypes
import operator

from pyscf.lib import direct_sum, dot  # noqa: F401
from pyscf.lib import einsum as pyscf_einsum  # noqa: F401

from ebcc import TENSOR_BACKEND
from ebcc import numpy as np
from ebcc import tensor_backend as tb

# Try to import TBLIS if needed
if TENSOR_BACKEND == "tblis":
    try:
        try:
            import tblis_einsum
        except ImportError:
            from pyscf.tblis_einsum import tblis_einsum
    except ImportError as e:
        raise ImportError(
            "Can't find `tblis_einsum` module which is required for the `tblis` backend."
        ) from e

# Define the size of problem to fall back on NumPy
NUMPY_EINSUM_SIZE = 2000


class EinsumOperandError(ValueError):
    """Exception for invalid inputs to `einsum`."""

    pass


def _fallback_einsum(*operands, **kwargs):
    """Handle the fallback to `numpy.einsum`."""

    # Parse the kwargs
    kwargs = kwargs.copy()
    alpha = kwargs.pop("alpha", 1.0)
    beta = kwargs.pop("beta", 0.0)
    out = kwargs.pop("out", None)

    # Perform the contraction
    res = np.einsum(*operands, **kwargs)
    res *= alpha

    # Scale the output
    if out is not None:
        res += beta * out

    return res


def _contract(subscript, *args, **kwargs):
    """Contract a pair of terms with the chosen backend."""

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
    inp, out, args = np.core.einsumfunc._parse_einsum_input((subscript, a, b))
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
    ranges = {}
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
        buf = np.asarray(buf.reshape(shape_ct_flat), order="F" if buf.flags.f_contiguous else "C")

    # Perform the contraction
    ct = dot(at, bt, alpha=alpha, beta=beta, c=buf)

    # Reshape and transpose
    ct = ct.reshape(shape_ct, order="A")
    c = ct.transpose(order_ct)

    return c


def _contract_tblis(subscript, *args, **kwargs):
    """Contract a pair of terms using TBLIS for the NumPy backend."""

    alpha = kwargs.get("alpha", 1.0)
    beta = kwargs.get("beta", 0.0)
    buf = kwargs.get("out", None)

    # If this is called for more than 2 arguments, fall back
    if len(args) > 2:
        return _fallback_einsum(subscript, *args, **kwargs)

    # Make sure that the input are numpy arrays
    a, b = args = (np.asarray(args[0]), np.asarray(args[1]))

    # Check if we should use NumPy
    if min(a.size, b.size) < NUMPY_EINSUM_SIZE:
        return _fallback_einsum(subscript, *args, **kwargs)

    # Make sure it can be done via DGEMM
    indices = subscript.replace(",", "").replace("->", "")
    if any(indices.count(x) != 2 for x in set(indices)):
        return _fallback_einsum(subscript, *args, **kwargs)

    # Get the characters for each input and output
    inp, out, args = np.core.einsumfunc._parse_einsum_input((subscript, a, b))
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
    ranges = {}
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
    shape_c = tuple(ranges[x] for x in out)

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


def contract(subscript, *args, **kwargs):
    """
    Contract a pair of terms in an einsum. Supports additional keyword
    arguments `alpha` and `beta` which operate as `pyscf.lib.dot`. In some
    cases this will still require copying, but it minimises the memory
    overhead in simple cases.

    Parameters
    ----------
    subscript : str
        Subscript representation of the contraction.
    args : tuple of numpy.ndarray
        Arrays to contract.
    alpha : float, optional
        Scaling factor for contraction. Default value is `1.0`.
    beta : float, optional
        Scaling factor for the output. Default value is `0.0`.
    out : numpy.ndarray, optional
        Output array. If `None`, a new array is created. Default value is
        `None`.
    **kwargs : dict
        Additional keyword arguments to `numpy.einsum`.

    Returns
    -------
    array : numpy.ndarray
        Result of the contraction.
    """
    if TENSOR_BACKEND == "tblis":
        return _contract_tblis(subscript, *args, **kwargs)
    else:
        return _contract(subscript, *args, **kwargs)


def _parse_einsum_input(operands):
    """
    Parse the input to `einsum`. Copied from `numpy.core.einsumfunc`.

    Returns
    -------
    input_strings : str
        Parsed input strings
    output_string : str
        Parsed output string
    operands : list of array_like
        The operands to use in the numpy contraction

    Examples
    --------
    The operand list is simplified to reduce printing:

    >>> np.random.seed(123)
    >>> a = np.random.rand(4, 4)
    >>> b = np.random.rand(4, 4, 4)
    >>> _parse_einsum_input(('...a,...a->...', a, b))
    ('za,xza', 'xz', [a, b]) # may vary

    >>> _parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
    ('za,xza', 'xz', [a, b]) # may vary
    """

    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        # operands = [np.asarray(v) for v in operands[1:]]
        operands = [v for v in operands[1:]]

        # Ensure all characters are valid
        for s in subscripts:
            if s in ".,->":
                continue
            if s not in np.core.einsumfunc.einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        # operands = [np.asarray(v) for v in operand_list]
        operands = [v for v in operand_list]
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            "For this input type lists must contain " "either int or Ellipsis"
                        ) from e
                    subscripts += np.core.einsumfunc.einsum_symbols[s]
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for s in output_list:
                if s is Ellipsis:
                    subscripts += "..."
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            "For this input type lists must contain " "either int or Ellipsis"
                        ) from e
                    subscripts += np.core.einsumfunc.einsum_symbols[s]
    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(np.core.einsumfunc.einsum_symbols_set - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(",")
            out_sub = False

        for num, sub in enumerate(split_subscripts):
            if "." in sub:
                if (sub.count(".") != 3) or (sub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= len(sub) - 3

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace("...", "")
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace("...", rep_inds)

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
                if s not in (np.core.einsumfunc.einsum_symbols):
                    raise ValueError("Character %s is not a valid symbol." % s)
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
            if s not in np.core.einsumfunc.einsum_symbols:
                raise ValueError("Character %s is not a valid symbol." % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if output_subscript.count(char) != 1:
            raise ValueError("Output character %s appeared more than once in " "the output." % char)
        if char not in input_subscripts:
            raise ValueError("Output character %s did not appear in the input" % char)

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(",")) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the " "number of operands.")

    return (input_subscripts, output_subscript, operands)


def _einsum_numpy(subscript, *args, **kwargs):
    """
    Evaluate an Einstein summation convention on the operands for the
    `"numpy"` or `"tblis"` backend.
    """

    # Perform the contraction
    if len(args) < 2:
        # If it's just a transpose, use the fallback
        out = _fallback_einsum(subscript, *args, **kwargs)
    elif len(args) < 3:
        # If it's a single contraction, call the backend directly
        out = contract(subscript, *args, **kwargs)
    else:
        # If it's a chain of contractions, use the path optimizer
        optimize = kwargs.pop("optimize", True)
        args = list(args)
        contractions = np.einsum_path(subscript, *args, optimize=optimize, einsum_call=True)[1]
        for contraction in contractions:
            inds, idx_rm, einsum_str, remain = contraction[:4]
            operands = [args.pop(x) for x in inds]
            out = contract(einsum_str, *operands)
            args.append(out)

    return out


def _einsum_jax(subscript, *args, **kwargs):
    """
    Evaluate an Einstein summation convention on the operands for the
    `"jax"` backend.
    """
    optimize = kwargs.pop("optimize", True)
    return tb.einsum(subscript, *args, optimize=optimize, **kwargs)


def _einsum_ctf(subscript, *args, **kwargs):
    """
    Evaluate an Einstein summation convention on the operands for the
    `"ctf"` backend.
    """
    kwargs.pop("optimize", True)  # CTF does not support this
    return tb.einsum(subscript, *args, **kwargs)


def einsum(*operands, **kwargs):
    """
    Evaluate an Einstein summation convention on the operands.

    Using the Einstein summation convention, many common
    multi-dimensional, linear algebraic array operations can be
    represented in a simple fashion. In *implicit* mode `einsum`
    computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical
    Einstein summation operations, by disabling, or forcing summation
    over specified subscript labels.

    See the `numpy.einsum` documentation for clarification.

    Parameters
    ----------
    operands : list
        Any valid input to `numpy.einsum`.
    alpha : float, optional
        Scaling factor for the contraction. Default value is `1.0`.
    beta : float, optional
        Scaling factor for the output. Default value is `0.0`.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    contract : callable, optional
        The function to use for contraction. Default value is
        `contract`.
    optimize : bool, optional
        If `True`, use the `numpy.einsum_path` to optimize the
        contraction. Default value is `True`.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    Notes
    -----
    This function may use `numpy.einsum`, `pyscf.lib.einsum`, or
    `tblis_einsum` as a backend, depending on the problem size and the
    modules available.
    """

    # Parse the kwargs
    inp, out, args = _parse_einsum_input(operands)
    subscript = "%s->%s" % (inp, out)

    # Call the appropriate backend
    if TENSOR_BACKEND in ("numpy", "tblis"):
        return _einsum_numpy(subscript, *args, **kwargs)
    elif TENSOR_BACKEND == "jax":
        return _einsum_jax(subscript, *args, **kwargs)
    elif TENSOR_BACKEND == "ctf":
        return _einsum_ctf(subscript, *args, **kwargs)
    else:
        raise ValueError("Unknown tensor backend: %s" % TENSOR_BACKEND)
