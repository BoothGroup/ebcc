# type: ignore
"""Cyclops Tensor Framework backend."""

import ctf
import numpy
import opt_einsum


def __getattr__(name):
    """Get the attribute from CTF."""
    return getattr(ctf, name)


bool_ = numpy.bool_
asarray = ctf.astensor


def astype(obj, dtype):
    return obj.astype(dtype)


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
