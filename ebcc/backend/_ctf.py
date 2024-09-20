# type: ignore
"""Cyclops Tensor Framework backend."""

import ctf
import opt_einsum


def __getattr__(name):
    """Get the attribute from CTF."""
    return getattr(ctf, name)


def astype(obj, dtype):
    return obj.astype(dtype)


def einsum_path(*args, **kwargs):
    """Evaluate the lowest cost contraction order for an einsum expression."""
    kwargs = dict(kwargs)
    if kwargs.get("optimize", True) is True:
        kwargs["optimize"] = "optimal"
    return opt_einsum.contract_path(*args, **kwargs)
