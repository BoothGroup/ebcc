# type: ignore
"""JAX backend."""

import jax
import jax.numpy
import opt_einsum

jax.config.update("jax_enable_x64", True)


def __getattr__(name):
    """Get the attribute from the NumPy drop-in."""
    return getattr(jax.numpy, name)


_jax_ix_ = jax.numpy.ix_


def ix_(*args):  # noqa: D103
    args_ = []
    for arg in args:
        if isinstance(arg, jax.numpy.ndarray) and arg.dtype == jax.numpy.bool_:
            arg = jax.numpy.where(arg)[0]
        args_.append(arg)
    return _jax_ix_(*args_)


def einsum_path(*args, **kwargs):
    """Evaluate the lowest cost contraction order for an einsum expression."""
    kwargs = dict(kwargs)
    if kwargs.get("optimize", True) is True:
        kwargs["optimize"] = "optimal"
    return opt_einsum.contract_path(*args, **kwargs)
