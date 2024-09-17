# type: ignore
"""JAX backend."""

import jax
import jax.numpy
import opt_einsum

jax.config.update("jax_enable_x64", True)


def __getattr__(name):
    """Get the attribute from the NumPy drop-in."""
    return getattr(jax.numpy, name)


def _argsort(strings, **kwargs):
    if not isinstance(strings, jax.numpy.ndarray):
        return jax.numpy.asarray(
            sorted(range(len(strings)), key=lambda i: strings[i]), dtype=jax.numpy.int32
        )
    return _jax_argsort(strings, **kwargs)


_jax_argsort = jax.numpy.argsort
jax.numpy.argsort = _argsort


def einsum_path(*args, **kwargs):  # type: ignore
    """Evaluate the lowest cost contraction order for an einsum expression."""
    kwargs = dict(kwargs)
    if kwargs.get("optimize", True) is True:
        kwargs["optimize"] = "optimal"
    return opt_einsum.contract_path(*args, **kwargs)
