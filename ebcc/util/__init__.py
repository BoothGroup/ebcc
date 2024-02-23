"""Utilities."""

from ebcc.util.einsumfunc import direct_sum, dot, einsum
from ebcc.util.inheritance import InheritDocstrings, Inherited, InheritedType, _mro, has_docstring
from ebcc.util.misc import ModelNotImplemented, Namespace, Timer
from ebcc.util.permutations import (
    antisymmetrise_array,
    combine_subscripts,
    compress_axes,
    decompress_axes,
    factorial,
    generate_spin_combinations,
    get_compressed_size,
    get_symmetry_factor,
    is_mixed_spin,
    ntril_ndim,
    pack_2e,
    permutations_with_signs,
    permute_string,
    symmetrise,
    tril_indices_ndim,
    unique,
)
