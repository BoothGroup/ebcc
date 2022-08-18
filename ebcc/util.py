"""Utilities.
"""

import functools
import inspect
import itertools
import logging
import sys

import numpy as np


class InheritedType:
    pass


Inherited = InheritedType()


def factorial(n):
    """Return the factorial of n."""

    if n in (0, 1):
        return 1
    else:
        return n * factorial(n - 1)


def permute_string(string, permutation):
    """Permute a string."""
    return "".join([string[i] for i in permutation])


def tril_indices_ndim(n, dims, include_diagonal=False):
    """Return lower triangular indices for a multidimensional array."""

    ranges = [np.arange(n)] * dims

    if dims == 0:
        return tuple()
    elif dims == 1:
        return (ranges[0],)

    if include_diagonal:
        func = np.greater_equal
    else:
        func = np.greater

    slices = [
        tuple(slice(None) if i == j else np.newaxis for i in range(dims))
        for j in range(dims)
    ]

    casted = [rng[ind] for rng, ind in zip(ranges, slices)]
    mask = functools.reduce(np.logical_and, [func(a, b) for a, b in zip(casted[:-1], casted[1:])])

    tril = tuple(
        np.broadcast_to(inds, mask.shape)[mask] for inds in np.indices(mask.shape, sparse=True)
    )

    return tril


def ntril_ndim(n, dims, include_diagonal=False):
    """Return the number of elements in an n-dimensional lower triangle."""

    assert dims < n

    #if include_diagonal:
    #    return sum(1 for tup in itertools.combinations_with_replacements(range(n), dims))
    #else:
    #    return sum(1 for tup in itertools.combinations(range(n), dims))

    offset = int(include_diagonal)
    out = 1

    for i in range(dims):
        out *= (n+offset)
        offset -= 1

    out //= factorial(dims)

    return out


def minimum_swaps(lst):
    """Find the minimum number of swaps needed to sort lst."""

    lst = np.argsort(np.argsort(lst))
    n = 0
    i = 0

    while i < (len(lst) - 1):
        while lst[i] != (i + 1):
            lst[lst[i] - 1], lst[i] = lst[i], lst[lst[i] - 1]
            n += 1
        i += 1

    return n


def index_axes(arr, *inds):
    """Apply each of inds over the dimensions of arr."""

    null = slice(None)
    for n, ind in enumerate(inds):
        arr = arr[(null,) * n + (ind,)]

    return arr


def generate_spin_combinations(n, excited=False):
    """Generate combinations of spin components for a given number
    of occupied and virtual axes.

    Parameters
    ----------
    n : int
        Order of cluster amplitude.
    excited : bool, optional
        If True, treat the amplitudes as excited. Default value is
        `None`.

    Yields
    ------
    spin : str
        String of spin combination.

    Examples
    --------
    >>> generate_spin_combinations(1)
    ['aa', 'bb']
    >>> generate_spin_combinations(2)
    ['aaaa', 'abab', 'baba', 'bbbb']
    >>> generate_spin_combinations(2, excited=True)
    ['aaa', 'aba', 'bab', 'bbb']
    """

    for tup in itertools.product(("a", "b"), repeat=n):
        comb = "".join(list(tup) * 2)
        if excited:
            comb = comb[:-1]
        yield comb


def permutations_with_signs(seq):
    """Generate permutations of seq, yielding also a sign which is
    equal to +1 for an even number of swaps, and -1 for an odd number
    of swaps.
    """

    def _permutations(seq):
        if not seq:
            return [[]]

        items = []
        for i, item in enumerate(_permutations(seq[:-1])):
            inds = range(len(item) + 1)
            if i % 2 == 0:
                inds = reversed(inds)
            items += [item[:i] + seq[-1:] + item[i:] for i in inds]

        return items

    return [(item, -1 if i % 2 else 1) for i, item in enumerate(_permutations(list(seq)))]


def get_symmetry_factor(*numbers):
    """Get a floating point value corresponding to the factor from
    the neglection of symmetry in repeated indices.

    Parameters
    ----------
    numbers : tuple of int
        Multiplicity of each distinct degree of freedom.

    Examples
    --------
    >>> build_factor_array(1, 1, 0)
    1.0
    >>> build_factor_array(2, 2, 0)
    0.25
    >>> build_factor_array(3, 2, 1)
    0.125
    """

    ntot = 0
    for n in (nocc, nvir, nbos):
        ntot += max(0, n-1)

    return 1.0 / (2.0**ntot)


def inherit_docstrings(cls):
    """Inherit docstring from superclass."""

    for name, func in inspect.getmembers(cls, inspect.isfunction):
        if not func.__doc__:
            for parent in cls.__mro__[1:]:
                if hasattr(parent, name):
                    func.__doc__ = getattr(parent, name).__doc__

    return cls


def antisymmetrise_array(v, axes=(0, 1)):
    """Antisymmetrise an array."""

    v_as = np.zeros_like(v)

    for perm, sign in permutations_with_signs(axes):
        transpose = list(range(v.ndim))
        for i, ax in enumerate(transpose):
            if ax in axes:
                j = axes.index(ax)
                transpose[i] = perm[j]
        v_as += sign * v.transpose(transpose).copy()

    return v_as


def is_mixed_spin(spin):
    return len(set(spin)) != 1


def compress_axes(subscript, array, include_diagonal=False, out=None):
    """Compress an array into lower-triangular representations using
    an einsum-like input.

    >>> t2 = np.zeros((4, 4, 10, 10))
    >>> compress_axes("iiaa", t2).shape
    (6, 45)
    """
    # TODO out

    assert "->" not in subscript

    # Substitute the input characters so that they are ordered:
    subs = {}
    for i, char in enumerate(subscript):
        if char not in subs:
            subs[char] = chr(97+i)
    subscript = "".join([subs[s] for s in subscript])

    # Reshape array so that all axes of the same character are adjacent:
    arg = np.argsort(list(subscript))
    array = array.transpose(arg)
    subscript = permute_string(subscript, arg)

    # Reshape array so that all axes of the same character are flattened:
    sizes = {}
    for char, n in zip(subscript, array.shape):
        if char in sizes:
            assert sizes[char] == n
        else:
            sizes[char] = n
    array = array.reshape([sizes[char] ** subscript.count(char) for char in sorted(set(subscript))])

    # For each axis type, get the necessary lower-triangular indices:
    indices = [
            tril_indices_ndim(sizes[char], subscript.count(char), include_diagonal=include_diagonal)
            for char in sorted(set(subscript))
    ]
    indices = [
            np.ravel_multi_index(ind, (sizes[char],) * subscript.count(char))
            for ind, char in zip(indices, sorted(set(subscript)))
    ]

    # Apply the indices:
    indices = [
            ind[tuple(np.newaxis if i != j else slice(None) for i in range(len(indices)))]
            for j, ind in enumerate(indices)
    ]
    array_flat = array[tuple(indices)]

    return array_flat


def decompress_axes(subscript, array_flat, shape=None, include_diagonal=False, symmetry=None, out=None):
    """Reverse operation of `compress_axes`, subscript input is the
    same. The input symmetry is a string of the same length as
    subscript, with a "+" indicating symmetry and "-" antisymmetry.
    """

    assert "->" not in subscript
    assert shape is not None or out is not None

    # Get symmetry string if needed:
    if symmetry is None:
        symmetry = "-" * len(subscript)

    # Initialise decompressed array
    if out is None:
        array = np.zeros(shape)
    else:
        array = out

    # Substitute the input characters so that they are ordered:
    subs = {}
    for i, char in enumerate(subscript):
        if char not in subs:
            subs[char] = chr(97+i)
    subscript = "".join([subs[s] for s in subscript])

    # Reshape array so that all axes of the same character are adjacent:
    arg = np.argsort(list(subscript))
    array = array.transpose(arg)
    subscript = permute_string(subscript, arg)

    # Reshape array so that all axes of the same character are flattened:
    sizes = {}
    for char, n in zip(subscript, array.shape):
        if char in sizes:
            assert sizes[char] == n
        else:
            sizes[char] = n
    array = array.reshape([sizes[char] ** subscript.count(char) for char in sorted(set(subscript))])

    # Check the symmetry string, and compress it:
    n = 0
    symmetry_compressed = ""
    for char in sorted(set(subscript)):
        assert len(set(symmetry[n:n+subscript.count(char)])) == 1
        symmetry_compressed += symmetry[n]
        n += subscript.count(char)

    # For each axis type, get the necessary lower-triangular indices:
    indices = [
            tril_indices_ndim(sizes[char], subscript.count(char), include_diagonal=include_diagonal)
            for char in sorted(set(subscript))
    ]

    # Iterate over permutations with signs:
    for tup in itertools.product(*[permutations_with_signs(ind) for ind in indices]):
        indices_perm, signs = zip(*tup)
        signs = [s if symm == "-" else 1 for s, symm in zip(signs, symmetry_compressed)]

        # Apply the indices:
        indices_perm = [
                np.ravel_multi_index(ind, (sizes[char],) * subscript.count(char))
                for ind, char in zip(indices_perm, sorted(set(subscript)))
        ]
        indices_perm = [
                ind[tuple(np.newaxis if i != j else slice(None) for i in range(len(indices_perm)))]
                for j, ind in enumerate(indices_perm)
        ]
        array[tuple(indices_perm)] = array_flat * np.prod(signs)

    # Reshape array to non-flattened format
    array = array.reshape(sum([(sizes[char],) * subscript.count(char) for char in sorted(set(subscript))], tuple()))

    # Undo transpose:
    arg = np.argsort(arg)
    array = array.transpose(arg)

    return array
