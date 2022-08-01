"""Utilities.
"""

import sys
import logging
import inspect
import itertools
import functools
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


def tril_indices_ndim(n, dims, include_diagonal=False):
    """Return lower triangular indices for a multidimensional array."""

    ranges = [np.arange(n)] * dims

    if dims == 0:
        return tuple()
    elif dims == 1:
        return (ranges[0],)

    if include_diagonal:
        tri = functools.reduce(np.greater_equal.outer, ranges)
    else:
        tri = functools.reduce(np.greater.outer, ranges)

    tril = tuple(
        np.broadcast_to(inds, tri.shape)[tri] for inds in np.indices(tri.shape, sparse=True)
    )

    return tril


def ntril_ndim(n, dims, include_diagonal=False):
    """Return the number of elements in an n-dimensional lower triangle."""

    offset = 0 if include_diagonal else -1

    out = n + offset
    offset += 1

    for i in range(1, dims):
        out *= n + offset
        offset += 1

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


def inherit_docstrings(cls):
    """Inherit docstring from superclass."""

    for name, func in inspect.getmembers(cls, inspect.isfunction):
        if not func.__doc__:
            for parent in cls.__mro__[1:]:
                if hasattr(parent, name):
                    func.__doc__ = getattr(parent, name).__doc__

    return cls
