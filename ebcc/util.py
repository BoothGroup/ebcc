"""Utilities.
"""

import functools
import inspect
import itertools
import logging
import sys
import time
import types

import numpy as np
from pyscf.lib import direct_sum
from pyscf.lib import einsum as pyscf_einsum


class InheritedType:
    pass


Inherited = InheritedType()


ModelNotImplemented = NotImplementedError


class Namespace:
    """Replacement for SimpleNamespace, which does not trivially allow
    conversion to a dict for heterogenously nested objects.
    """

    def __init__(self, **kwargs):
        self.__dict__["_keys"] = set()
        for key, val in kwargs.items():
            self[key] = val

    def __setitem__(self, key, val):
        self._keys.add(key)
        self.__dict__[key] = val

    def __getitem__(self, key):
        if key not in self._keys:
            raise IndexError(key)
        return self.__dict__[key]

    def __delitem__(self, key):
        if key not in self._keys:
            raise IndexError(key)
        del self.__dict__[key]

    __setattr__ = __setitem__

    def __iter__(self):
        for key in self._keys:
            yield (key, self[key])

    def __eq__(self, other):
        return dict(self) == dict(other)

    def __contains__(self, key):
        return key in self._keys


class Timer:
    def __init__(self):
        self.t_init = time.perf_counter()
        self.t_prev = time.perf_counter()
        self.t_curr = time.perf_counter()

    def lap(self):
        self.t_prev, self.t_curr = self.t_curr, time.perf_counter()
        return self.t_curr - self.t_prev

    __call__ = lap

    def total(self):
        return time.perf_counter() - self.t_init

    @staticmethod
    def format_time(seconds, precision=2):
        """Return a formatted time."""

        seconds, milliseconds = divmod(seconds, 1)
        milliseconds *= 1000
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

        out = []
        if hours:
            out.append("%d h" % hours)
        if minutes:
            out.append("%d m" % minutes)
        if seconds:
            out.append("%d s" % seconds)
        if milliseconds:
            out.append("%d ms" % milliseconds)

        return " ".join(out[-max(precision, len(out)) :])


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
        tuple(slice(None) if i == j else np.newaxis for i in range(dims)) for j in range(dims)
    ]

    casted = [rng[ind] for rng, ind in zip(ranges, slices)]
    mask = functools.reduce(np.logical_and, [func(a, b) for a, b in zip(casted[:-1], casted[1:])])

    tril = tuple(
        np.broadcast_to(inds, mask.shape)[mask] for inds in np.indices(mask.shape, sparse=True)
    )

    return tril


def ntril_ndim(n, dims, include_diagonal=False):
    """Return the number of elements in an n-dimensional lower triangle."""

    # FIXME hack until this function is fixed:
    if include_diagonal:
        return sum(1 for tup in itertools.combinations_with_replacement(range(n), dims))
    else:
        return sum(1 for tup in itertools.combinations(range(n), dims))

    offset = int(include_diagonal)
    out = 1

    for i in range(dims):
        out *= n + offset
        offset -= 1

    out //= factorial(dims)

    return out


def generate_spin_combinations(n, excited=False, unique=False):
    """Generate combinations of spin components for a given number
    of occupied and virtual axes.

    Parameters
    ----------
    n : int
        Order of cluster amplitude.
    excited : bool, optional
        If True, treat the amplitudes as excited. Default value is
        `False`.
    unique : bool, optional
        If True, return only unique combinations.

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
    >>> generate_spin_combinations(2, unique=True)
    ['aaaa', 'abab', 'bbbb']
    """

    if unique:
        check = set()

    for tup in itertools.product(("a", "b"), repeat=n):
        comb = "".join(list(tup) * 2)
        if excited:
            comb = comb[:-1]

        if unique:
            sorted_comb = "".join(sorted(comb[:n])) + "".join(sorted(comb[n:]))
            if sorted_comb in check:
                continue
            check.add(sorted_comb)

            if not excited:  # FIXME
                nab = (comb[:n].count("a"), comb[:n].count("b"))
                if nab == (n // 2, n - n // 2):
                    comb = ("ab" * n)[:n] * 2
                elif nab == (n - n // 2, n // 2):
                    comb = ("ba" * n)[:n] * 2

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
    >>> build_symmetry_factor(1, 1)
    1.0
    >>> build_symmetry_factor(2, 2)
    0.25
    >>> build_symmetry_factor(3, 2, 1)
    0.125
    """

    ntot = 0
    for n in numbers:
        ntot += max(0, n - 1)

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
    # TODO can this be OpenMP parallel?

    assert "->" not in subscript

    # Substitute the input characters so that they are ordered:
    subs = {}
    i = 0
    for char in subscript:
        if char not in subs:
            subs[char] = chr(97 + i)
            i += 1
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


def decompress_axes(
    subscript, array_flat, shape=None, include_diagonal=False, symmetry=None, out=None
):
    """Reverse operation of `compress_axes`, subscript input is the
    same. The input symmetry is a string of the same length as
    subscript, with a "+" indicating symmetry and "-" antisymmetry.
    """
    # FIXME: if you pass out=array here, it doesn't work - it's not touching all parts of the array??
    #        --> I guess the diagonals actually!! set out to zero first if used as input.

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
    i = 0
    for char in subscript:
        if char not in subs:
            subs[char] = chr(97 + i)
            i += 1
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
        assert len(set(symmetry[n : n + subscript.count(char)])) == 1
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
        shape = array[tuple(indices_perm)].shape
        array[tuple(indices_perm)] = array_flat.reshape(shape) * np.prod(signs)

    # Reshape array to non-flattened format
    array = array.reshape(
        sum([(sizes[char],) * subscript.count(char) for char in sorted(set(subscript))], tuple())
    )

    # Undo transpose:
    arg = np.argsort(arg)
    array = array.transpose(arg)

    return array


def get_compressed_size(subscript, **sizes):
    """Get the size of a compressed representation of a matrix
    based on the subscript input to `compressed_axes` and the
    sizes of each character.

    >>> get_compressed_shape("iiaa", i=5, a=3)
    30
    """

    n = 1
    for char in set(subscript):
        dims = subscript.count(char)
        n *= ntril_ndim(sizes[char], dims)

    return n


def symmetrise(subscript, array, symmetry=None, apply_factor=True):
    """Enforce a symmetry in an array. `subscript` and `symmetry` are
    as `decompress_axes`.
    """

    # Substitute the input characters so that they are ordered:
    subs = {}
    i = 0
    for char in subscript:
        if char not in subs:
            subs[char] = chr(97 + i)
            i += 1
    subscript = "".join([subs[s] for s in subscript])

    # Check the symmetry string, and compress it:
    if symmetry is None:
        symmetry = "-" * len(subscript)
    n = 0
    symmetry_compressed = ""
    for char in sorted(set(subscript)):
        assert len(set(symmetry[n : n + subscript.count(char)])) == 1
        symmetry_compressed += symmetry[n]
        n += subscript.count(char)

    # Iterate over permutations and signs:
    array_as = np.zeros_like(array)
    groups = tuple(sorted(set(zip(sorted(set(subscript)), symmetry_compressed))))  # don't ask
    inds = [tuple(i for i, s in enumerate(subscript) if s == char) for char, symm in groups]
    for tup in itertools.product(*(permutations_with_signs(ind) for ind in inds)):
        perms, signs = zip(*tup)
        perm = list(range(len(subscript)))
        for inds_part, perms_part in zip(inds, perms):
            for i, p in zip(inds_part, perms_part):
                perm[i] = p
        perm = tuple(perm)
        sign = np.prod(signs) if symmetry[perm[0]] == "-" else 1
        array_as = array_as + sign * array.transpose(perm)

    if apply_factor:
        # Apply factor
        sizes = [subscript.count(s) for s in sorted(set(subscript))]
        array_as = array_as * get_symmetry_factor(*sizes)

    return array_as


ov_2e = [
    "oooo",
    "ooov",
    "oovo",
    "ovoo",
    "vooo",
    "oovv",
    "ovov",
    "ovvo",
    "voov",
    "vovo",
    "vvoo",
    "ovvv",
    "vovv",
    "vvov",
    "vvvo",
    "vvvv",
]


def pack_2e(*args):
    # TODO remove
    # args should be in the order of ov_2e

    assert len(args) == len(ov_2e)

    nocc = args[0].shape
    nvir = args[-1].shape
    occ = [slice(None, n) for n in nocc]
    vir = [slice(n, None) for n in nocc]
    out = np.zeros(tuple(no + nv for no, nv in zip(nocc, nvir)))

    for key, arg in zip(ov_2e, args):
        slices = [occ[i] if x == "o" else vir[i] for i, x in enumerate(key)]
        out[tuple(slices)] = arg

    return out


def einsum(*operands, symmetry=False, **kwargs):  # pragma: no cover
    """Dispatch an einsum. If `symmetry`, then assume that all arrays
    are totally symmetric within occupied, virtual and bosonic
    sectors. If `symmetry` is an iterable, then it should provide
    permutations corresponding to each input array which give the
    desired symmetry.
    """
    # TODO custom symmetry
    # TODO assert the symmetry?

    inp, out, args = np.core.einsumfunc._parse_einsum_input(operands)
    subscript = "%s->%s" % (inp, out)

    if not symmetry:
        return pyscf_einsum(subscript, *args, **kwargs)

    raise NotImplementedError("Work in progress")

    try:
        # Two input arrays only:
        assert len(args) == 2
        assert subscript.count(",") == 1

        array1, array2 = args
        indices = set(subscript) - {",", "-", ">"}
        lhs1, lhs2, rhs = subscript.replace("->", ",").split(",")

        # No internal traces:
        assert len(lhs1) == len(set(lhs1))
        assert len(lhs2) == len(set(lhs2))

        # No repeated external indices, and no free indices:
        for index in lhs1 + lhs2:
            if index in rhs:
                assert (int(index in lhs1) + int(index in lhs2)) == 1
            else:
                assert index in lhs1 and index in lhs2

    except AssertionError:
        return pyscf_einsum(subscript, *args, **kwargs)

    # Categorise the indices:
    categories = {}
    sizes = {}
    for index in indices:
        category = 0 if index not in rhs else (1 if index in lhs1 else 2)
        sector = 0 if index in "ijklmnop" else (1 if index in "abcdefgh" else 2)
        categories[index] = category * 3 + sector
        sizes[index] = (array1.shape + array2.shape)[(lhs1 + lhs2).index(index)]

    # Get compressed and flattened subscripts:
    lhs1_comp = "".join([chr(97 + categories[i]) for i in lhs1])
    lhs2_comp = "".join([chr(97 + categories[i]) for i in lhs2])
    rhs_comp = "".join([chr(97 + categories[i]) for i in rhs])
    lhs1_flat = "".join(list(dict.fromkeys(lhs1_comp)))
    lhs2_flat = "".join(list(dict.fromkeys(lhs2_comp)))
    rhs_flat = "".join(list(dict.fromkeys(rhs_comp)))

    # Apply a factor depending on the symmetry of the dummies:
    dummies = set(i_comp for i, i_comp in zip(lhs1, lhs1_comp) if categories[i] < 3)
    for index in set(lhs1_comp):
        if index in dummies:
            mask = [i == index for i in lhs1_comp]
            factor = np.zeros(np.array(array1.shape)[mask])
            inds = tril_indices_ndim(factor.shape[0], factor.ndim, include_diagonal=True)
            for perm, _ in permutations_with_signs(inds):
                factor[tuple(perm)] += 1
            factor = 2 ** (factor.ndim - factor)
            lhs_factor = "".join([i for i in lhs1 if categories[i] < 3])
            subscript = lhs1 + "," + lhs_factor + "->" + lhs1
            array1 = pyscf_einsum(subscript, array1, factor)

    # Compress the arrays:
    array1_flat = compress_axes(lhs1_comp, array1, include_diagonal=True)
    array2_flat = compress_axes(lhs2_comp, array2, include_diagonal=True)

    # Dispatch the einsum:
    subscript_flat = "{lhs1},{lhs2}->{rhs}".format(lhs1=lhs1_flat, lhs2=lhs2_flat, rhs=rhs_flat)
    output_flat = pyscf_einsum(subscript_flat, array1_flat, array2_flat, **kwargs)

    # Decompress the output:
    rank = len(rhs_comp)
    shape = tuple(sizes[i] for i in rhs)
    output = decompress_axes(
        rhs_comp, output_flat, include_diagonal=True, shape=shape, symmetry="+" * rank
    )

    return output


def unique(lst):
    """Get unique elements of a list.
    """

    done = set()
    out = []
    for el in lst:
        if el not in done:
            out.append(el)
            done.add(el)

    return out


def einsum_symmetric(*operands, symmetries=[], symmetrise_dummies=False, **kwargs):
    """Dispatch an einsum in a symmetric representation. The argument
    `symmetries` should be an iterable of the symmetry subscripts for
    each input and for the output array, in the format of the
    `compress_axes` and `decompress_axes` functions. Assumes that the
    phase of the symmetries is positive, i.e. the arrays are not
    antisymmetric.
    """

    assert not symmetrise_dummies

    inp, out, args = np.core.einsumfunc._parse_einsum_input(operands)
    subscript = "%s->%s" % (inp, out)

    # Get the sizes of each index
    sizes = {}
    for part, arg in zip(inp.split(","), args):
        for p, size in zip(part, arg.shape):
            sizes[p] = size

    # Find the number of dummy indices in each input
    dummies = [[i not in out for i in part] for part in inp.split(",")]

    # Make sure that external and dummy variables are compressed
    # separately
    not_used = [char for char in "abcdefghijklmnopqrstuvwxyz" if not any(char in s for s in symmetries)]
    for i, symmetry in enumerate(symmetries):
        char_map = {}
        new_symmetry = ""
        for s, d in zip(symmetry, dummies[i]):
            if d:
                if (s not in char_map) or (not symmetrise_dummies):
                    char_map[s] = not_used.pop(0)
                new_symmetry += char_map[s]
            else:
                new_symmetry += s
        symmetries[i] = new_symmetry

    # Get the flattened subscripts
    subscripts_flat = []
    indices = {}
    for part, symmetry in zip(inp.split(","), symmetries):
        done = set()
        part_flat = ""
        for symm in symmetry:
            if symm in done:
                continue

            part_flat_contr = "".join([p for p, s in zip(part, symmetry) if s == symm])
            part_flat_contr = "".join(sorted(part_flat_contr)[0] * len(part_flat_contr))
            part_flat += part_flat_contr

            done.add(symm)

        subscripts_flat.append(part_flat)

        for i, j in zip(part, part_flat):
            if i in indices and indices[i] != j:
                raise ValueError
            indices[i] = j

    subscripts_flat.append("".join([indices[i] for i in out]))
    subscripts_flat_uniq = ["".join(unique(list(subscript))) for subscript in subscripts_flat]

    # Compress the inputs
    args_flat = [
            compress_axes(subscript, arg, include_diagonal=True)
            for subscript, arg in zip(subscripts_flat, args)
    ]

    # Get the factors for the compressed dummies  # TODO improve
    if symmetrise_dummies:
        # Get a flattened version of the dummies
        dummies_flat = []
        for dummy, part in zip(dummies, subscripts_flat):
            done = set()
            dummies_flat_contr = []
            for d, i in zip(dummy, part):
                if i not in done:
                    dummies_flat_contr.append(d)
                    done.add(i)
            dummies_flat.append(dummies_flat_contr)

        # Apply the factors
        for i in range(len(args)):
            factors = []
            for inds in itertools.product(*[
                    itertools.combinations_with_replacement(
                        range(args[i].shape[subscripts_flat[i].index(x)]),
                        subscripts_flat[i].count(x),
                    )
                    for x in unique(list(subscripts_flat[i]))
            ]):
                factor = 1
                for dumm, tup in zip(dummies_flat[i], inds):
                    if dumm:
                        factor *= 2 ** (len(set(tup)) - 1)
                factors.append(factor)
            factors = np.array(factors).reshape(args_flat[i].shape)
            args_flat[i] *= factors

    # Dispatch the einsum
    subscript_flat = ",".join(subscripts_flat_uniq[:-1])
    subscript_flat += "->"
    subscript_flat += subscripts_flat_uniq[-1]
    out_flat = pyscf_einsum(subscript_flat, *args_flat, **kwargs)

    # Decompress the output
    shape = tuple(sizes[i] for i in out)
    out = decompress_axes(subscripts_flat[-1], out_flat, include_diagonal=True, shape=shape, symmetry="+"*len(shape))

    return out
