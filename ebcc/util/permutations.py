"""Symmetry and permutational utilities."""

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc import util
from ebcc.backend import _put

if TYPE_CHECKING:
    from typing import Any, Generator, Hashable, Iterable, Optional

    from numpy import floating, integer
    from numpy.typing import NDArray

    T = floating


def factorial(n: int) -> int:
    """Return the factorial of `n`."""
    if n in (0, 1):
        return 1
    else:
        return n * factorial(n - 1)


def permute_string(string: str, permutation: tuple[int, ...]) -> str:
    """Permute a string.

    Args:
        string: String to permute.
        permutation: Permutation to apply.

    Returns:
        Permuted string.

    Examples:
        >>> permute_string("abcd", (2, 0, 3, 1))
        "cbda"
    """
    return "".join([string[i] for i in permutation])


def get_string_permutation(string: str, target: str) -> tuple[int, ...]:
    """Get the permutation to transform one string into another.

    Args:
        string: Initial string.
        target: Target string.

    Returns:
        Permutation to transform `string` into `target`.

    Examples:
        >>> get_string_permutation("abcd", "cbda")
        (2, 0, 3, 1)
        >>> get_string_permutation("iijj", "jjii")
        (2, 3, 0, 1)
    """
    # Find the indices of each character in the string
    indices: dict[str, list[int]] = {char: [] for char in set(string)}
    for i, char in enumerate(string):
        indices[char].append(i)

    # Get the permutation
    perm: list[int] = []
    for char in target:
        perm.append(indices[char].pop(0))

    return tuple(perm)


def tril_indices_ndim(
    n: int, dims: int, include_diagonal: Optional[bool] = False
) -> tuple[NDArray[integer], ...]:
    """Return lower triangular indices for a multidimensional array.

    Args:
        n: Size of each dimension.
        dims: Number of dimensions.
        include_diagonal: If True, include diagonal elements.

    Returns:
        Lower triangular indices for each dimension.
    """
    ranges = [np.arange(n)] * dims
    if dims == 1:
        return (ranges[0],)
    # func: Callable[[Any, ...], Any] = np.greater_equal if include_diagonal else np.greater

    slices = [tuple(slice(None) if i == j else None for i in range(dims)) for j in range(dims)]

    casted = [rng[ind] for rng, ind in zip(ranges, slices)]
    if include_diagonal:
        mask = functools.reduce(
            lambda x, y: x & y, map(lambda x, y: x >= y, casted[:-1], casted[1:])
        )
    else:
        mask = functools.reduce(
            lambda x, y: x & y, map(lambda x, y: x > y, casted[:-1], casted[1:])
        )

    tril = tuple(
        np.broadcast_to(inds, mask.shape)[mask] for inds in np.indices(mask.shape, sparse=True)
    )

    return tril


def ntril_ndim(n: int, dims: int, include_diagonal: Optional[bool] = False) -> int:
    """Return `len(tril_indices_ndim(n, dims, include_diagonal))`."""

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


def generate_spin_combinations(
    n: int, excited: Optional[bool] = False, unique: Optional[bool] = False
) -> Generator[str, None, None]:
    """Generate combinations of spin components for a given number of occupied and virtual axes.

    Args:
        n: Order of cluster amplitude.
        excited: If True, treat the amplitudes as excited.
        unique: If True, return only unique combinations.

    Returns:
        List of spin combinations.

    Examples:
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


def permutations_with_signs(seq: Iterable[Any]) -> list[tuple[Any, int]]:
    """Return permutations of a sequence with a sign indicating the number of swaps.

    The sign is equal to +1 for an even number of swaps, and -1 for an odd number of swaps.

    Args:
        seq: Sequence to permute.

    Returns:
        List of tuples of the form (permuted, sign).
    """

    def _permutations(seq: list[Any]) -> list[list[Any]]:
        if not seq:
            return [[]]

        items = []
        for i, item in enumerate(_permutations(seq[:-1])):
            if i % 2 == 1:
                inds = range(len(item) + 1)
            else:
                inds = range(len(item), -1, -1)
            items += [item[:i] + seq[-1:] + item[i:] for i in inds]

        return items

    return [(item, -1 if i % 2 else 1) for i, item in enumerate(_permutations(list(seq)))]


def get_symmetry_factor(*numbers: int) -> float:
    """Get a value corresponding to the factor from the neglection of symmetry in repeated indices.

    Args:
        numbers: Multiplicity of each distinct degree of freedom.

    Returns:
        Symmetry factor.

    Examples:
        >>> get_symmetry_factor(1, 1)
        1.0
        >>> get_symmetry_factor(2, 2)
        0.25
        >>> get_symmetry_factor(3, 2, 1)
        0.125
    """
    ntot = 0
    for n in numbers:
        ntot += max(0, n - 1)
    return 1.0 / (2.0**ntot)


def symmetry_factor(subscript: str) -> float:
    """Get the symmetry factor for a given subscript.

    Args:
        subscript: Subscript to get the symmetry factor for.

    Returns:
        Symmetry factor.
    """
    counts = {char: subscript.count(char) for char in set(subscript)}
    return get_symmetry_factor(*counts.values())


def antisymmetrise_array(v: NDArray[T], axes: Optional[tuple[int, ...]] = None) -> NDArray[T]:
    """Antisymmetrise an array.

    Args:
        v: Array to antisymmetrise.
        axes: Axes to antisymmetrise over.

    Returns:
        Antisymmetrised array.
    """
    if axes is None:
        axes = tuple(range(v.ndim))
    v_as = np.zeros(v.shape, dtype=v.dtype)

    for perm, sign in permutations_with_signs(axes):
        transpose = list(range(v.ndim))
        for i, ax in enumerate(transpose):
            if ax in axes:
                j = axes.index(ax)
                transpose[i] = perm[j]
        v_as += np.copy(np.transpose(v, transpose)) * sign

    return v_as


def is_mixed_spin(spin: Iterable[Hashable]) -> bool:
    """Return a boolean indicating if a list of spins is mixed."""
    return len(set(spin)) != 1


def combine_subscripts(
    *subscripts: str,
    sizes: Optional[dict[tuple[str, ...], int]] = None,
) -> tuple[str, dict[str, int]]:
    """Combine subscripts into new unique subscripts for functions such as `compress_axes`.

    For example, one may wish to compress an amplitude according to both
    occupancy and spin signatures.

    The output string of this function has the same length as the input
    subscripts, where the `i`th character is an arbitrary character chosen
    such that it is unique for a unique value of
    `tuple(s[i] for s in subscripts)` among other values of `i`.

    This function also returns a dictionary indicating the size of each new character in the
    subscript according to the size of the corresponding original character in the dictionary
    `sizes`.

    Args:
        subscripts: Subscripts to combine.
        sizes: Dictionary of sizes for each index.

    Returns:
        New subscript, with a dictionary of sizes of each new index.
    """
    if len(set(len(s) for s in subscripts)) != 1:
        raise ValueError("Subscripts must be of the same length.")

    char_map: dict[tuple[str, ...], str] = {}
    new_subscript = ""
    new_sizes: dict[str, int] = {}
    j = 0
    for i in range(len(subscripts[0])):
        key = tuple(s[i] for s in subscripts)
        if key not in char_map:
            if j == 91:
                raise ValueError("Too many unique characters.")
            char_map[key] = chr(97 + j)
            j += 1
            if j == 123:
                j = 65
        new_subscript += char_map[key]
        if sizes:
            new_sizes[char_map[key]] = sizes[key]

    return new_subscript, new_sizes


def compress_axes(
    subscript: str, array: NDArray[T], include_diagonal: Optional[bool] = False
) -> NDArray[T]:
    """Compress an array into lower-triangular representations using an einsum-like input.

    Args:
        subscript: Subscript for the input array.
        array: Array to compress.
        include_diagonal: Whether to include the diagonal elements of the input array in the output
            array.

    Returns:
        Compressed array.

    Examples:
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
    arg = tuple(util.argsort(list(subscript)))
    array = np.transpose(array, arg)
    subscript = permute_string(subscript, arg)

    # Reshape array so that all axes of the same character are flattened:
    sizes: dict[str, int] = {}
    for char, n in zip(subscript, array.shape):
        if char in sizes:
            assert sizes[char] == n
        else:
            sizes[char] = n
    array = np.reshape(
        array, [sizes[char] ** subscript.count(char) for char in sorted(set(subscript))]
    )

    # For each axis type, get the necessary lower-triangular indices:
    indices_ndim = [
        tril_indices_ndim(sizes[char], subscript.count(char), include_diagonal=include_diagonal)
        for char in sorted(set(subscript))
    ]
    indices = [
        np.ravel_multi_index(ind, (sizes[char],) * subscript.count(char))
        for ind, char in zip(indices_ndim, sorted(set(subscript)))
    ]

    # Apply the indices:
    indices = [
        ind[tuple(None if i != j else slice(None) for i in range(len(indices)))]
        for j, ind in enumerate(indices)
    ]
    array_flat: NDArray[T] = array[tuple(indices)]

    return array_flat


def decompress_axes(
    subscript: str,
    array_flat: NDArray[T],
    shape: Optional[tuple[int, ...]] = None,
    include_diagonal: Optional[bool] = False,
    symmetry: Optional[str] = None,
    out: Optional[NDArray[T]] = None,
) -> NDArray[T]:
    """Reverse operation of `compress_axes`, subscript input is the same.

    One of `shape` or `out` must be passed.

    Args:
        subscript: Subscript for the output array.
        array_flat: Array to decompress.
        shape: Shape of the output array. Must be passed if `out` is `None`.
        include_diagonal: Whether to include the diagonal elements of the output array in the input
            array.
        symmetry: Symmetry of the output array, with a "+" indicating symmetry and "-" indicating
            antisymmetry for each dimension in the decompressed array.
        out: Output array. If `None`, a new array is created, and `shape` must be passed.

    Returns:
        Decompressed array.
    """

    assert "->" not in subscript
    assert shape is not None or out is not None

    # Get symmetry string if needed:
    if symmetry is None:
        symmetry = "-" * len(subscript)

    # Initialise decompressed array
    if out is None:
        if shape is None:
            raise ValueError("One of `shape` or `out` must be passed.")
        array = np.zeros(shape, dtype=array_flat.dtype)
    else:
        array = out
        out[:] = 0.0

    # Substitute the input characters so that they are ordered:
    subs = {}
    i = 0
    for char in subscript:
        if char not in subs:
            subs[char] = chr(97 + i)
            i += 1
    subscript = "".join([subs[s] for s in subscript])

    # Reshape array so that all axes of the same character are adjacent:
    arg = tuple(util.argsort(list(subscript)))
    array = np.transpose(array, arg)
    subscript = permute_string(subscript, arg)

    # Reshape array so that all axes of the same character are flattened:
    sizes: dict[str, int] = {}
    for char, n in zip(subscript, array.shape):
        if char in sizes:
            assert sizes[char] == n
        else:
            sizes[char] = n
    array = np.reshape(
        array, [sizes[char] ** subscript.count(char) for char in sorted(set(subscript))]
    )

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
        signs = tuple(s if symm == "-" else 1 for s, symm in zip(signs, symmetry_compressed))

        # Apply the indices:
        indices_perm = tuple(
            np.ravel_multi_index(ind, (sizes[char],) * subscript.count(char))
            for ind, char in zip(indices_perm, sorted(set(subscript)))
        )
        indices_perm = tuple(
            ind[tuple(None if i != j else slice(None) for i in range(len(indices_perm)))]
            for j, ind in enumerate(indices_perm)
        )
        array = _put(array, indices_perm, array_flat * util.prod(signs))

    # Reshape array to non-flattened format
    array = np.reshape(
        array,
        (sum([(sizes[char],) * subscript.count(char) for char in sorted(set(subscript))], tuple())),
    )

    # Undo transpose:
    arg = tuple(util.argsort(list(arg)))
    array = np.transpose(array, arg)

    return array


def get_compressed_size(subscript: str, **sizes: int) -> int:
    """Get the size of a compressed representation of a matrix based on the subscript input.

    Args:
        subscript: Subscript for the output array. See `compressed_axes` for details.
        **sizes: Sizes of each character in the subscript.

    Returns:
        Size of the compressed representation of the array.

    Examples:
        >>> get_compressed_size("iiaa", i=5, a=3)
        30
    """
    n = 1
    for char in set(subscript):
        dims = subscript.count(char)
        n *= ntril_ndim(sizes[char], dims)
    return n


def symmetrise(
    subscript: str,
    array: NDArray[T],
    symmetry: Optional[str] = None,
    apply_factor: Optional[bool] = True,
) -> NDArray[T]:
    """Enforce a symmetry in an array.

    Args:
        subscript: Subscript for the input array.
        array: Array to symmetrise.
        symmetry: Symmetry of the output array, with a "+" indicating symmetry and "-" indicating
            antisymmetry for each dimension in the decompressed array.
        apply_factor: Whether to apply a factor to the output array, to account for the symmetry.

    Returns:
        Symmetrised array.
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
    array_as = np.zeros(array.shape, dtype=array.dtype)
    groups = tuple(sorted(set(zip(sorted(set(subscript)), symmetry_compressed))))  # don't ask
    inds = [tuple(i for i, s in enumerate(subscript) if s == char) for char, symm in groups]
    for tup in itertools.product(*(permutations_with_signs(ind) for ind in inds)):
        perms, signs = zip(*tup)
        perm = list(range(len(subscript)))
        for inds_part, perms_part in zip(inds, perms):
            for i, p in zip(inds_part, perms_part):
                perm[i] = p
        sign = util.prod(signs) if symmetry[perm[0]] == "-" else 1
        array_as = array_as + np.transpose(array, perm) * sign

    if apply_factor:
        # Apply factor
        sizes = [subscript.count(s) for s in sorted(set(subscript))]
        array_as = array_as * get_symmetry_factor(*sizes)

    return array_as


def pack_2e(*args):  # type: ignore  # noqa
    # args should be in the order of ov_2e
    # TODO remove

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

    assert len(args) == 16
    blocks = [[[[None, None] for _ in range(2)] for _ in range(2)] for _ in range(2)]

    for n in range(16):
        i, j, k, l = ["ov".index(x) for x in ov_2e[n]]
        blocks[i][j][k][l] = args[n]

    return np.block(blocks)  # type: ignore


def unique(lst: list[Hashable]) -> list[Hashable]:
    """Get unique elements of a list."""
    done = set()
    out = []
    for el in lst:
        if el not in done:
            out.append(el)
            done.add(el)
    return out
