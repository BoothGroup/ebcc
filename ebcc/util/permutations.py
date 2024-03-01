"""Symmetry and permutational utilities."""

import functools
import itertools

import numpy as np


def factorial(n):
    """Return the factorial of `n`."""
    if n in (0, 1):
        return 1
    else:
        return n * factorial(n - 1)


def permute_string(string, permutation):
    """
    Permute a string.

    Parameters
    ----------
    string : str
        String to permute.
    permutation : list of int
        Permutation to apply.

    Returns
    -------
    permuted : str
        Permuted string.

    Examples
    --------
    >>> permute_string("abcd", [2, 0, 3, 1])
    "cbda"
    """
    return "".join([string[i] for i in permutation])


def tril_indices_ndim(n, dims, include_diagonal=False):
    """
    Return lower triangular indices for a multidimensional array.

    Parameters
    ----------
    n : int
        Size of each dimension.
    dims : int
        Number of dimensions.
    include_diagonal : bool, optional
        If `True`, include diagonal elements. Default value is `False`.

    Returns
    -------
    tril : tuple of ndarray
        Lower triangular indices for each dimension.
    """

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


def generate_spin_combinations(n, excited=False, unique=False):
    """
    Generate combinations of spin components for a given number of
    occupied and virtual axes.

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
    """
    Return permutations of seq, yielding also a sign which is equal to +1
    for an even number of swaps, and -1 for an odd number of swaps.

    Parameters
    ----------
    seq : iterable
        Sequence to permute.

    Returns
    -------
    permuted : list of tuple
        List of tuples of the form (permuted, sign).
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
    """
    Get a floating point value corresponding to the factor from the
    neglection of symmetry in repeated indices.

    Parameters
    ----------
    numbers : tuple of int
        Multiplicity of each distinct degree of freedom.

    Returns
    -------
    factor : float
        Symmetry factor.

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


def antisymmetrise_array(v, axes=(0, 1)):
    """
    Antisymmetrise an array.

    Parameters
    ----------
    v : ndarray
        Array to antisymmetrise.
    axes : tuple of int, optional
        Axes to antisymmetrise over. Default value is `(0, 1)`.

    Returns
    -------
    v_as : ndarray
        Antisymmetrised array.
    """

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
    """Return a boolean indicating if a list of spins is mixed."""
    return len(set(spin)) != 1


def combine_subscripts(*subscripts, sizes=None):
    """
    Combine subscripts into new unique subscripts for functions such as
    `compress_axes`.

    For example, one may wish to compress an amplitude according to both
    occupancy and spin signatures.

    The output string of this function has the same length as the input
    subscripts, where the `i`th character is an arbitrary character chosen
    such that it is unique for a unique value of
    `tuple(s[i] for s in subscripts)` among other values of `i`.

    If `sizes` is passed, this function also returns a dictionary
    indicating the size of each new character in the subscript according to
    the size of the corresponding original character in the dictionary
    `sizes`.

    Parameters
    ----------
    subscripts : tuple of str
        Subscripts to combine. Each subscript must be a string of the same
        length.
    sizes : dict, optional
        Dictionary of sizes for each index. Keys should be
        `tuple(s[i] for s in subscripts)`. Default value is `None`.

    Returns
    -------
    new_subscript : str
        Output subscript.
    new_sizes : dict, optional
        Dictionary of the sizes of each new index. Only returned if the
        `sizes` keyword argument is provided.
    """

    if len(set(len(s) for s in subscripts)) != 1:
        raise ValueError("Subscripts must be of the same length.")

    char_map = {}
    new_subscript = ""
    new_sizes = {}
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
        if sizes is not None:
            new_sizes[char_map[key]] = sizes[key]

    if sizes is None:
        return new_subscript
    else:
        return new_subscript, new_sizes


def compress_axes(subscript, array, include_diagonal=False):
    """
    Compress an array into lower-triangular representations using an
    einsum-like input.

    Parameters
    ----------
    subscript : str
        Subscript for the input array. The output array will have a
        compressed representation of the input array according to this
        subscript, where repeated characters indicate symmetrically
        equivalent axes.
    array : numpy.ndarray
        Array to compress.
    include_diagonal : bool, optional
        Whether to include the diagonal elements of the input array in the
        output array. Default value is `False`.

    Returns
    -------
    compressed_array : numpy.ndarray
        Compressed array.

    Examples
    --------
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
    subscript,
    array_flat,
    shape=None,
    include_diagonal=False,
    symmetry=None,
    out=None,
):
    """
    Reverse operation of `compress_axes`, subscript input is the same. One
    of `shape` or `out` must be passed.

    Parameters
    ----------
    subscript : str
        Subscript for the output array. The input array will have a
        compressed representation of the output array according to this
        subscript, where repeated characters indicate symmetrically
        equivalent axes.
    array_flat : numpy.ndarray
        Array to decompress.
    shape : tuple of int, optional
        Shape of the output array. Must be passed if `out` is `None`.
        Default value is `None`.
    include_diagonal : bool, optional
        Whether to include the diagonal elements of the output array in the
        input array. Default value is `False`.
    symmetry : str, optional
        Symmetry of the output array, with a `"+"` indicating symmetry and
        `"-"` indicating antisymmetry for each dimension in the
        decompressed array. If `None`, defaults to fully antisymmetric
        (i.e. all characters are `"-"`). Default value is `None`.
    out : numpy.ndarray, optional
        Output array. If `None`, a new array is created, and `shape` must
        be passed. Default value is `None`.
    """

    assert "->" not in subscript
    assert shape is not None or out is not None

    # Get symmetry string if needed:
    if symmetry is None:
        symmetry = "-" * len(subscript)

    # Initialise decompressed array
    if out is None:
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
    """
    Get the size of a compressed representation of a matrix based on the
    subscript input to `compressed_axes` and the sizes of each character.

    Parameters
    ----------
    subscript : str
        Subscript for the output array. See `compressed_axes` for details.
    **sizes : int
        Sizes of each character in the subscript.

    Returns
    -------
    n : int
        Size of the compressed representation of the array.

    Examples
    --------
    >>> get_compressed_shape("iiaa", i=5, a=3)
    30
    """

    n = 1
    for char in set(subscript):
        dims = subscript.count(char)
        n *= ntril_ndim(sizes[char], dims)

    return n


def symmetrise(subscript, array, symmetry=None, apply_factor=True):
    """
    Enforce a symmetry in an array.

    Parameters
    ----------
    subscript : str
        Subscript for the input array. The output array will have a
        compressed representation of the input array according to this
        subscript, where repeated characters indicate symmetrically
        equivalent axes.
    array : numpy.ndarray
        Array to compress.
    symmetry : str, optional
        Symmetry of the output array, with a `"+"` indicating symmetry and
        `"-"` indicating antisymmetry for each dimension in the
        decompressed array. If `None`, defaults to fully symmetric (i.e.
        all characters are `"+"`). Default value is `None`.
    apply_factor : bool, optional
        Whether to apply a factor of to the output array, to account for
        the symmetry. Default value is `True`.

    Returns
    -------
    array : numpy.ndarray
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


def pack_2e(*args):  # noqa
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


def unique(lst):
    """Get unique elements of a list."""

    done = set()
    out = []
    for el in lst:
        if el not in done:
            out.append(el)
            done.add(el)

    return out


def mask_to_slice(mask):
    """
    Convert a boolean mask to a slice. If not possible, then an
    exception is raised.
    """

    if isinstance(mask, slice):
        return mask

    differences = np.diff(np.where(mask > 0)[0])

    if np.any(differences != 1):
        raise ValueError

    return slice(min(differences), max(differences) + 1)
