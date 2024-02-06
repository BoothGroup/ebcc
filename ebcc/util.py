"""Utilities."""

import ctypes
import functools
import itertools
import time

from pyscf.lib import direct_sum, dot  # noqa: F401
from pyscf.lib import einsum as pyscf_einsum

from ebcc import numpy as np

try:
    try:
        import tblis_einsum
    except ImportError:
        from pyscf.tblis_einsum import tblis_einsum
    FOUND_TBLIS = True
except ImportError:
    FOUND_TBLIS = False

NUMPY_EINSUM_SIZE = 2000


class InheritedType:
    """Type for an inherited variable in `Options` classes."""

    pass


Inherited = InheritedType()


class ModelNotImplemented(NotImplementedError):
    """Error for unsupported models."""

    pass


class Namespace:
    """
    Replacement for SimpleNamespace, which does not trivially allow
    conversion to a dict for heterogenously nested objects.

    Attributes can be added and removed, using either string indexing or
    accessing the attribute directly.
    """

    def __init__(self, **kwargs):
        self.__dict__["_keys"] = set()
        for key, val in kwargs.items():
            self[key] = val

    def __setitem__(self, key, val):
        """Set an attribute."""
        self._keys.add(key)
        self.__dict__[key] = val

    def __getitem__(self, key):
        """Get an attribute."""
        if key not in self._keys:
            raise IndexError(key)
        return self.__dict__[key]

    def __delitem__(self, key):
        """Delete an attribute."""
        if key not in self._keys:
            raise IndexError(key)
        del self.__dict__[key]

    __setattr__ = __setitem__

    def __iter__(self):
        """Iterate over the namespace as a dictionary."""
        yield from {key: self[key] for key in self._keys}

    def __eq__(self, other):
        """Check equality."""
        return dict(self) == dict(other)

    def __ne__(self, other):
        """Check inequality."""
        return dict(self) != dict(other)

    def __contains__(self, key):
        """Check if an attribute exists."""
        return key in self._keys

    def __len__(self):
        """Return the number of attributes."""
        return len(self._keys)

    def keys(self):
        """Return keys of the namespace as a dictionary."""
        return {k: None for k in self._keys}.keys()

    def values(self):
        """Return values of the namespace as a dictionary."""
        return dict(self).values()

    def items(self):
        """Return items of the namespace as a dictionary."""
        return dict(self).items()

    def get(self, *args, **kwargs):
        """Get an item of the namespace as a dictionary."""
        return dict(self).get(*args, **kwargs)


class Timer:
    """Timer class."""

    def __init__(self):
        self.t_init = time.perf_counter()
        self.t_prev = time.perf_counter()
        self.t_curr = time.perf_counter()

    def lap(self):
        """Return the time since the last call to `lap`."""
        self.t_prev, self.t_curr = self.t_curr, time.perf_counter()
        return self.t_curr - self.t_prev

    __call__ = lap

    def total(self):
        """Return the total time since initialization."""
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


def _mro(*bases):
    """Find the method resolution order of bases using the C3 algorithm."""

    seqs = [list(x.__mro__) for x in bases] + [list(bases)]
    res = []

    while True:
        non_empty = list(filter(None, seqs))
        if not non_empty:
            return tuple(res)

        for seq in non_empty:
            candidate = seq[0]
            not_head = [s for s in non_empty if candidate in s[1:]]
            if not_head:
                candidate = None
            else:
                break

        if not candidate:
            raise TypeError("Inconsistent hierarchy")

        res.append(candidate)

        for seq in non_empty:
            if seq[0] == candidate:
                del seq[0]


class InheritDocstrings(type):
    """
    Metaclass to inherit docstrings from superclasses. All attributes which
    are public (no underscore prefix) are updated with the docstring of the
    first superclass in the MRO containing that attribute with a docstring.

    Additionally checks that all methods are documented at runtime.
    """

    def __new__(cls, name, bases, attrs):
        """Create an instance of the class with inherited docstrings."""

        for key, val in attrs.items():
            if key.startswith("_") or val.__doc__ is not None:
                continue

            for supcls in _mro(*bases):
                supval = getattr(supcls, key, None)
                if supval is None:
                    continue
                val.__doc__ = supval.__doc__
                break
            else:
                raise RuntimeError("Method {} does not exist in superclass".format(key))

            if val.__doc__ is None:
                raise RuntimeError("Could not find docstring for {}".format(key))

            attrs[key] = val

        return super().__new__(cls, name, bases, attrs)


def has_docstring(obj):
    """
    Decorate a function or class to inform a static analyser that it has a
    docstring even if one is not visible, for example via inheritance.
    """
    return obj


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
        decompressed array. If `None`, defaults to fully symmetric (i.e.
        all characters are `"+"`). Default value is `None`.
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


def pack_2e(*args):  # noqa
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


class EinsumOperandError(ValueError):
    """Exception for invalid inputs to `einsum`."""

    pass


def _fallback_einsum(*operands, **kwargs):
    """Handle the fallback to `numpy.einsum`."""

    kwargs = kwargs.copy()
    alpha = kwargs.pop("alpha", 1.0)
    beta = kwargs.pop("beta", 0.0)
    out = kwargs.pop("out", None)

    res = np.einsum(*operands, **kwargs)
    res *= alpha

    if out is not None:
        res += beta * out

    return res


def contract(subscript, *args, **kwargs):
    """
    Contract a pair of terms in an einsum. Supports additional keyword
    arguments `alpha` and `beta` which operate as `pyscf.lib.dot`. In some
    cases this will still require copying, but it minimises the memory
    overhead in simple cases.

    Parameters
    ----------
    subscript : str
        Subscript representation of the contraction.
    args : tuple of numpy.ndarray
        Arrays to contract.
    alpha : float, optional
        Scaling factor for contraction. Default value is `1.0`.
    beta : float, optional
        Scaling factor for the output. Default value is `0.0`.
    out : numpy.ndarray, optional
        Output array. If `None`, a new array is created. Default value is
        `None`.
    **kwargs : dict
        Additional keyword arguments to `numpy.einsum`.

    Returns
    -------
    array : numpy.ndarray
        Result of the contraction.
    """

    alpha = kwargs.get("alpha", 1.0)
    beta = kwargs.get("beta", 0.0)
    buf = kwargs.get("out", None)

    # If this is called for more than 2 arguments, fall back
    if len(args) > 2:
        return _fallback_einsum(subscript, *args, **kwargs)

    # Make sure that the input are numpy arrays
    a, b = args
    a = np.asarray(a)
    b = np.asarray(b)

    # Check if we should use NumPy
    if min(a.size, b.size) < NUMPY_EINSUM_SIZE:
        return _fallback_einsum(subscript, *args, **kwargs)

    # Make sure it can be done via DGEMM
    indices = subscript.replace(",", "").replace("->", "")
    if any(indices.count(x) != 2 for x in set(indices)):
        return _fallback_einsum(subscript, *args, **kwargs)

    # Get the characters for each input and output
    inp, out, args = np.core.einsumfunc._parse_einsum_input((subscript, a, b))
    inp_a, inp_b = inps = inp.split(",")
    assert len(inps) == len(args) == 2
    assert all(len(inp) == arg.ndim for inp, arg in zip(inps, args))

    # If there is an internal trace, consume it:
    if any(len(inp) != len(set(inp)) for inp in inps):
        # FIXME
        return _fallback_einsum(subscript, *args, **kwargs)

    # Find the dummy indices
    dummy = set(inp_a).intersection(set(inp_b))
    if not dummy or inp_a == dummy or inp_b == dummy:
        return _fallback_einsum(subscript, *args, **kwargs)

    # Find the sizes of the indices
    ranges = {}
    for inp, arg in zip(inps, args):
        for i, s in zip(inp, arg.shape):
            if i in ranges:
                if ranges[i] != s:
                    raise EinsumOperandError(
                        "Incompatible shapes for einsum: {} with A={}, B={}".format(
                            subscript, a.shape, b.shape
                        )
                    )
            ranges[i] = s

    if not FOUND_TBLIS:
        # Reorder the indices appropriately
        inp_at = list(inp_a)
        inp_bt = list(inp_b)
        inner_shape = 1
        for i, n in enumerate(sorted(dummy)):
            j = len(inp_at) - 1
            inp_at.insert(j, inp_at.pop(inp_at.index(n)))
            inp_bt.insert(i, inp_bt.pop(inp_bt.index(n)))
            inner_shape *= ranges[n]

        # Find transposes
        order_a = [inp_a.index(idx) for idx in inp_at]
        order_b = [inp_b.index(idx) for idx in inp_bt]

        # Get shape and transpose for the output
        shape_ct = []
        inp_ct = []
        for idx in inp_at:
            if idx in dummy:
                break
            shape_ct.append(ranges[idx])
            inp_ct.append(idx)
        for idx in inp_bt:
            if idx in dummy:
                continue
            shape_ct.append(ranges[idx])
            inp_ct.append(idx)
        order_ct = [inp_ct.index(idx) for idx in out]

        # If any dimension has size zero, return here
        if a.size == 0 or b.size == 0:
            shape_c = [shape_ct[i] for i in order_ct]
            if buf is not None:
                return buf.reshape(shape_c) * beta if beta != 1.0 else buf.reshape(shape_c)
            else:
                return np.zeros(shape_c, dtype=np.result_type(a, b))

        # Apply transposes
        at = a.transpose(order_a)
        bt = b.transpose(order_b)

        # Find the optimal memory alignment
        at = np.asarray(at.reshape(-1, inner_shape), order="F" if at.flags.f_contiguous else "C")
        bt = np.asarray(bt.reshape(inner_shape, -1), order="F" if bt.flags.f_contiguous else "C")

        # Get the output buffer
        if buf is not None:
            shape_ct_flat = (at.shape[0], bt.shape[1])
            order_c = [out.index(idx) for idx in inp_ct]
            buf = buf.transpose(order_c)
            buf = np.asarray(
                buf.reshape(shape_ct_flat), order="F" if buf.flags.f_contiguous else "C"
            )

        # Perform the contraction
        ct = dot(at, bt, alpha=alpha, beta=beta, c=buf)

        # Reshape and transpose
        ct = ct.reshape(shape_ct, order="A")
        c = ct.transpose(order_ct)

    else:
        # Cast the data types
        dtype = np.result_type(a, b, alpha, beta)
        alpha = np.asarray(alpha, dtype=dtype)
        beta = np.asarray(beta, dtype=dtype)
        a = np.asarray(a, dtype=dtype)
        b = np.asarray(b, dtype=dtype)
        tblis_dtype = tblis_einsum.tblis_dtype[dtype]

        # Get the shapes
        shape_a = a.shape
        shape_b = b.shape
        shape_c = tuple(ranges[x] for x in out)

        # If any dimension has size zero, return here
        if a.size == 0 or b.size == 0:
            if buf is not None:
                return buf.reshape(shape_c) * beta
            else:
                return np.zeros(shape_c, dtype=dtype)

        # Get the output buffer
        if buf is None:
            order = kwargs.get("order", "C")
            c = np.empty(shape_c, dtype=dtype, order=order)
        else:
            assert buf.dtype == dtype
            assert buf.size == np.prod(shape_c)
            c = buf.reshape(shape_c)

        # Get the C types
        shape_a = (ctypes.c_size_t * a.ndim)(*shape_a)
        shape_b = (ctypes.c_size_t * b.ndim)(*shape_b)
        shape_c = (ctypes.c_size_t * c.ndim)(*shape_c)
        strides_a = (ctypes.c_size_t * a.ndim)(*[x // dtype.itemsize for x in a.strides])
        strides_b = (ctypes.c_size_t * b.ndim)(*[x // dtype.itemsize for x in b.strides])
        strides_c = (ctypes.c_size_t * c.ndim)(*[x // dtype.itemsize for x in c.strides])

        # Perform the contraction
        tblis_einsum.libtblis.as_einsum(
            a,
            a.ndim,
            shape_a,
            strides_a,
            inp_a.encode("ascii"),
            b,
            b.ndim,
            shape_b,
            strides_b,
            inp_b.encode("ascii"),
            c,
            c.ndim,
            shape_c,
            strides_c,
            out.encode("ascii"),
            tblis_dtype,
            alpha,
            beta,
        )

    return c


def einsum(*operands, **kwargs):
    """Dispatch an einsum. Input arguments are the same as `numpy`."""

    inp, out, args = np.core.einsumfunc._parse_einsum_input(operands)
    subscript = "%s->%s" % (inp, out)

    _contract = kwargs.get("contract", contract)

    if len(args) < 2:
        out = _fallback_einsum(subscript, *args, **kwargs)
    elif len(args) < 3:
        out = _contract(subscript, *args, **kwargs)
    else:
        optimize = kwargs.pop("optimize", True)
        args = list(args)
        contractions = np.einsum_path(subscript, *args, optimize=optimize, einsum_call=True)[1]
        for contraction in contractions:
            inds, idx_rm, einsum_str, remain = contraction[:4]
            operands = [args.pop(x) for x in inds]
            out = _contract(einsum_str, *operands)
            args.append(out)

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


def einsum_symmetric(*operands, symmetries=[], symmetrise_dummies=False, **kwargs):  # noqa
    """Dispatch an einsum in a symmetric representation. The argument
    `symmetries` should be an iterable of the symmetry subscripts for
    each input and for the output array, in the format of the
    `compress_axes` and `decompress_axes` functions. Assumes that the
    phase of the symmetries is positive, i.e. the arrays are not
    antisymmetric.
    """

    assert not symmetrise_dummies

    inp, out, args = np.core.einsumfunc._parse_einsum_input(operands)

    # Get the sizes of each index
    sizes = {}
    for part, arg in zip(inp.split(","), args):
        for p, size in zip(part, arg.shape):
            sizes[p] = size

    # Find the number of dummy indices in each input
    dummies = [[i not in out for i in part] for part in inp.split(",")]

    # Make sure that external and dummy variables are compressed
    # separately
    not_used = [
        char for char in "abcdefghijklmnopqrstuvwxyz" if not any(char in s for s in symmetries)
    ]
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
            for inds in itertools.product(
                *[
                    itertools.combinations_with_replacement(
                        range(args[i].shape[subscripts_flat[i].index(x)]),
                        subscripts_flat[i].count(x),
                    )
                    for x in unique(list(subscripts_flat[i]))
                ]
            ):
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
    out = decompress_axes(
        subscripts_flat[-1], out_flat, include_diagonal=True, shape=shape, symmetry="+" * len(shape)
    )

    return out
