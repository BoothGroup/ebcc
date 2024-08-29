"""Tiled arrays with MPI support."""

from __future__ import annotations

import itertools
import functools
from typing import TYPE_CHECKING

from ebcc import numpy as np
from ebcc.core.precision import types
from ebcc.util.permutations import sorted_with_signs

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


if TYPE_CHECKING:
    from typing import Type, TypeVar, Optional, Iterator, Union, Any, Callable

    from ebcc.numpy.typing import NDArray

    T = TypeVar("T", float, complex)
    Permutation = tuple[tuple[int, ...], int]


def loop_block_indices(block_num: tuple[int]) -> Iterator[tuple[int]]:
    """Loop over the block indices.

    Args:
        block_num: The number of blocks in each dimension.

    Returns:
        The block indices.
    """
    return itertools.product(*map(range, block_num))


def symmetrise_indices(
    indices: tuple[int],
    permutations: Optional[tuple[Permutation]],
    forward: bool = True,
) -> tuple[int]:
    """Symmetrise indices to get the canonical block index.

    Args:
        indices: The indices.
        permutations: The permutations of the tensor. Should be a list of tuples where the first
            element is the permutation and the second element is the sign.
        forward: If `True`, return the forward permutation. If `False`, return the backward
            permutation.

    Returns:
        The canonical block index, the permutation required to canonicalise the block, and the sign
        of the permutation.
    """
    if not permutations:
        return indices, tuple(range(len(indices))), 1

    best = None
    for perm, sign in permutations:
        candidate = (tuple(indices[i] for i in perm), perm, sign)
        if best is None or candidate < best:
            best = candidate

    if not forward:
        perm = tuple(np.argsort(best[1]))
        best = (tuple(indices[i] for i in perm), perm, best[2])

    return best


def _combine_permutations(
    subscript: str, permutations1: tuple[Permutation], permutations2: tuple[Permutation]
) -> tuple[Permutation]:
    """Get the permutations for the result of a contraction.

    Args:
        subscript: The contraction subscript.
        permutations1: The permutations of the first tensor.
        permutations2: The permutations of the second tensor.

    Returns:
        The permutations of the result.
    """
    ab, c = subscript.split("->")
    a, b = ab.split(",")
    perms_and_signs1 = {perm: sign for perm, sign in permutations1}
    perms_and_signs2 = {perm: sign for perm, sign in permutations2}

    # Generate all possible permutations of output indices
    permutations = set()
    for perm in itertools.permutations(range(len(c))):
        c_p = [c[i] for i in perm]
        perm1 = tuple(a.index(c_p[i]) if c_p[i] in a else b.index(c_p[i]) for i in range(len(c)))
        perm2 = tuple(b.index(c_p[i]) if c_p[i] in b else a.index(c_p[i]) for i in range(len(c)))
        if perm1 in perms_and_signs1 and perm2 in perms_and_signs2:
            sign = perms_and_signs1[perm1] * perms_and_signs2[perm2]
            permutations.add((tuple(perm), sign))

    return tuple(sorted(permutations))


def _check_shapes(first: Tensor, second: Tensor) -> None:
    """Check that the shapes of two tiled arrays are compatible.

    Args:
        first: The first tiled array.
        second: The second tiled array.
    """
    if first.shape != second.shape:
        raise ValueError("Shapes must be equal.")
    if first.permutations != second.permutations:
        raise ValueError("Permutations must be equal.")


def _bilinear_map(
    func: Callable[[NDArray[float], NDArray[float]], NDArray[float]],
    first: Tensor,
    second: Tensor,
) -> Tensor:
    """Bilinear map between two tiled arrays.

    Applies a function to each pair of blocks, handling communication. The function should be a
    bilinear map of the matrix elements, i.e. `result[i, j] = func(first[i, j], second[i, j])`.

    Args:
        func: The function to apply.
        first: The first tiled array.
        second: The second tiled array.

    Returns:
        The result of the bilinear map.
    """
    _check_shapes(first, second)
    result = Tensor(
        first.shape,
        first.block_num,
        permutations=first.permutations,
        world=first.world,
    )

    # Loop over the blocks
    for block_index in loop_block_indices(first.block_num):
        if first.owns_block(block_index):
            if second.owns_block(block_index):
                block = func(first._get_block(block_index), second._get_block(block_index))
            else:
                other_block = second.recv_block(block_index, second.get_owner(block_index), store=False)
                block = func(first._get_block(block_index), other_block)
            result._set_block(block_index, block)

    return result


def _unary_map(func: Callable[[NDArray[float]], NDArray[float]], array: Tensor) -> Tensor:
    """Unary map on a tiled array.

    Args:
        func: The function to apply.
        array: The tiled array.

    Returns:
        The result of the unary map.
    """
    result = Tensor(
        array.shape,
        array.block_num,
        permutations=array.permutations,
        world=array.world,
    )

    # Loop over the blocks
    for block_index in loop_block_indices(array.block_num):
        if array.owns_block(block_index):
            block = func(array._get_block(block_index))
            result._set_block(block_index, block)

    return result


class Tensor:
    """Tensor class."""

    def __init__(
        self,
        shape: tuple[int],
        block_num: Optional[tuple[int]] = None,
        dtype: Optional[Type[T]] = None,
        permutations: Optional[tuple[Permutation]] = None,
        world: Optional[MPI.Comm] = None,
    ) -> None:
        """Initialise the tensor.

        Args:
            shape: The shape of the tensor.
            dtype: The data type of the tensor.
            block_num: The number of blocks in each dimension.
            permutations: The permutations of the tensor. Should be a list of tuples where the first
                element is the permutation and the second element is the sign.
            world: The MPI communicator.
        """
        if world is None:
            world = MPI.COMM_WORLD if MPI is not None else None
        if dtype is None:
            dtype = types[float]

        self.shape = shape
        self.dtype = dtype
        self.block_num = block_num
        self.permutations = permutations
        self.world = world

        self._blocks: dict[tuple[int], NDArray[T]] = {}

    def _set_block(self, index: tuple[int], block: NDArray[T]) -> None:
        """Set a block in the distributed array."""
        self._blocks[index] = block

    def __setitem__(self, index: tuple[int], block: NDArray[T]) -> None:
        """Set a block in the distributed array.

        Args:
            index: The index of the block.
            block: The block.
        """
        if any(i >= n for i, n in zip(index, self.block_num)):
            raise ValueError(
                "Block index out of range. The `__setitem__` method sets blocks, not elements."
            )
        index, perm, sign = symmetrise_indices(index, self.permutations, forward=False)
        block = block.transpose(perm) * sign
        if not (block.flags.c_contiguous or block.flags.f_contiguous):
            block = np.copy(block, order="C")
        self._set_block(index, block)

    def _get_block(self, index: tuple[int]) -> NDArray[T]:
        """Get a block from the distributed array."""
        return self._blocks[index]

    def __getitem__(self, index: tuple[int]) -> NDArray[T]:
        """Get a block from the distributed array.

        Args:
            index: The index of the block.

        Returns:
            The block.
        """
        if any(i >= n for i, n in zip(index, self.block_num)):
            raise ValueError(
                "Block index out of range. The `__getitem__` method gets blocks, not elements."
            )
        try:
            canon, perm, sign = symmetrise_indices(index, self.permutations)
            return self._get_block(canon).transpose(perm) * sign
        except KeyError:
            raise ValueError(f"Block {index} not found.")

    @functools.lru_cache
    def get_block_shape(self, index: tuple[int]) -> tuple[int]:
        """Get the shape of a block.

        Args:
            index: The index of the block.

        Returns:
            The shape of the block.
        """
        return tuple(
            s // bs + (1 if i < s % bs else 0)
            for i, bs, s in zip(index, self.block_num, self.shape)
        )

    @functools.lru_cache
    def get_block_slice(self, index: tuple[int]) -> tuple[slice]:
        """Get the slice of a block.

        Args:
            index: The index of the block.

        Returns:
            The slice of the block.
        """
        _shape = lambda i, bs, s: s // bs + (1 if i < s % bs else 0)
        return tuple(
            slice(sum(_shape(j, bs, s) for j in range(i)), sum(_shape(j, bs, s) for j in range(i + 1)))
            for i, bs, s in zip(index, self.block_num, self.shape)
        )

    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return len(self.shape)

    def owns_block(self, index: tuple[int]) -> bool:
        """Check if the current process owns a block.

        Args:
            index: The index of the block.

        Returns:
            Whether the current process owns the block.
        """
        index, _, _ = symmetrise_indices(index, self.permutations)
        return index in self._blocks

    def get_owner(self, index: tuple[int], raise_error: bool = True) -> Optional[int]:
        """Get the owner of a block.

        Args:
            index: The index of the block.
            raise_error: Whether to raise an error if the block is not owned.

        Returns:
            The rank of the owner.
        """
        if self.world is None or self.world.size == 1:
            return 0 if self.owns_block(index) else None
        check = np.zeros(self.world.size, dtype=np.int32)
        if self.owns_block(index):
            check[self.world.rank] = 1
        check = self.world.allreduce(check, op=MPI.SUM)
        rank = np.argmax(check)
        if check[rank] != 1:
            rank = None
        if raise_error and rank is None:
            raise ValueError(f"Block {index} is not owned by any process.")
        return rank

    def send_block(self, index: tuple[int], dest: int) -> None:
        """Send a block to another process.

        Args:
            index: The index of the block.
            dest: The rank of the destination process.
        """
        if self.world is None or self.world.size == 1:
            return
        self.world.Send(self[index], dest=dest)

    def recv_block(self, index: tuple[int], source: int, store: bool = True) -> NDArray[T]:
        """Receive a block from another process.

        Args:
            index: The index of the block.
            source: The rank of the source process.
            store: Whether to store the block.

        Returns:
            The block.
        """
        index, perm, sign = symmetrise_indices(index, self.permutations)
        if self.world is None or self.world.size == 1:
            return self[index].transpose(perm) * sign
        block = np.empty(self.get_block_shape(index), dtype=self.dtype)
        self.world.Recv(block, source=source)
        if store:
            self[index] = block
        return block.transpose(perm) * sign

    def as_local_ndarray(self, fill=0.0) -> NDArray[T]:
        """Convert the tiled array to a local numpy array.

        Args:
            fill: The value to fill the array with where blocks are not owned.

        Returns:
            The local numpy array.
        """
        array = np.full(shape=self.shape, dtype=self.dtype, fill_value=fill)
        for block_index in loop_block_indices(self.block_num):
            if self.owns_block(block_index):
                s = self.get_block_slice(block_index)
                array[s] = self[block_index]
        return array

    def as_global_ndarray(self) -> NDArray[T]:
        """Convert the tiled array to a global numpy array.

        Returns:
            The global numpy array.
        """
        array = np.empty(self.shape, dtype=self.dtype)
        for block_index in loop_block_indices(self.block_num):
            owner = self.get_owner(block_index)
            s = self.get_block_slice(block_index)
            if self.owns_block(block_index):
                array[s] = self[block_index]
                for i in range(self.world.size):
                    if i != self.world.rank:
                        self.send_block(block_index, i)
            else:
                array[s] = self.recv_block(block_index, owner, store=False)
        return array

    def __add__(self, other: Tensor) -> Tensor:
        """Distributed addition.

        Args:
            other: The other tiled array.

        Returns:
            The sum of the two tiled arrays.
        """
        return _bilinear_map(np.add, self, other)

    def __sub__(self, other: Tensor) -> Tensor:
        """Distributed subtraction.

        Args:
            other: The other tiled array.

        Returns:
            The difference of the two tiled arrays.
        """
        return _bilinear_map(np.subtract, self, other)

    def __mul__(self, other: Tensor) -> Tensor:
        """Distributed multiplication.

        Args:
            other: The other tiled array.

        Returns:
            The product of the two tiled arrays.
        """
        return _bilinear_map(np.multiply, self, other)

    def __truediv__(self, other: Tensor) -> Tensor:
        """Distributed division.

        Args:
            other: The other tiled array.

        Returns:
            The quotient of the two tiled arrays.
        """
        return _bilinear_map(np.divide, self, other)

    def __neg__(self) -> Tensor:
        """Negation.

        Returns:
            The negated tiled array.
        """
        return _unary_map(np.negative, self)

    def __abs__(self) -> Tensor:
        """Absolute value.

        Returns:
            The absolute tiled array.
        """
        return _unary_map(np.abs, self)

    def __pow__(self, other: Union[int, float, Tensor]) -> Tensor:
        """Distributed exponentiation.

        Args:
            other: The other tiled array.

        Returns:
            The exponentiated tiled array.
        """
        if isinstance(other, Tensor):
            return _bilinear_map(np.power, self, other)
        else:
            return _unary_map(lambda x: np.power(x, other), self)

    def __matmul__(self, other: Tensor) -> Tensor:
        """Distributed matrix multiplication.

        Args:
            other: The other tiled array.

        Returns:
            The matrix product of the two tiled arrays.
        """
        return dot(self, other)

    def __iadd__(self, other: Tensor) -> Tensor:
        """In-place addition.

        Args:
            other: The other tiled array.

        Returns:
            The sum of the two tiled arrays.
        """
        return _bilinear_map(np.add, self, other)

    def __isub__(self, other: Tensor) -> Tensor:
        """In-place subtraction.

        Args:
            other: The other tiled array.

        Returns:
            The difference of the two tiled arrays.
        """
        return _bilinear_map(np.subtract, self, other)

    def __imul__(self, other: Tensor) -> Tensor:
        """In-place multiplication.

        Args:
            other: The other tiled array.

        Returns:
            The product of the two tiled arrays.
        """
        return _bilinear_map(np.multiply, self, other)

    def copy(self) -> Tensor:
        """Copy the tiled array.

        Returns:
            The copy of the tiled array.
        """
        return _unary_map(np.copy, self)

    def transpose(self, *axes: int) -> Tensor:
        """Transpose the tiled array.

        Args:
            axes: The axes to transpose.

        Returns:
            The transposed tiled array.
        """
        if axes == tuple(range(self.ndim)):
            return self
        res = _unary_map(lambda x: x.transpose(*axes), self)
        res.shape = tuple(res.shape[i] for i in axes)
        res.block_num = tuple(res.block_num[i] for i in axes)
        res._blocks = {tuple(i[j] for j in axes): block for i, block in res._blocks.items()}
        res.permutations = tuple(
            (tuple(perm[i] for i in axes), sign) for perm, sign in res.permutations
        )
        return res

    def swapaxes(self, axis1: int, axis2: int) -> Tensor:
        """Swap the axes of the tiled array.

        Args:
            axis1: The first axis.
            axis2: The second axis.

        Returns:
            The tiled array with the axes swapped.
        """
        transpose = list(range(self.ndim))
        transpose[axis1], transpose[axis2] = transpose[axis2], transpose[axis1]
        return self.transpose(*transpose)


def dot(
    first: Tensor,
    second: Tensor,
) -> Tensor:
    """Distributed dot product.

    Args:
        first: The first tiled array.
        second: The second tiled array.

    Returns:
        The dot product of the two tiled arrays.
    """
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("Arrays must be 2D.")
    if first.shape[1] != second.shape[0]:
        raise ValueError("Shapes must be compatible.")
    if first.block_num[1] != second.block_num[0]:
        raise ValueError("Block shapes must be compatible.")

    world = first.world
    world_size = world.size if world is not None else 1
    world_rank = world.rank if world is not None else 0

    result = Tensor(
        (first.shape[0], second.shape[1]),
        dtype=first.dtype,
        block_num=first.block_num,
        permutations=_combine_permutations("ij,jk->ik", first.permutations, second.permutations),
        world=world,
    )

    block_num = tuple(first.block_num) + (second.block_num[1],)
    rank = 0
    for i, j, k in itertools.product(*[range(n) for n in block_num]):
        ij = (i, j)
        ik = (i, k)
        kj = (k, j)

        # Check if the computation has already been done under a different permutation
        ij_canon, _, _ = symmetrise_indices(ij, result.permutations)
        if ij != ij_canon:
            continue

        # Get the owners
        owner_ij = result.get_owner(ij, raise_error=False)  # In case a permutation exists
        if owner_ij is None:
            owner_ij = rank
            rank = (rank + 1) % world_size
        owner_ik = first.get_owner(ik)
        owner_kj = second.get_owner(kj)

        # Send [ik] to owner_ij
        if owner_ik == world_rank:
            first.send_block(ik, owner_ij)
        if owner_ij == world_rank:
            block_ik = first.recv_block(ik, owner_ik, store=False)

        # Send [kj] to owner_ij
        if owner_kj == world_rank:
            second.send_block(kj, owner_ij)
        if owner_ij == world_rank:
            block_kj = second.recv_block(kj, owner_kj, store=False)

        # Multiply [ik] @ [kj] and add to [ij]
        if owner_ij == world_rank:
            block_ij = block_ik @ block_kj
            if result.owns_block(ij):
                result[ij] += block_ij
            else:
                result[ij] = block_ij

    return result


def tensordot(
    a: Tensor,
    b: Tensor,
    axes: Union[int, tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]],
    subscript: Optional[str] = None,
) -> Tensor:
    """Compute a generalized tensor dot product over the specified axes.

    See `numpy.tensordot` for more information on the `axes` argument.

    Args:
        a: The first tiled array.
        b: The second tiled array.
        axes: The axes to sum over.
        subscript: The subscript notation for the contraction. Must be provided to preserve
            permutations.

    Returns:
        The tensor dot product of the two tiled arrays.
    """
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes

    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1

    try:
        nb = len(axes_a)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    axes_a = [a.ndim + axis if axis < 0 else axis for axis in axes_a]
    axes_b = [b.ndim + axis if axis < 0 else axis for axis in axes_b]

    def _combined_block_indices(
        block_indices_external: tuple[int, ...],
        block_indices_dummy: tuple[int, ...],
        axes_external: tuple[int, ...],
        axes_dummy: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Combine the external and dummy block indices."""
        block_indices = [None] * (len(axes_external) + len(axes_dummy))
        for axes, block in zip(axes_external, block_indices_external):
            block_indices[axes] = block
        for axes, block in zip(axes_dummy, block_indices_dummy):
            block_indices[axes] = block
        return tuple(block_indices)

    # Get the dummy and external axes
    axes_dummy = (axes_a, axes_b)
    axes_external = (
        [i for i in range(a.ndim) if i not in axes_a],
        [i for i in range(b.ndim) if i not in axes_b],
    )

    # Get the output array
    output_shape = [a.shape[i] for i in axes_external[0]]
    output_shape += [b.shape[i] for i in axes_external[1]]
    world = a.world
    world_rank = world.rank if world is not None else 0
    c = Tensor(
        output_shape,
        dtype=a.dtype,
        block_num=a.block_num,
        permutations=(
            _combine_permutations(subscript, a.permutations, b.permutations)
            if subscript is not None
            else None
        ),
        world=world,
    )

    for block_indices_external in loop_block_indices(a.block_num):
        for block_indices_dummy in itertools.product(
            *map(range, [a.block_num[i] for i in axes_a])
        ):
            block_indices_a = _combined_block_indices(
                block_indices_external[: len(axes_external[0])],
                block_indices_dummy,
                axes_external[0],
                axes_dummy[0],
            )
            block_indices_b = _combined_block_indices(
                block_indices_external[len(axes_external[0]) :],
                block_indices_dummy,
                axes_external[1],
                axes_dummy[1],
            )
            block_indices_c = block_indices_external

            # Check if the computation has already been done under a different permutation
            block_indices_c_canon, _, _ = symmetrise_indices(block_indices_c, c.permutations)
            if block_indices_c != block_indices_c_canon:
                continue

            # Get the owners
            owner_a = a.get_owner(block_indices_a)
            owner_b = b.get_owner(block_indices_b)
            owner_c = np.ravel_multi_index(block_indices_c, c.block_num) % c.world.size

            # Send [a] to [b]
            if owner_a == world_rank:
                a.send_block(block_indices_a, owner_c)
            if owner_c == world_rank:
                block_a = a.recv_block(block_indices_a, owner_a, store=False)

            # Send [b] to [a]
            if owner_b == world_rank:
                b.send_block(block_indices_b, owner_c)
            if owner_c == world_rank:
                block_b = b.recv_block(block_indices_b, owner_b, store=False)

            # Multiply [a] @ [b] and add to [c]
            if owner_c == world_rank:
                block_c = block_a @ block_b
                if c.owns_block(block_indices_c):
                    c[block_indices_external] += block_c
                else:
                    c[block_indices_external] = block_c

    return c


def contract(*operands: Any, **kwargs: Any) -> Union[T, NDArray[T]]:
    """Dispatch a two-term einsum contraction.
    """
    subscript, a, b = operands  # FIXME
    abk, rk = subscript.split("->")
    ak, bk = abk.split(",")
    rk_def = ""

    # Sum over any axes that are not in the output for the first array
    for i, aki in enumerate(ak):
        if aki not in bk and aki not in rk:
            a = np.sum(a, axis=i)
            ak = ak[:i] + ak[i+1:]

    # Sum over any axes that are not in the output for the second array
    for i, bki in enumerate(bk):
        if bki not in ak and bki not in rk:
            b = np.sum(b, axis=i)
            bk = bk[:i] + bk[i+1:]

    # Get the axes for the first array
    axes_a = []
    for i,k in enumerate(ak):
        if k in rk:
            if k not in rk_def:
                rk_def += k
        else:
            axes_a.append(i)

    # Get the axes for the second array
    axes_b = []
    for i,k in enumerate(bk):
        if k in rk:
            if k not in rk_def:
                rk_def += k
        else:
            axes_b.append(i)

    # Must be the same length
    if len(axes_a) != len(axes_b):
        raise ValueError(f"Could not dispatch \"{ak},{bk}->{rk}\"")

    # Check the transpose for the second array
    k_sum_a = "".join([ak[x] for x in axes_a])
    k_sum_b = "".join([bk[x] for x in axes_b])
    if k_sum_a != k_sum_b:
        perm = np.argsort([k_sum_a.index(k) for k in k_sum_b])
        axes_b = [axes_b[x] for x in perm]

    # Dispatch the contraction
    axes = (tuple(axes_a[::-1]), tuple(axes_b[::-1])) # reverse necessary?
    res = tensordot(a, b, axes=axes, subscript=subscript)

    # Transpose the result
    if rk != rk_def:
        perm = [rk_def.index(k) for k in rk]
        res = res.transpose(*perm)

    return res


def einsum(*operands: Any, **kwargs: Any) -> Union[T, NDArray[T]]:
    """Evaluate an Einstein summation convention on the operands.

    Using the Einstein summation convention, many common
    multi-dimensional, linear algebraic array operations can be
    represented in a simple fashion. In *implicit* mode `einsum`
    computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical
    Einstein summation operations, by disabling, or forcing summation
    over specified subscript labels.

    See the `numpy.einsum` documentation for clarification.

    Args:
        operands: Any valid input to `numpy.einsum`.
        alpha: Scaling factor for the contraction.
        beta: Scaling factor for the output.
        out: If provided, the calculation is done into this array.
        contract: The function to use for contraction.
        optimize: If `True`, use the `numpy.einsum_path` to optimize the contraction.

    Returns:
        The calculation based on the Einstein summation convention.

    Notes:
        This function may use `numpy.einsum`, `pyscf.lib.einsum`, or `tblis_einsum` as a backend,
        depending on the problem size and the modules available.
    """
    # Parse the kwargs
    subscript = operands[0]  # FIXME
    args = operands[1:]
    _contract = kwargs.get("contract", contract)

    # Perform the contraction
    if len(args) < 2:
        # If it's just a transpose, use the fallback
        inp, out = subscript.split("->")
        transpose = [inp.index(x) for x in out]
        out = args[0].transpose(*transpose)
    elif len(args) < 3:
        # If it's a single contraction, call the backend directly
        out = _contract(subscript, *args, **kwargs)
    else:
        # If it's a chain of contractions, use the path optimizer
        optimize = kwargs.pop("optimize", True)
        args = list(args)
        path_kwargs = dict(optimize=optimize, einsum_call=True)
        contractions = np.einsum_path(subscript, *args, **path_kwargs)[1]
        for contraction in contractions:
            inds, idx_rm, einsum_str, remain = list(contraction[:4])
            contraction_args = [args.pop(x) for x in inds]
            if kwargs.get("alpha", 1.0) != 1.0 or kwargs.get("beta", 0.0) != 0.0:
                raise NotImplementedError("Scaling factors not supported for >2 arguments")
            out = _contract(einsum_str, *contraction_args, **kwargs)
            args.append(out)

    return out
