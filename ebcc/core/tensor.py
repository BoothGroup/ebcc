"""Tensor class."""

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING, Generic, TypeVar

from ebcc import numpy as np
from ebcc.core.precision import types

if TYPE_CHECKING:
    from typing import Any, Callable, Iterator, Optional, Type, Union
    from unittest.mock import MagicMock

    from ebcc.numpy.typing import NDArray

    Permutation = tuple[tuple[int, ...], int]

    MPI = MagicMock()

    class Comm:
        """Mock MPI communicator."""

        size: int
        rank: int

        def Send(self, buf: Any, dest: Optional[int]) -> None: ...  # noqa: D102, E704
        def Recv(self, buf: Any, source: Optional[int]) -> None: ...  # noqa: D102, E704
        def allreduce(self, buf: Any, op: Callable[[Any, Any], Any]) -> Any: ...  # noqa: D102, E704

else:
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None

T = TypeVar("T", float, complex)

EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
EINSUM_SYMBOLS_SET = set(EINSUM_SYMBOLS)

DEFAULT_BLOCK_SIZE = 16


def loop_block_indices(block_num: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Loop over the block indices.

    Args:
        block_num: The number of blocks in each dimension.

    Yields:
        The block indices.
    """
    return itertools.product(*map(range, block_num))


def loop_rank_block_indices(tensor: Tensor[T]) -> Iterator[tuple[int, ...]]:
    """Loop over the block indices for a given rank.

    Includes consideration of symmetry, keeping symmetrically equivalent blocks on the same rank.

    Args:
        tensor: The tensor.

    Yields:
        The block indices.
    """
    done = set()
    rank = 0
    for i, indices in enumerate(loop_block_indices(tensor.block_num)):
        canon, _, _ = symmetrise_indices(indices, tensor.permutations)
        if canon in done:
            continue
        if rank == tensor.world.rank:
            yield indices
        done.add(canon)
        rank = (rank + 1) % tensor.world.size


def set_blocks_from_array(tensor: Tensor[T], array: NDArray[T]) -> Tensor[T]:
    """Set the blocks of a tensor from a numpy array.

    Args:
        tensor: The tensor.
        NDArray: The numpy array.

    Returns:
        The tensor with the blocks set.

    Notes:
        The blocks are set in-place.
    """
    for block_index in loop_rank_block_indices(tensor):
        s = tensor.get_block_slice(block_index)
        tensor[block_index] = array[s]
    return tensor


def initialise_from_array(array: NDArray[T], **kwargs: Any) -> Tensor[T]:
    """Initialise a tensor from a numpy array.

    Args:
        array: The numpy array.
        **kwargs: Additional arguments to pass to the tensor constructor.

    Returns:
        The tensor.
    """
    tensor = Tensor(array.shape, **kwargs)
    set_blocks_from_array(tensor, array)
    return tensor


def zeros(shape: tuple[int, ...], **kwargs: Any) -> Tensor[T]:
    """Create a tensor of zeros.

    Args:
        shape: The shape of the tensor.
        **kwargs: Additional arguments to pass to the tensor constructor.

    Returns:
        The tensor of zeros.
    """
    return Tensor(shape, **kwargs)


def symmetrise_indices(
    indices: tuple[int, ...],
    permutations: Optional[list[Permutation, ...]],
    forward: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
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

    assert best is not None

    if not forward:
        perm = tuple(np.argsort(best[1]))
        best = (tuple(indices[i] for i in perm), perm, best[2])

    return best


def _combine_permutations(
    subscript: str, permutations1: list[Permutation, ...], permutations2: list[Permutation, ...]
) -> list[Permutation, ...]:
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

    # Generate all possible permutations of output indices and check if they are valid
    permutations = set()
    for perm in itertools.permutations(range(len(c))):
        # Permute characters in a
        swaps_a = {c[i]: c[j] for i, j in enumerate(perm) if c[i] in a}
        if set(swaps_a.keys()) != set(swaps_a.values()):
            continue
        a_p = "".join([swaps_a.get(a[i], a[i]) for i in range(len(a))])

        # Permute characters in b
        swaps_b = {c[i]: c[j] for i, j in enumerate(perm) if c[i] in b}
        if set(swaps_b.keys()) != set(swaps_b.values()):
            continue
        b_p = "".join([swaps_b.get(b[i], b[i]) for i in range(len(b))])

        # Get the permutations of a and b
        perm1 = tuple(a.index(a_p[i]) for i in range(len(a)))
        perm2 = tuple(b.index(b_p[i]) for i in range(len(b)))
        if perm1 in perms_and_signs1 and perm2 in perms_and_signs2:
            sign = perms_and_signs1[perm1] * perms_and_signs2[perm2]
            permutations.add((tuple(perm), sign))

    return sorted(permutations)


def _check_shapes(first: Tensor[T], second: Tensor[T]) -> None:
    """Check that the shapes of two tensors are compatible.

    Args:
        first: The first tensor.
        second: The second tensor.
    """
    if first.shape != second.shape:
        raise ValueError("Shapes must be equal.")


def _bilinear_map(
    func: Callable[[NDArray[float], NDArray[float]], NDArray[float]],
    first: Tensor[T],
    second: Tensor[T],
) -> Tensor[T]:
    """Bilinear map between two tensors.

    Applies a function to each pair of blocks, handling communication. The function should be a
    bilinear map of the matrix elements, i.e. `result[i, j] = func(first[i, j], second[i, j])`.

    Args:
        func: The function to apply.
        first: The first tensor.
        second: The second tensor.

    Returns:
        The result of the bilinear map.
    """
    _check_shapes(first, second)
    permutations = list(set(first.permutations or ()) & set(second.permutations or ()))
    result: Tensor[T] = Tensor(
        first.shape,
        first.block_num,
        permutations=permutations,
        world=first.world,
        dtype=first.dtype,
    )

    # Loop over the blocks
    for block_index in loop_block_indices(first.block_num):
        if first.owns_block(block_index):
            if second.owns_block(block_index):
                block = func(first._get_block(block_index), second._get_block(block_index))
            else:
                other_block = second.recv_block(
                    block_index, second.get_owner(block_index), store=False
                )
                block = func(first._get_block(block_index), other_block)
            result._set_block(block_index, block)

    return result


def _unary_map(func: Callable[[NDArray[float]], NDArray[float]], array: Tensor[T]) -> Tensor[T]:
    """Unary map on a tensor.

    Args:
        func: The function to apply.
        array: The tensor.

    Returns:
        The result of the unary map.
    """
    result: Tensor[T] = Tensor(
        array.shape,
        array.block_num,
        permutations=array.permutations,
        world=array.world,
        dtype=array.dtype,
    )

    # Loop over the blocks
    for block_index in loop_rank_block_indices(array):
        block = func(array._get_block(block_index))
        result._set_block(block_index, block)

    return result


class Tensor(Generic[T]):
    """Tensor class.

    Tensors are distributed arrays that are tiled into blocks. Each block is assigned to a process
    when MPI is enabled. The blocks are stored in a dictionary where the key is the block index and
    the value is the block. The blocks are stored in a C-contiguous order.

    When symmetry is present, the blocks are stored in a canonical form. The canonical form is
    determined by the permutation that sorts the indices in ascending order. The sign of the
    permutation is stored to account for the sign of the permutation.
    """

    dtype: Type[T]

    def __init__(
        self,
        shape: tuple[int, ...],
        block_num: Optional[tuple[int, ...]] = None,
        dtype: Optional[Type[T]] = None,
        permutations: Optional[list[Permutation, ...]] = None,
        world: Optional[Comm] = None,
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
        self.block_num = (
            block_num if block_num is not None else tuple(s // DEFAULT_BLOCK_SIZE for s in shape)
        )
        self.permutations = permutations
        self.world = world

        self._blocks: dict[tuple[int, ...], NDArray[T]] = {}

    def _set_block(self, index: tuple[int, ...], block: NDArray[T]) -> None:
        """Set a block in the distributed array."""
        self._blocks[index] = block

    def __setitem__(self, index: tuple[int, ...], block: NDArray[T]) -> None:
        """Set a block in the distributed array.

        Args:
            index: The index of the block.
            block: The block.
        """
        if any(i >= n for i, n in zip(index, self.block_num)):
            raise ValueError(
                "Block index out of range. The `__setitem__` method sets blocks, not elements."
            )
        assert block.shape == self.get_block_shape(index)
        index, perm, sign = symmetrise_indices(index, self.permutations, forward=False)
        block = block.transpose(perm) * sign
        if not (block.flags.c_contiguous or block.flags.f_contiguous):
            block = np.copy(block, order="C")
        self._set_block(index, block)

    def _get_block(self, index: tuple[int, ...]) -> NDArray[T]:
        """Get a block from the distributed array."""
        return self._blocks[index]

    def __getitem__(self, index: tuple[int, ...]) -> NDArray[T]:
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
    def get_block_shape(self, index: tuple[int, ...]) -> tuple[int, ...]:
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
    def get_block_slice(self, index: tuple[int, ...]) -> tuple[slice, ...]:
        """Get the slice of a block.

        Args:
            index: The index of the block.

        Returns:
            The slice of the block.
        """

        def _shape(i: int, bs: int, s: int) -> int:
            return s // bs + (1 if i < s % bs else 0)

        return tuple(
            slice(
                sum(_shape(j, bs, s) for j in range(i)), sum(_shape(j, bs, s) for j in range(i + 1))
            )
            for i, bs, s in zip(index, self.block_num, self.shape)
        )

    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return len(self.shape)

    def owns_block(self, index: tuple[int, ...]) -> bool:
        """Check if the current process owns a block.

        Args:
            index: The index of the block.

        Returns:
            Whether the current process owns the block.
        """
        index, _, _ = symmetrise_indices(index, self.permutations)
        return index in self._blocks

    def get_owner(self, index: tuple[int, ...], raise_error: bool = True) -> Optional[int]:
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
        rank: Optional[int] = int(np.argmax(check))
        if check[rank] != 1:
            rank = None
        if raise_error and rank is None:
            raise ValueError(f"Block {index} is not owned by any process.")
        return rank

    def send_block(self, index: tuple[int, ...], dest: Optional[int]) -> None:
        """Send a block to another process.

        Args:
            index: The index of the block.
            dest: The rank of the destination process. Must be provided if MPI is used.
        """
        if dest is None and self.world is not None:
            raise ValueError("Destination process must be provided if MPI is used.")
        index, perm, sign = symmetrise_indices(index, self.permutations)
        if self.world is None or self.world.size == 1:
            return
        self.world.Send(self[index], dest=dest)

    def recv_block(
        self, index: tuple[int, ...], source: Optional[int], store: bool = True
    ) -> NDArray[T]:
        """Receive a block from another process.

        Args:
            index: The index of the block.
            source: The rank of the source process. Must be provided if MPI is used.
            store: Whether to store the block.

        Returns:
            The block.
        """
        if source is None and self.world is not None:
            raise ValueError("Source process must be provided if MPI is used.")
        index, perm, sign = symmetrise_indices(index, self.permutations)
        if self.world is None or self.world.size == 1:
            return self[index].transpose(perm) * sign
        block = np.empty(self.get_block_shape(index), dtype=self.dtype)
        self.world.Recv(block, source=source)
        if store:
            self[index] = block
        return block.transpose(perm) * sign

    def as_local_ndarray(self, fill: Any = 0.0) -> NDArray[T]:
        """Convert the tensor to a local numpy array.

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
        """Convert the tensor to a global numpy array.

        Returns:
            The global numpy array.
        """
        array = self.as_local_ndarray()
        if self.world is None or self.world.size == 1:
            return array
        return self.world.allreduce(array, op=MPI.SUM)

    def __add__(self, other: Union[Tensor[T], T]) -> Tensor[T]:
        """Distributed addition.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The sum of the two tensors.
        """
        if isinstance(other, Tensor):
            return _bilinear_map(np.add, self, other)
        return _unary_map(lambda x: np.add(x, other), self)

    def __sub__(self, other: Union[Tensor[T], T]) -> Tensor[T]:
        """Distributed subtraction.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The difference of the two tensors.
        """
        if isinstance(other, Tensor):
            return _bilinear_map(np.subtract, self, other)
        return _unary_map(lambda x: np.subtract(x, other), self)

    def __mul__(self, other: Union[Tensor[T], T]) -> Tensor[T]:
        """Distributed multiplication.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The product of the two tensors.
        """
        if isinstance(other, Tensor):
            return _bilinear_map(np.multiply, self, other)
        return _unary_map(lambda x: np.multiply(x, other), self)

    def __truediv__(self, other: Union[Tensor[T], T]) -> Tensor[T]:
        """Distributed division.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The quotient of the two tensors.
        """
        if isinstance(other, Tensor):
            return _bilinear_map(np.divide, self, other)
        return _unary_map(lambda x: np.divide(x, other), self)

    def __neg__(self) -> Tensor[T]:
        """Negation.

        Returns:
            The negated tensor.
        """
        return _unary_map(np.negative, self)

    def __abs__(self) -> Tensor[T]:
        """Absolute value.

        Returns:
            The absolute tensor.
        """
        return _unary_map(np.abs, self)

    def __pow__(self, other: Union[Tensor[T], T]) -> Tensor[T]:
        """Distributed exponentiation.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The exponentiated tensor.
        """
        if isinstance(other, Tensor):
            return _bilinear_map(np.power, self, other)
        return _unary_map(lambda x: np.power(x, other), self)

    def __matmul__(self, other: Tensor[T]) -> Tensor[T]:
        """Distributed matrix multiplication.

        Args:
            other: The other tensor.

        Returns:
            The matrix product of the two tensors.
        """
        return dot(self, other)

    def __iadd__(self, other: Union[Tensor[T], T]) -> Tensor[T]:
        """In-place addition.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The sum of the two tensors.
        """
        if isinstance(other, Tensor):
            return _bilinear_map(np.add, self, other)
        return _unary_map(lambda x: np.add(x, other), self)

    def __isub__(self, other: Union[Tensor[T], T]) -> Tensor[T]:
        """In-place subtraction.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The difference of the two tensors.
        """
        if isinstance(other, Tensor):
            return _bilinear_map(np.subtract, self, other)
        return _unary_map(lambda x: np.subtract(x, other), self)

    def __imul__(self, other: Union[Tensor[T], T]) -> Tensor[T]:
        """In-place multiplication.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The product of the two tensors.
        """
        if isinstance(other, Tensor):
            return _bilinear_map(np.multiply, self, other)
        return _unary_map(lambda x: np.multiply(x, other), self)

    def __array__(self, dtype: Optional[Type[T]] = None) -> NDArray[T]:
        """Convert the tensor to a numpy array.

        Args:
            dtype: The data type of the array.

        Returns:
            The array.
        """
        return self.as_global_ndarray().astype(dtype)

    def copy(self) -> Tensor[T]:
        """Copy the tensor.

        Returns:
            The copy of the tensor.
        """
        return _unary_map(np.copy, self)

    def astype(self, dtype: Type[T]) -> Tensor[T]:
        """Change the data type of the tensor.

        Args:
            dtype: The data type.

        Returns:
            The tensor with the new data type.
        """
        return _unary_map(lambda x: x.astype(dtype), self)

    def transpose(self, *axes: Union[int, tuple[int, ...]]) -> Tensor[T]:
        """Transpose the tensor.

        Args:
            axes: The axes to transpose.

        Returns:
            The transposed tensor.
        """
        if isinstance(axes[0], tuple):
            if len(axes) > 1:
                raise ValueError("Only one tuple of axes can be provided.")
            axes = axes[0]
        if axes == tuple(range(self.ndim)):
            return self

        # Transpose the blocks
        res = _unary_map(lambda x: x.transpose(*axes), self)
        res.shape = tuple(res.shape[i] for i in axes)
        res.block_num = tuple(res.block_num[i] for i in axes)
        res._blocks = {tuple(i[j] for j in axes): block for i, block in res._blocks.items()}

        # Transpose the permutations
        if res.permutations is not None:
            chars = [EINSUM_SYMBOLS[i] for i in range(len(axes))]
            chars_transpose = [chars[i] for i in axes]
            permutations = []
            for perm, sign in res.permutations:
                chars_perm = [chars_transpose[i] for i in perm]
                perm = tuple(chars.index(c) for c in chars_perm)
                perm = tuple(perm[i] for i in axes)
                permutations.append((perm, sign))
            res.permutations = permutations

        # Canonicalise the blocks
        blocks = {}
        for index, block in res._blocks.items():
            canon, perm, sign = symmetrise_indices(index, res.permutations)
            blocks[canon] = block.transpose(perm) * sign
        res._blocks = blocks

        return res

    def T(self) -> Tensor[T]:
        """Transpose the tensor.

        Returns:
            The transposed tensor.
        """
        return self.transpose(*range(self.ndim)[::-1])

    def swapaxes(self, axis1: int, axis2: int) -> Tensor[T]:
        """Swap the axes of the tensor.

        Args:
            axis1: The first axis.
            axis2: The second axis.

        Returns:
            The tensor with the axes swapped.
        """
        transpose = list(range(self.ndim))
        transpose[axis1], transpose[axis2] = transpose[axis2], transpose[axis1]
        return self.transpose(*transpose)

    def ravel(self) -> NDArray[T]:
        """Ravel the tensor.

        Returns:
            The ravelled array.
        """
        return self.as_local_ndarray().ravel()  # FIXME


def dot(
    first: Tensor[T],
    second: Tensor[T],
) -> Tensor[T]:
    """Distributed dot product.

    Args:
        first: The first tensor.
        second: The second tensor.

    Returns:
        The dot product of the two tensors.
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

    permutations = None
    if first.permutations is not None and second.permutations is not None:
        permutations = _combine_permutations("ij,jk->ik", first.permutations, second.permutations)

    result: Tensor[T] = Tensor(
        (first.shape[0], second.shape[1]),
        dtype=first.dtype,
        block_num=first.block_num,
        permutations=permutations,
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
    a: Tensor[T],
    b: Tensor[T],
    axes: Union[int, tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]],
    subscript: Optional[str] = None,
    permutations: Optional[list[Permutation, ...]] = None,
) -> Tensor[T]:
    """Compute a generalized tensor dot product over the specified axes.

    See `numpy.tensordot` for more information on the `axes` argument.

    Args:
        a: The first tensor.
        b: The second tensor.
        axes: The axes to sum over.
        subscript: The subscript notation for the contraction. Must be provided to preserve
            permutations, unless the permutations are provided explicitly.
        permutations: The permutations of the result. If not provided, the permutations are
            inferred from the subscript.

    Returns:
        The tensor dot product of the two tensors.
    """
    # Parse the axes
    if isinstance(axes, int):
        axes_a = tuple(range(-axes, 0))
        axes_b = tuple(range(0, axes))
    else:
        axes_a = (axes[0],) if isinstance(axes[0], int) else tuple(axes[0])
        axes_b = (axes[1],) if isinstance(axes[1], int) else tuple(axes[1])
    axes_a = tuple(a.ndim + axis if axis < 0 else axis for axis in axes_a)
    axes_b = tuple(b.ndim + axis if axis < 0 else axis for axis in axes_b)

    def _combined_block_indices(
        block_indices_external: tuple[int, ...],
        block_indices_dummy: tuple[int, ...],
        axes_external: tuple[int, ...],
        axes_dummy: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Combine the external and dummy block indices."""
        block_indices = [-1] * (len(axes_external) + len(axes_dummy))
        for axes, block in zip(axes_external, block_indices_external):
            block_indices[axes] = block
        for axes, block in zip(axes_dummy, block_indices_dummy):
            block_indices[axes] = block
        return tuple(block_indices)

    # Get the dummy and external axes
    axes_dummy = (axes_a, axes_b)
    axes_external = (
        tuple(i for i in range(a.ndim) if i not in axes_a),
        tuple(i for i in range(b.ndim) if i not in axes_b),
    )

    # Get the output permutations
    if permutations is None and subscript is not None:
        if a.permutations is None or b.permutations is None:
            raise ValueError(
                "Input tensors must have permutations to determine the output permutations."
            )
        permutations = _combine_permutations(subscript, a.permutations, b.permutations)

    # Get the output array
    output_shape = tuple(a.shape[i] for i in axes_external[0])
    output_shape += tuple(b.shape[i] for i in axes_external[1])
    block_num_dummy = tuple(a.block_num[i] for i in axes_dummy[0])
    if block_num_dummy != tuple(b.block_num[i] for i in axes_dummy[1]):
        raise ValueError("Block shapes must be compatible.")
    block_num_external = tuple(a.block_num[i] for i in axes_external[0]) + tuple(
        b.block_num[i] for i in axes_external[1]
    )
    world = a.world
    world_rank = world.rank if world is not None else 0
    world_size = world.size if world is not None else 1
    c: Tensor[T] = Tensor(
        output_shape,
        dtype=a.dtype,
        block_num=block_num_external,
        permutations=permutations,
        world=world,
    )

    rank = 0
    for block_indices_external in loop_block_indices(block_num_external):
        for block_indices_dummy in loop_block_indices(block_num_dummy):
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
            owner_c = c.get_owner(block_indices_c, raise_error=False)  # In case permutation exists
            if owner_c is None:
                owner_c = rank
                rank = (rank + 1) % world_size

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
                block_c = np.tensordot(block_a, block_b, axes=(axes_a, axes_b))
                if c.owns_block(block_indices_c):
                    c[block_indices_c] += block_c
                else:
                    c[block_indices_c] = block_c

    return c


def _contract(*operands: Any, **kwargs: Any) -> Union[T, NDArray[T]]:
    """Dispatch a two-term einsum contraction."""
    subscript, a, b = operands  # FIXME
    abk, rk = subscript.split("->")
    ak, bk = abk.split(",")
    rk_def = ""

    # Sum over any axes that are not in the output for the first array
    for i, aki in enumerate(ak):
        if aki not in bk and aki not in rk:
            a = np.sum(a, axis=i)
            ak = ak[:i] + ak[i + 1 :]

    # Sum over any axes that are not in the output for the second array
    for i, bki in enumerate(bk):
        if bki not in ak and bki not in rk:
            b = np.sum(b, axis=i)
            bk = bk[:i] + bk[i + 1 :]

    # Get the axes for the first array
    axes_a = []
    for i, k in enumerate(ak):
        if k in rk:
            if k not in rk_def:
                rk_def += k
        else:
            axes_a.append(i)

    # Get the axes for the second array
    axes_b = []
    for i, k in enumerate(bk):
        if k in rk:
            if k not in rk_def:
                rk_def += k
        else:
            axes_b.append(i)

    # Must be the same length
    if len(axes_a) != len(axes_b):
        raise ValueError(f'Could not dispatch "{ak},{bk}->{rk}"')

    # Check the transpose for the second array
    k_sum_a = "".join([ak[x] for x in axes_a])
    k_sum_b = "".join([bk[x] for x in axes_b])
    if k_sum_a != k_sum_b:
        perm = list(np.argsort([k_sum_a.index(k) for k in k_sum_b]))
        axes_b = [axes_b[x] for x in perm]

    # Get the output permutations
    permutations = kwargs.get("permutations", None)
    if permutations is not None and rk != rk_def:
        perm = [rk_def.index(k) for k in rk]
        permutations = [(tuple(perm[i] for i in p), sign) for p, sign in permutations]

    # Dispatch the contraction
    axes = (tuple(axes_a[::-1]), tuple(axes_b[::-1]))  # reverse necessary?
    res = tensordot(a, b, axes=axes, subscript=subscript, permutations=permutations)

    # Transpose the result
    if rk != rk_def:
        perm_res = [rk_def.index(k) for k in rk]
        res = res.transpose(*perm_res)

    return res


def _parse_einsum_input(operands: list[Any]) -> tuple[str, str, list[Tensor[T]]]:
    """Parse the input for an einsum contraction.

    Args:
        operands: The input operands.

    Returns:
        The parsed input.

    Notes:
        This function is a modified version of `numpy.core.einsumfunc._parse_einsum_input`. The
        modifications are necessary to prevent `numpy` from converting the `Tensor` objects to
        `ndarray` objects.
    """
    if len(operands) == 0:
        raise ValueError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = operands[1:]

        # Ensure all characters are valid
        for s in subscripts:
            if s in ".,->":
                continue
            if s not in EINSUM_SYMBOLS:
                raise ValueError(f"Character {s} is not a valid symbol.")

    else:
        tmp_operands = list(operands)
        operand_list: list[Tensor[T]] = []
        subscript_list: list[tuple[int, ...]] = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list: Optional[tuple[int, ...]] = tmp_operands[-1] if len(tmp_operands) else None
        operands = operand_list
        subscripts = ""
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for i in sub:
                subscripts += EINSUM_SYMBOLS[i]
            if num != last:
                subscripts += ","

        if output_list is not None:
            subscripts += "->"
            for i in output_list:
                subscripts += EINSUM_SYMBOLS[i]

    # Check for proper "->"
    if ("-" in subscripts) or (">" in subscripts):
        invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
        if invalid or (subscripts.count("->") != 1):
            raise ValueError("Subscripts can only contain one '->'.")

    # Parse ellipses
    if "." in subscripts:
        used = subscripts.replace(".", "").replace(",", "").replace("->", "")
        unused = list(EINSUM_SYMBOLS_SET - set(used))
        ellipse_inds = "".join(unused)
        longest = 0

        if "->" in subscripts:
            input_tmp, output_sub = subscripts.split("->")
            split_subscripts = input_tmp.split(",")
            out_sub = True
        else:
            split_subscripts = subscripts.split(",")
            out_sub = False

        for num, ssub in enumerate(split_subscripts):
            if "." in ssub:
                if (ssub.count(".") != 3) or (ssub.count("...") != 1):
                    raise ValueError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= len(ssub) - 3

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError("Ellipses lengths do not match.")
                elif ellipse_count == 0:
                    split_subscripts[num] = ssub.replace("...", "")
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = ssub.replace("...", rep_inds)

        subscripts = ",".join(split_subscripts)
        if longest == 0:
            out_ellipse = ""
        else:
            out_ellipse = ellipse_inds[-longest:]

        if out_sub:
            subscripts += "->" + output_sub.replace("...", out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ""
            tmp_subscripts = subscripts.replace(",", "")
            for s in sorted(set(tmp_subscripts)):
                if s not in (EINSUM_SYMBOLS):
                    raise ValueError(f"Character {s} is not a valid symbol.")
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = "".join(sorted(set(output_subscript) - set(out_ellipse)))

            subscripts += "->" + out_ellipse + normal_inds

    # Build output string if does not exist
    if "->" in subscripts:
        input_subscripts, output_subscript = subscripts.split("->")
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(",", "")
        output_subscript = ""
        for s in sorted(set(tmp_subscripts)):
            if s not in EINSUM_SYMBOLS:
                raise ValueError(f"Character {s} is not a valid symbol.")
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if output_subscript.count(char) != 1:
            raise ValueError(f"Output character {s} appeared more than once in the output.")
        if char not in input_subscripts:
            raise ValueError(f"Output character {s} did not appear in the input")

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(",")) != len(operands):
        raise ValueError("Number of einsum subscripts must be equal to the number of operands.")

    return (input_subscripts, output_subscript, operands)


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
        permutations: The permutations of the output tensor. If not provided, the permutations
            are inferred from the input tensors.

    Returns:
        The calculation based on the Einstein summation convention.

    Notes:
        This function may use `numpy.einsum`, `pyscf.lib.einsum`, or `tblis_einsum` as a backend,
        depending on the problem size and the modules available.
    """
    # Parse the operands
    inp, out, args = _parse_einsum_input(list(operands))
    subscript = inp + "->" + out

    # Parse the kwargs
    contract = kwargs.get("contract", _contract)
    optimize = kwargs.get("optimize", True)

    # Perform the contraction
    if len(args) < 2:
        # If it's just a transpose, use the fallback
        transpose = [inp.index(x) for x in out]
        res = args[0].transpose(*transpose)
    elif len(args) < 3:
        # If it's a single contraction, call the backend directly
        res = contract(subscript, *args, **kwargs)
    else:
        raise NotImplementedError("More than two arguments not yet supported")

        # If it's a chain of contractions, use the path optimizer
        contractions = np.einsum_path(subscript, *args, optimize=optimize, einsum_call=True)[1]
        for contraction in contractions:
            inds, idx_rm, einsum_str, remain = list(contraction[:4])
            contraction_args = [args.pop(x) for x in inds]
            if kwargs.get("alpha", 1.0) != 1.0 or kwargs.get("beta", 0.0) != 0.0:
                raise NotImplementedError("Scaling factors not supported for >2 arguments")
            res = contract(einsum_str, *contraction_args, **kwargs)
            args.append(res)

    return res
