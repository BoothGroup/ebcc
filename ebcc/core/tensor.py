"""Tensor class."""

from __future__ import annotations

import functools
import itertools
import math
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
        """Mock MPI communicator type."""

        size: int
        rank: int

        def Send(self, buf: Any, dest: Optional[int]) -> None: ...  # noqa: D102, E704
        def Recv(self, buf: Any, source: Optional[int]) -> None: ...  # noqa: D102, E704
        def allreduce(self, buf: Any, op: Callable[[Any, Any], Any]) -> Any: ...  # noqa: D102, E704
        def barrier(self) -> None: ...  # noqa: D102, E704

else:
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None

F = TypeVar("F", float, complex)

EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
EINSUM_SYMBOLS_SET = set(EINSUM_SYMBOLS)

DEFAULT_BLOCK_SIZE = 8  # FIXME

DEBUG = False

# TODO: Generate permutations from contraction of identical tensors
# TODO: Redistribute blocks if balance is poor
# TODO: Implement Davidson without needing to ravel the tensor


class TensorError(Exception):
    """Tensor error."""

    def __init__(self, message: str) -> None:
        """Initialise the error.

        Args:
            message: The error message.
        """
        message = f"[Rank {MPI.COMM_WORLD.rank}] {message}"
        super().__init__(message)


def loop_block_indices(block_num: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    """Loop over the block indices.

    Args:
        block_num: The number of blocks in each dimension.

    Yields:
        The block indices.
    """
    return itertools.product(*map(range, block_num))


def loop_rank_block_indices(tensor: Tensor[F]) -> Iterator[tuple[int, ...]]:
    """Loop over the block indices for a given rank.

    Includes consideration of symmetry, keeping symmetrically equivalent blocks on the same rank.

    Args:
        tensor: The tensor.

    Yields:
        The block indices.
    """
    done = set()
    rank = 0
    for indices in loop_block_indices(tensor.block_num):
        canon, _, _ = symmetrise_indices(indices, tensor.permutations)
        if canon in done:
            continue
        if rank == tensor.world.rank:
            yield canon
        done.add(canon)
        rank = (rank + 1) % tensor.world.size


def set_blocks_from_array(tensor: Tensor[F], array: NDArray[F]) -> Tensor[F]:
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


def initialise_from_array(array: NDArray[F], **kwargs: Any) -> Tensor[F]:
    """Initialise a tensor from a numpy array.

    Args:
        array: The numpy array.
        **kwargs: Additional arguments to pass to the tensor constructor.

    Returns:
        The tensor.
    """
    tensor: Tensor[F] = Tensor(array.shape, **kwargs)
    set_blocks_from_array(tensor, array)
    return tensor


def zeros(shape: tuple[int, ...], **kwargs: Any) -> Tensor[F]:
    """Create a tensor of zeros.

    Args:
        shape: The shape of the tensor.
        **kwargs: Additional arguments to pass to the tensor constructor.

    Returns:
        The tensor of zeros.
    """
    tensor: Tensor[F] = Tensor(shape, **kwargs)
    for block_index in loop_rank_block_indices(tensor):
        tensor[block_index] = np.zeros(tensor.get_block_shape(block_index), dtype=tensor.dtype)
    return tensor


def zeros_like(tensor: Tensor[F], **kwargs: Any) -> Tensor[F]:
    """Create a tensor of zeros with the same shape as another tensor.

    Args:
        tensor: The tensor.
        **kwargs: Additional arguments to pass to the tensor constructor.

    Returns:
        The tensor of zeros.
    """
    for key in ("shape", "block_num", "dtype", "permutations", "world"):
        if key not in kwargs:
            kwargs[key] = getattr(tensor, key)
    return zeros(**kwargs)


def eye(size: int, **kwargs: Any) -> Tensor[F]:
    """Create a tensor with ones on the diagonal and zeros elsewhere.

    Args:
        size: The size of the tensor.
        **kwargs: Additional arguments to pass to the tensor constructor.

    Returns:
        The tensor.
    """
    if "permutations" not in kwargs:
        kwargs["permutations"] = [((0, 1), 1), ((1, 0), 1)]
    tensor: Tensor[F] = zeros((size, size), **kwargs)
    for block_index in loop_rank_block_indices(tensor):
        if block_index[0] == block_index[1]:
            tensor[block_index] = np.eye(tensor.get_block_shape(block_index)[0], dtype=tensor.dtype)
    return tensor


def symmetrise_indices(
    indices: tuple[int, ...],
    permutations: Optional[list[Permutation]],
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
    subscript: str, permutations1: list[Permutation], permutations2: list[Permutation]
) -> list[Permutation]:
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
        swaps = {c[i]: c[j] for i, j in enumerate(perm)}
        a_p = "".join([swaps.get(a[i], a[i]) for i in range(len(a))])
        b_p = "".join([swaps.get(b[i], b[i]) for i in range(len(b))])

        if tuple(sorted(a)) != tuple(sorted(a_p)):
            continue
        if tuple(sorted(b)) != tuple(sorted(b_p)):
            continue

        perm1 = tuple(a.index(a_p[i]) for i in range(len(a)))
        perm2 = tuple(b.index(b_p[i]) for i in range(len(b)))

        if perm1 in perms_and_signs1 and perm2 in perms_and_signs2:
            sign = perms_and_signs1[perm1] * perms_and_signs2[perm2]
            permutations.add((tuple(perm), sign))

    return sorted(permutations)


def _check_shapes(first: Tensor[F], second: Tensor[F]) -> None:
    """Check that the shapes of two tensors are compatible.

    Args:
        first: The first tensor.
        second: The second tensor.
    """
    if first.shape != second.shape:
        raise TensorError("Shapes must be equal.")


def _check_not_array(tensor: Union[F, Tensor[F]]) -> None:
    """Check that a tensor is not a `numpy` array.

    Args:
        tensor: The tensor.
    """
    if isinstance(tensor, np.ndarray):
        raise TensorError("Cannot perform operation on a numpy array.")


def _bilinear_map(
    func: Callable[[NDArray[float], NDArray[float]], NDArray[float]],
    a: Tensor[F],
    b: Tensor[F],
) -> Tensor[F]:
    """Bilinear map between two tensors.

    Applies a function to each pair of blocks, handling communication. The function should be a
    bilinear map of the matrix elements, i.e. `result[i, j] = func(first[i, j], second[i, j])`.

    Args:
        func: The function to apply.
        a: The first tensor.
        b: The second tensor.

    Returns:
        The result of the bilinear map.
    """
    if DEBUG:
        a._check_sanity()
        b._check_sanity()
    _check_shapes(a, b)
    permutations = list(set(a.permutations or ()) & set(b.permutations or ()))
    same_permutations_a = set(a.permutations or ()) == set(permutations)
    same_permutations_b = set(b.permutations or ()) == set(permutations)
    c: Tensor[F] = Tensor(
        a.shape,
        a.block_num,
        permutations=permutations,
        world=a.world,
        dtype=a.dtype,
    )

    # Loop over the blocks
    done = set()
    rank = 0
    for block_index in loop_block_indices(c.block_num):
        # Permute the block index
        block_index_c_canon, perm_c, sign_c = symmetrise_indices(block_index, permutations)
        if block_index_c_canon in done:
            continue
        done.add(block_index_c_canon)
        if same_permutations_a:
            block_index_a_canon, perm_a, sign_a = block_index_c_canon, perm_c, sign_c
        else:
            block_index_a_canon, perm_a, sign_a = symmetrise_indices(block_index, a.permutations)
        if same_permutations_b:
            block_index_b_canon, perm_b, sign_b = block_index_c_canon, perm_c, sign_c
        else:
            block_index_b_canon, perm_b, sign_b = symmetrise_indices(block_index, b.permutations)
        rank = (rank + 1) % c.world.size
        c.world.barrier()

        # Get the first block
        owner_a = a.get_owner(block_index_a_canon)
        if owner_a == rank == a.world.rank:
            block_a = a[block_index]
        elif owner_a == a.world.rank:
            a.send_block(block_index_a_canon, rank)
        elif rank == a.world.rank:
            block_a = a.recv_block(block_index_a_canon, owner_a, store=False)
            block_a = block_a.transpose(perm_a) * sign_a
        c.world.barrier()

        # Get the b block
        owner_b = b.get_owner(block_index_b_canon)
        if owner_b == rank == b.world.rank:
            block_b = b[block_index]
        elif owner_b == b.world.rank:
            b.send_block(block_index_b_canon, rank)
        elif rank == b.world.rank:
            block_b = b.recv_block(block_index_b_canon, owner_b, store=False)
            block_b = block_b.transpose(perm_b) * sign_b
        c.world.barrier()

        # Compute the c
        if rank == a.world.rank:
            c[block_index] = func(block_a, block_b)

    if DEBUG:
        c._check_sanity()

    return c


def _unary_map(func: Callable[[NDArray[float]], NDArray[float]], array: Tensor[F]) -> Tensor[F]:
    """Unary map on a tensor.

    Args:
        func: The function to apply.
        array: The tensor.

    Returns:
        The result of the unary map.
    """
    if DEBUG:
        array._check_sanity()
    result: Tensor[F] = Tensor(
        array.shape,
        array.block_num,
        permutations=array.permutations,
        world=array.world,
        dtype=array.dtype,
    )

    # Loop over the blocks
    for block_index, block in array._blocks.items():
        result[block_index] = func(block)
    array.world.barrier()

    if DEBUG:
        result._check_sanity()

    return result


class Tensor(Generic[F]):
    """Tensor class.

    Tensors are distributed arrays that are tiled into blocks. Each block is assigned to a process
    when MPI is enabled. The blocks are stored in a dictionary where the key is the block index and
    the value is the block. The blocks are stored in a C-contiguous order.

    When symmetry is present, the blocks are stored in a canonical form. The canonical form is
    determined by the permutation that sorts the indices in ascending order. The sign of the
    permutation is stored to account for the sign of the permutation.
    """

    dtype: Type[F]

    def __init__(
        self,
        shape: tuple[int, ...],
        block_num: Optional[tuple[int, ...]] = None,
        dtype: Optional[Type[F]] = None,
        permutations: Optional[list[Permutation]] = None,
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
        if not permutations:
            permutations = [(tuple(range(len(shape))), 1)]

        self.shape = shape
        self.dtype = dtype
        self.block_num = (
            block_num
            if block_num is not None
            else tuple(math.ceil(s / DEFAULT_BLOCK_SIZE) for s in shape)
        )
        self.permutations = permutations
        self.world = world

        self._blocks: dict[tuple[int, ...], NDArray[F]] = {}
        if DEBUG:
            self._check_sanity()

    def _set_block(self, index: tuple[int, ...], block: NDArray[F]) -> None:
        """Set a block in the distributed array."""
        self._blocks[index] = block

    def __setitem__(self, index: tuple[int, ...], block: NDArray[F]) -> None:
        """Set a block in the distributed array.

        Args:
            index: The index of the block.
            block: The block.
        """
        if any(i >= n for i, n in zip(index, self.block_num)):
            raise TensorError(
                "Block index out of range. The `__setitem__` method sets blocks, not elements."
            )
        if block.shape != self.get_block_shape(index):
            raise TensorError("Block shape does not match the block shape of the tensor.")
        index, perm, sign = symmetrise_indices(index, self.permutations, forward=False)
        block = block.transpose(perm) * sign
        if not block.flags.c_contiguous:
            block = np.copy(block, order="C")
        assert block.flags.c_contiguous
        self._set_block(index, block)

    def _get_block(self, index: tuple[int, ...]) -> NDArray[F]:
        """Get a block from the distributed array."""
        return self._blocks[index]

    def __getitem__(self, index: tuple[int, ...]) -> NDArray[F]:
        """Get a block from the distributed array.

        Args:
            index: The index of the block.

        Returns:
            The block.
        """
        if any(i >= n for i, n in zip(index, self.block_num)):
            raise TensorError(
                "Block index out of range. The `__getitem__` method gets blocks, not elements."
            )
        try:
            canon, perm, sign = symmetrise_indices(index, self.permutations)
            # This block can become non-contiguous if it is transposed, but we should never
            # communicate it in this state (only the contiguous version is communicated)
            return self._get_block(canon).transpose(perm) * sign
        except KeyError:
            raise TensorError(f"Block {index} not found.")

    def _check_sanity(self) -> None:
        """Run sanity checks on the tensor."""
        self.world.barrier()
        block_check = np.zeros(self.block_num, dtype=np.int32)
        for block_index in self._blocks.keys():
            block_check[block_index] = 1
        block_check = self.world.allreduce(block_check, op=MPI.SUM)
        if np.any(block_check > 1):
            raise TensorError("Duplicate blocks found.")
        for block_index, block in self._blocks.items():
            if block.shape != self.get_block_shape(block_index):
                raise TensorError(f"Block shape mismatch ({block_index}).")
        for block_index, block in self._blocks.items():
            for perm, sign in self.permutations:
                shape = tuple(block.shape[i] for i in perm)
                block_index_perm = tuple(block_index[i] for i in perm)
                if shape != self.get_block_shape(block_index_perm):
                    raise TensorError(f"Block shape mismatch ({block_index}).")
        for block_index, block in self._blocks.items():
            if not block.flags.c_contiguous:
                raise TensorError(f"Block {block_index} is not C-contiguous.")

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

    @property
    def size(self) -> int:
        """Get the number of elements."""
        return int(np.prod(self.shape))

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
            raise TensorError(f"Block {index} is not owned by any process.")
        return rank

    def send_block(self, index: tuple[int, ...], dest: Optional[int]) -> None:
        """Send a block to another process.

        This method does not canonicalise or check the block index.

        Args:
            index: The index of the block.
            dest: The rank of the destination process. Must be provided if MPI is used.
        """
        if dest is None and self.world is not None:
            raise TensorError("Destination process must be provided if MPI is used.")
        if self.world is None or self.world.size == 1:
            return
        self.world.Send(self._get_block(index), dest=dest)

    def recv_block(
        self, index: tuple[int, ...], source: Optional[int], store: bool = True
    ) -> NDArray[F]:
        """Receive a block from another process.

        This method does not canonicalise or check the block index.

        Args:
            index: The index of the block.
            source: The rank of the source process. Must be provided if MPI is used.
            store: Whether to store the block.

        Returns:
            The block.
        """
        if source is None and self.world is not None:
            raise TensorError("Source process must be provided if MPI is used.")
        if self.world is None or self.world.size == 1:
            return self._get_block(index)
        block = np.empty(self.get_block_shape(index), dtype=self.dtype)
        self.world.Recv(block, source=source)
        if store:
            self._set_block(index, block)
        return block

    def as_local_ndarray(self, fill: Any = 0.0) -> NDArray[F]:
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

    def as_global_ndarray(self) -> NDArray[F]:
        """Convert the tensor to a global numpy array.

        Returns:
            The global numpy array.
        """
        array = self.as_local_ndarray()
        if self.world is None or self.world.size == 1:
            return array
        return self.world.allreduce(array, op=MPI.SUM)

    def __add__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """Distributed addition.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The sum of the two tensors.
        """
        _check_not_array(other)
        if isinstance(other, Tensor):
            return _bilinear_map(np.add, self, other)
        return _unary_map(lambda x: np.add(x, other), self)

    def __radd__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """Distributed addition.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The sum of the two tensors.
        """
        _check_not_array(other)
        if isinstance(other, Tensor):
            return _bilinear_map(np.add, other, self)
        return _unary_map(lambda x: np.add(other, x), self)

    def __sub__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """Distributed subtraction.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The difference of the two tensors.
        """
        _check_not_array(other)
        if isinstance(other, Tensor):
            return _bilinear_map(np.subtract, self, other)
        return _unary_map(lambda x: np.subtract(x, other), self)

    def __rsub__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """Distributed subtraction.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The difference of the two tensors.
        """
        _check_not_array(other)
        if isinstance(other, Tensor):
            return _bilinear_map(np.subtract, other, self)
        return _unary_map(lambda x: np.subtract(other, x), self)

    def __mul__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """Distributed multiplication.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The product of the two tensors.
        """
        _check_not_array(other)
        if isinstance(other, Tensor):
            return _bilinear_map(np.multiply, self, other)
        return _unary_map(lambda x: np.multiply(x, other), self)

    def __rmul__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """Distributed multiplication.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The product of the two tensors.
        """
        _check_not_array(other)
        if isinstance(other, Tensor):
            return _bilinear_map(np.multiply, other, self)
        return _unary_map(lambda x: np.multiply(other, x), self)

    def __truediv__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """Distributed division.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The quotient of the two tensors.
        """
        _check_not_array(other)
        if isinstance(other, Tensor):
            return _bilinear_map(np.divide, self, other)
        return _unary_map(lambda x: np.divide(x, other), self)

    def __rtruediv__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """Distributed division.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The quotient of the two tensors.
        """
        _check_not_array(other)
        if isinstance(other, Tensor):
            return _bilinear_map(np.divide, other, self)
        return _unary_map(lambda x: np.divide(other, x), self)

    def __neg__(self) -> Tensor[F]:
        """Negation.

        Returns:
            The negated tensor.
        """
        return _unary_map(np.negative, self)

    def __abs__(self) -> Tensor[F]:
        """Absolute value.

        Returns:
            The absolute tensor.
        """
        return _unary_map(np.abs, self)

    def __pow__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """Distributed exponentiation.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The exponentiated tensor.
        """
        _check_not_array(other)
        if isinstance(other, Tensor):
            return _bilinear_map(np.power, self, other)
        return _unary_map(lambda x: np.power(x, other), self)

    def __matmul__(self, other: Tensor[F]) -> Tensor[F]:
        """Distributed matrix multiplication.

        Args:
            other: The other tensor.

        Returns:
            The matrix product of the two tensors.
        """
        _check_not_array(other)
        return dot(self, other)

    def __iadd__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """In-place addition.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The sum of the two tensors.
        """
        result = self + other
        self._blocks = result._blocks
        self.permutations = result.permutations
        return self

    def __isub__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """In-place subtraction.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The difference of the two tensors.
        """
        result = self - other
        self._blocks = result._blocks
        self.permutations = result.permutations
        return self

    def __imul__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """In-place multiplication.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The product of the two tensors.
        """
        result = self * other
        self._blocks = result._blocks
        self.permutations = result.permutations
        return self

    def __itruediv__(self, other: Union[Tensor[F], F]) -> Tensor[F]:
        """In-place division.

        Args:
            other: The other tensor, or a scalar.

        Returns:
            The quotient of the two tensors.
        """
        result = self / other
        self._blocks = result._blocks
        self.permutations = result.permutations
        return self

    def __array__(self, dtype: Optional[Type[F]] = None) -> NDArray[F]:
        """Convert the tensor to a numpy array.

        Args:
            dtype: The data type of the array.

        Returns:
            The array.
        """
        return self.as_global_ndarray().astype(dtype)

    def copy(self) -> Tensor[F]:
        """Copy the tensor.

        Returns:
            The copy of the tensor.
        """
        return _unary_map(np.copy, self)

    def astype(self, dtype: Type[F]) -> Tensor[F]:
        """Change the data type of the tensor.

        Args:
            dtype: The data type.

        Returns:
            The tensor with the new data type.
        """
        return _unary_map(lambda x: x.astype(dtype), self)

    def real(self) -> Tensor[F]:
        """Get the real part of the tensor.

        Returns:
            The real part of the tensor.
        """
        return _unary_map(np.real, self)

    def imag(self) -> Tensor[F]:
        """Get the imaginary part of the tensor.

        Returns:
            The imaginary part of the tensor.
        """
        return _unary_map(np.imag, self)

    def conj(self) -> Tensor[F]:
        """Get the complex conjugate of the tensor.

        Returns:
            The complex conjugate of the tensor.
        """
        return _unary_map(np.conj, self)

    def abs(self) -> Tensor[F]:
        """Get the absolute value of the tensor.

        Returns:
            The absolute value of the tensor.
        """
        tensor_abs = _unary_map(np.abs, self)
        tensor_abs.permutations = [(perm, 1) for perm, _ in tensor_abs.permutations]
        return tensor_abs

    def min(self) -> F:
        """Get the minimum value of the tensor.

        Returns:
            The minimum value of the tensor.
        """
        min_local = np.inf
        for block_index, block in self._blocks.items():
            done = set()
            for perm, sign in self.permutations:
                block_index_perm = tuple(block_index[i] for i in perm)
                if block_index_perm in done:
                    continue
                done.add(block_index_perm)
                min_local = min(min_local, np.min(block * sign))
        if self.world is None or self.world.size == 1:
            return min_local
        min_global: F = self.world.allreduce(min_local, op=MPI.MIN)
        return min_global

    def max(self) -> F:
        """Get the maximum value of the tensor.

        Returns:
            The maximum value of the tensor.
        """
        max_local = -np.inf
        for block_index, block in self._blocks.items():
            done = set()
            for perm, sign in self.permutations:
                block_index_perm = tuple(block_index[i] for i in perm)
                if block_index_perm in done:
                    continue
                done.add(block_index_perm)
                max_local = max(max_local, np.max(block * sign))
        if self.world is None or self.world.size == 1:
            return max_local
        max_global: F = self.world.allreduce(max_local, op=MPI.MAX)
        return max_global

    def sum(self) -> F:
        """Sum the tensor."""
        sum_local = 0.0
        for block_index, block in self._blocks.items():
            sum_block = block.sum()
            done = set()
            for perm, sign in self.permutations:
                block_index_perm = tuple(block_index[i] for i in perm)
                if block_index_perm in done:
                    continue
                done.add(block_index_perm)
                sum_local += sum_block * sign
        if self.world is None or self.world.size == 1:
            return sum_local
        sum_global: F = self.world.allreduce(sum_local, op=MPI.SUM)
        return sum_global

    def transpose(self, *_axes: Union[int]) -> Tensor[F]:
        """Transpose the tensor.

        Args:
            axes: The axes to transpose.

        Returns:
            The transposed tensor.
        """
        if not _axes:
            raise TensorError("No axes provided.")
        if isinstance(_axes[0], tuple):
            if len(_axes) > 1:
                raise TensorError("Only one tuple of axes can be provided.")
            axes = tuple(_axes[0])
        else:
            if len(_axes) != self.ndim:
                raise TensorError("The number of axes provided does not match the tensor rank.")
            axes = tuple(_axes)
        if axes == tuple(range(self.ndim)):
            return self

        # Transpose the blocks
        shape = tuple(self.shape[i] for i in axes)
        block_num = tuple(self.block_num[i] for i in axes)
        blocks = {
            tuple(i[j] for j in axes): block.transpose(axes) for i, block in self._blocks.items()
        }

        # Transpose the permutations
        permutations = []
        swaps = {axes[i]: i for i in range(len(axes))}
        for perm, sign in self.permutations:
            perm = tuple(swaps[perm[i]] for i in axes)
            permutations.append((perm, sign))

        # Canonicalise the blocks
        canon_blocks = {}
        for index, block in blocks.items():
            canon, perm, sign = symmetrise_indices(index, permutations)
            canon_blocks[canon] = block.transpose(perm) * sign
            if not canon_blocks[canon].flags.c_contiguous:
                canon_blocks[canon] = np.copy(canon_blocks[canon], order="C")

        # Build the tensor
        res = Tensor(
            shape=shape,
            block_num=block_num,
            dtype=self.dtype,
            permutations=permutations,
            world=self.world,
        )
        res._blocks = canon_blocks

        return res

    @property
    def T(self) -> Tensor[F]:
        """Transpose the tensor.

        Returns:
            The transposed tensor.
        """
        return self.transpose(*range(self.ndim)[::-1])

    @property
    def H(self) -> Tensor[F]:
        """Hermitian transpose the tensor.

        Returns:
            The Hermitian transposed tensor.
        """
        return self.conj().T

    def swapaxes(self, axis1: int, axis2: int) -> Tensor[F]:
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

    def ravel(self) -> NDArray[F]:
        """Ravel the tensor.

        Returns:
            The ravelled array.
        """
        blocks = []
        for block_index in sorted(self._blocks.keys()):
            blocks.append(self[block_index].ravel())
        return np.concatenate(blocks)

    def _describe(self) -> str:
        """Describe the tensor for debugging."""
        output = ""
        output += f"Tensor [{self}, rank {self.world.rank}]\n"
        output += f"  Shape: {self.shape}\n"
        output += f"  Block num: {self.block_num}\n"
        output += "  Block shapes:\n"
        for block_index, block in self._blocks.items():
            output += f"    {block_index}: {block.shape}\n"
        if self.permutations:
            output += "  Permutations:\n"
            for perm, sign in self.permutations:
                chars = f"{''.join(EINSUM_SYMBOLS[i + 8] for i in range(len(perm)))}"
                chars += f" <-> {'+' if sign == 1 else '-'}"
                chars += f"{''.join(EINSUM_SYMBOLS[i + 8] for i in perm)}"
                output += f"    {perm} ({sign:+d}) [{chars}]\n"
        return output


def dot(
    first: Tensor[F],
    second: Tensor[F],
) -> Tensor[F]:
    """Distributed dot product.

    Args:
        first: The first tensor.
        second: The second tensor.

    Returns:
        The dot product of the two tensors.
    """
    raise NotImplementedError
    if first.ndim != 2 or second.ndim != 2:
        raise TensorError("Arrays must be 2D.")
    if first.shape[1] != second.shape[0]:
        raise TensorError("Shapes must be compatible.")
    if first.block_num[1] != second.block_num[0]:
        raise TensorError("Block shapes must be compatible.")

    world = first.world
    world_size = world.size if world is not None else 1
    world_rank = world.rank if world is not None else 0

    permutations = _combine_permutations("ij,jk->ik", first.permutations, second.permutations)

    result: Tensor[F] = Tensor(
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
    a: Tensor[F],
    b: Tensor[F],
    axes: Union[int, tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]],
    permutations: Optional[list[Permutation]] = None,
) -> Tensor[F]:
    """Compute a generalized tensor dot product over the specified axes.

    See `numpy.tensordot` for more information on the `axes` argument.

    Args:
        a: The first tensor.
        b: The second tensor.
        axes: The axes to sum over.
        permutations: The permutations of the result. If not provided, the permutations are
            inferred from the summed axes.

    Returns:
        The tensor dot product of the two tensors.
    """
    if DEBUG:
        a._check_sanity()
        b._check_sanity()

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
    chars_a = [""] * a.ndim
    chars_b = [""] * b.ndim
    chars_c = []
    n = 0
    for i, j in zip(axes_a, axes_b):
        chars_a[i] = EINSUM_SYMBOLS[n]
        chars_b[j] = EINSUM_SYMBOLS[n]
        n += 1
    for i in range(len(axes_external[0])):
        chars_a[chars_a.index("")] = EINSUM_SYMBOLS[n]
        chars_c.append(EINSUM_SYMBOLS[n])
        n += 1
    for i in range(len(axes_external[1])):
        chars_b[chars_b.index("")] = EINSUM_SYMBOLS[n]
        chars_c.append(EINSUM_SYMBOLS[n])
        n += 1
    subscript = f"{''.join(chars_a)},{''.join(chars_b)}->{''.join(chars_c)}"
    permutations = _combine_permutations(subscript, a.permutations, b.permutations)

    # Get the output array
    output_shape = tuple(a.shape[i] for i in axes_external[0])
    output_shape += tuple(b.shape[i] for i in axes_external[1])
    block_num_dummy = tuple(a.block_num[i] for i in axes_dummy[0])
    if block_num_dummy != tuple(b.block_num[i] for i in axes_dummy[1]):
        raise TensorError("Block shapes must be compatible.")
    block_num_external = tuple(a.block_num[i] for i in axes_external[0]) + tuple(
        b.block_num[i] for i in axes_external[1]
    )
    world = a.world
    world_rank = world.rank if world is not None else 0
    world_size = world.size if world is not None else 1
    c: Tensor[F] = Tensor(
        output_shape,
        dtype=a.dtype,
        block_num=block_num_external,
        permutations=permutations,
        world=world,
    )

    rank = 0
    for block_indices_external in loop_block_indices(block_num_external):
        for block_indices_dummy in loop_block_indices(
            block_num_dummy
        ):  # FIXME need to remove perms?
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
            block_indices_a_canon, perm_a, sign_a = symmetrise_indices(
                block_indices_a, a.permutations
            )
            block_indices_b_canon, perm_b, sign_b = symmetrise_indices(
                block_indices_b, b.permutations
            )
            block_indices_c_canon, perm_c, sign_c = symmetrise_indices(
                block_indices_c, c.permutations
            )
            if block_indices_c != block_indices_c_canon:
                continue
            world.barrier()

            # Get the owners
            owner_a = a.get_owner(block_indices_a_canon)
            owner_b = b.get_owner(block_indices_b_canon)
            owner_c = c.get_owner(block_indices_c, raise_error=False)  # In case permutation exists
            if owner_c is None:
                owner_c = rank
                rank = (rank + 1) % world_size
            if owner_a is None:
                raise TensorError(f"Block {block_indices_a} is not owned by any process.")
            if owner_b is None:
                raise TensorError(f"Block {block_indices_b} is not owned by any process.")
            world.barrier()

            # Send [a] to [b]
            if owner_a == owner_c == world_rank:
                block_a = a[block_indices_a]
            elif owner_a == world_rank:
                a.send_block(block_indices_a_canon, owner_c)
            elif owner_c == world_rank:
                block_a = a.recv_block(block_indices_a_canon, owner_a, store=False)
                block_a = block_a.transpose(perm_a) * sign_a
            world.barrier()

            # Send [b] to [a]
            if owner_b == owner_c == world_rank:
                block_b = b[block_indices_b]
            elif owner_b == world_rank:
                b.send_block(block_indices_b_canon, owner_c)
            elif owner_c == world_rank:
                block_b = b.recv_block(block_indices_b_canon, owner_b, store=False)
                block_b = block_b.transpose(perm_b) * sign_b
            world.barrier()

            # Multiply [a] @ [b] and add to [c]
            if owner_c == world_rank:
                block_c = np.tensordot(block_a, block_b, axes=(axes_a, axes_b))
                if c.owns_block(block_indices_c):
                    c[block_indices_c] += block_c
                else:
                    c[block_indices_c] = block_c

    if DEBUG:
        c._check_sanity()

    return c


def _contract(*operands: Any, **kwargs: Any) -> Union[F, NDArray[F]]:
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
        raise TensorError(f'Could not dispatch "{ak},{bk}->{rk}"')

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

    # Get the result transpose
    perm_res = [rk_def.index(k) for k in rk]

    # Dispatch the contraction
    axes = (tuple(axes_a[::-1]), tuple(axes_b[::-1]))  # reverse necessary?
    res = tensordot(a, b, axes=axes, permutations=permutations)

    # Transpose the result
    if rk != rk_def:
        res = res.transpose(*perm_res)

    return res


def _parse_einsum_input(operands: list[Any]) -> tuple[str, str, list[Tensor[F]]]:
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
        raise TensorError("No input operands")

    if isinstance(operands[0], str):
        subscripts = operands[0].replace(" ", "")
        operands = operands[1:]

        # Ensure all characters are valid
        for s in subscripts:
            if s in ".,->":
                continue
            if s not in EINSUM_SYMBOLS:
                raise TensorError(f"Character {s} is not a valid symbol.")

    else:
        tmp_operands = list(operands)
        operand_list: list[Tensor[F]] = []
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
            raise TensorError("Subscripts can only contain one '->'.")

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
                    raise TensorError("Invalid Ellipses.")

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= len(ssub) - 3

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise TensorError("Ellipses lengths do not match.")
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
                    raise TensorError(f"Character {s} is not a valid symbol.")
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
                raise TensorError(f"Character {s} is not a valid symbol.")
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if output_subscript.count(char) != 1:
            raise TensorError(f"Output character {s} appeared more than once in the output.")
        if char not in input_subscripts:
            raise TensorError(f"Output character {s} did not appear in the input")

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(",")) != len(operands):
        raise TensorError("Number of einsum subscripts must be equal to the number of operands.")

    return (input_subscripts, output_subscript, operands)


def einsum(*operands: Any, **kwargs: Any) -> Union[F, NDArray[F]]:
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

    # Squeeze and reduce if scalar output
    if not out:
        return res.as_global_ndarray().item()
    else:
        return res
