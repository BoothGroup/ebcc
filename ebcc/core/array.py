"""Tiled arrays with MPI support."""

from __future__ import annotations

import itertools

from ebcc import numpy as np
from ebcc.core.precision import types

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


if TYPE_CHECKING:
    from typing import Type, TypeVar, Optional, Iterator

    from ebcc.numpy.typing import NDArray

    T = TypeVar("T", float, complex)


def loop_block_indices(block_num: tuple[int]) -> Iterator[tuple[int]]:
    """Loop over the block indices.

    Args:
        block_num: The number of blocks in each dimension.

    Returns:
        The block indices.
    """
    yield from itertools.product(*map(range, block_num))


def _check_shapes(first: TiledArray, second: TiledArray) -> None:
    """Check that the shapes of two tiled arrays are compatible.

    Args:
        first: The first tiled array.
        second: The second tiled array.
    """
    if first.shape != second.shape:
        raise ValueError("Shapes must be equal.")
    if first.block_shape != second.block_shape:
        raise ValueError("Block shapes must be equal.")


def _bilinear_map(
    func: Callable[[NDArray[float], NDArray[float]], NDArray[float]],
    first: TiledArray,
    second: TiledArray,
) -> TiledArray:
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
    result = TiledArray(first.shape, first.block_shape, world=first.world)

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


def _auto_block(shape: tuple[int], world: Optional[MPI.Comm] = None) -> tuple[int]:
    """Automatically determine the block shape.

    Args:
        shape: The shape of the tensor.

    Returns:
        The block shape.
    """
    best_divisors = [1] * len(shape)
    for i, s in enumerate(shape):
        for divisor in range(2, 9):
            if s % divisor == 0:
                best_divisors[i] = divisor


class Tensor:
    """Tensor class."""

    def __init__(
        self,
        shape: tuple[int],
        dtype: Optional[Type[T]] = None,
        block_shape: Optional[tuple[int]] = None,
        block_num: Optional[tuple[int]] = None,
        world: Optional[MPI.Comm] = None,
    ) -> None:
        """Initialise the tensor.

        Args:
            shape: The shape of the tensor.
            dtype: The data type of the tensor.
            block_shape: The shape of the blocks in each dimension.
            block_num: The number of blocks in each dimension. If provided, `block_shape` is
                ignored.
            world: The MPI communicator.
        """
        if block_num is not None:
            block_shape = tuple(s // bs for s, bs in zip(shape, block_num))
        if any(s % bs != 0 for s, bs in zip(shape, block_shape)):
            raise ValueError("Block shape must divide shape.")

        if world is None:
            world = MPI.COMM_WORLD if MPI is not None else None
        if dtype is None:
            dtype = types[float]

        self.shape = shape
        self.dtype = dtype
        self.block_shape = block_shape
        self.world = world

        self._blocks: dict[tuple[int], NDArray[T]] = {}

    def _set_block(self, index: tuple[int], block: NDArray[T]) -> None:
        """Set a block in the distributed array.

        Args:
            index: The index of the block.
            block: The block.
        """
        self._blocks[index] = block

    def _get_block(self, index: tuple[int]) -> NDArray[T]:
        """Get a block from the distributed array.

        Args:
            index: The index of the block.

        Returns:
            The block.
        """
        return self._blocks[index]

    @property
    def block_num(self) -> tuple[int]:
        """Get the number of blocks in each dimension."""
        return tuple(s // bs for s, bs in zip(self.shape, self.block_shape))

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
        return index in self._blocks

    def get_owner(self, index: tuple[int], raise_error: bool = True) -> Optional[int]:
        """Get the owner of a block.

        Args:
            index: The index of the block.
            raise_error: Whether to raise an error if the block is not owned.

        Returns:
            The rank of the owner.
        """
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
        block = self._get_block(index)
        self.world.Send(block, dest=dest)

    def recv_block(self, index: tuple[int], source: int, store: bool = True) -> NDArray[T]:
        """Receive a block from another process.

        Args:
            index: The index of the block.
            source: The rank of the source process.
            store: Whether to store the block.

        Returns:
            The block.
        """
        block = np.empty(self.block_shape, dtype=self.dtype)
        self.world.Recv(block, source=source)
        if store:
            self._set_block(index, block)
        return block

    def as_local_ndarray(self) -> NDArray[T]:
        """Convert the tiled array to a local numpy array.

        Returns:
            The local numpy array.
        """
        array = np.zeros(self.shape, dtype=self.dtype)
        for block_index in loop_block_indices(self.block_num):
            if self.owns_block(block_index):
                index = tuple(
                    slice(i * bs, (i + 1) * bs)
                    for i, bs in zip(block_index, self.block_shape)
                )
                array._set_block(index, self._get_block(block_index))
        return array

    def as_global_ndarray(self) -> NDArray[T]:
        """Convert the tiled array to a global numpy array.

        Returns:
            The global numpy array.
        """
        array = np.empty(self.shape, dtype=self.dtype)
        for block_index in loop_block_indices(self.block_num):
            index = tuple(
                slice(i * bs, (i + 1) * bs)
                for i, bs in zip(block_index, self.block_shape)
            )
            owner = self.get_owner(block_index)
            if self.owns_block(block_index):
                array._set_block(index, self._get_block(block_index))
        return array

    def __add__(self, other: TiledArray) -> TiledArray:
        """Distributed addition.

        Args:
            other: The other tiled array.

        Returns:
            The sum of the two tiled arrays.
        """
        return _bilinear_map(np.add, self, other)

    def __sub__(self, other: TiledArray) -> TiledArray:
        """Distributed subtraction.

        Args:
            other: The other tiled array.

        Returns:
            The difference of the two tiled arrays.
        """
        return _bilinear_map(np.subtract, self, other)

    def __mul__(self, other: TiledArray) -> TiledArray:
        """Distributed multiplication.

        Args:
            other: The other tiled array.

        Returns:
            The product of the two tiled arrays.
        """
        return _bilinear_map(np.multiply, self, other)

    def __truediv__(self, other: TiledArray) -> TiledArray:
        """Distributed division.

        Args:
            other: The other tiled array.

        Returns:
            The quotient of the two tiled arrays.
        """
        return _bilinear_map(np.divide, self, other)

    def __iadd__(self, other: TiledArray) -> TiledArray:
        """In-place addition.

        Args:
            other: The other tiled array.

        Returns:
            The sum of the two tiled arrays.
        """
        return _bilinear_map(np.add, self, other)

    def __isub__(self, other: TiledArray) -> TiledArray:
        """In-place subtraction.

        Args:
            other: The other tiled array.

        Returns:
            The difference of the two tiled arrays.
        """
        return _bilinear_map(np.subtract, self, other)

    def __imul__(self, other: TiledArray) -> TiledArray:
        """In-place multiplication.

        Args:
            other: The other tiled array.

        Returns:
            The product of the two tiled arrays.
        """
        return _bilinear_map(np.multiply, self, other)
