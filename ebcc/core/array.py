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


def loop_block_indices(num_blocks: tuple[int]) -> Iterator[tuple[int]]:
    """Loop over the block indices.

    Args:
        num_blocks: The number of blocks in each dimension.

    Returns:
        The block indices.
    """
    yield from itertools.product(*map(range, num_blocks))


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


class TiledArray:
    """Tiled array."""

    def __init__(
        self,
        shape: tuple[int],
        block_shape: tuple[int],
        dtype: Optional[Type[T]] = None,
        world: Optional[MPI.Comm] = None,
    ) -> None:
        """Initialise the tiled array.

        Args:
            shape: The shape of the array.
            block_shape: The shape of the blocks.
            dtype: The data type of the array.
            world: The MPI communicator.
        """
        if any(s % bs != 0 for s, bs in zip(shape, block_shape)):
            raise ValueError("Block shape must divide shape.")

        if world is None:
            world = MPI.COMM_WORLD if MPI is not None else None
        if dtype is None:
            dtype = types[float]

        self.shape = shape
        self.block_shape = block_shape
        self.dtype = dtype
        self.world = world

        self._blocks: dict[tuple[int], NDArray[T]] = {}

    def __setitem__(self, index: tuple[int], block: NDArray[T]) -> None:
        """Set a block in the distributed array.

        Args:
            index: The index of the block.
            block: The block.
        """
        self._blocks[index] = block

    def __getitem__(self, index: tuple[int]) -> NDArray[T]:
        """Get a block from the distributed array.

        Args:
            index: The index of the block.

        Returns:
            The block.
        """
        return self._blocks[index]

    @property
    def num_blocks(self) -> tuple[int]:
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
        block = self._blocks[index]
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
            self._blocks[index] = block
        return block

    def as_local_ndarray(self) -> NDArray[T]:
        """Convert the tiled array to a local numpy array.

        Returns:
            The local numpy array.
        """
        array = np.zeros(self.shape, dtype=self.dtype)
        for block_index in loop_block_indices(self.num_blocks):
            if self.owns_block(block_index):
                index = tuple(
                    slice(i * bs, (i + 1) * bs)
                    for i, bs in zip(block_index, self.block_shape)
                )
                array[index] = self._blocks[block_index]
        return array

    def as_global_ndarray(self) -> NDArray[T]:
        """Convert the tiled array to a global numpy array.

        Returns:
            The global numpy array.
        """
        array = np.empty(self.shape, dtype=self.dtype)
        for block_index in loop_block_indices(self.num_blocks):
            index = tuple(
                slice(i * bs, (i + 1) * bs)
                for i, bs in zip(block_index, self.block_shape)
            )
            owner = self.get_owner(block_index)
            if self.owns_block(block_index):
                array[index] = self._blocks[block_index]
        return array

    def __add__(self, other: TiledArray) -> TiledArray:
        """Distributed addition.

        Args:
            other: The other tiled array.

        Returns:
            The sum of the two tiled arrays.
        """
        _check_shapes(self, other)
        output = TiledArray(self.shape, self.block_shape, world=self.world)
        for block_index in loop_block_indices(self.num_blocks):
            if self.owns_block(block_index):
                if other.owns_block(block_index):
                    block = self[block_index] + other[block_index]
                else:
                    other_block = other.recv_block(block_index, other.get_owner(block_index), store=False)
                    block = self[block_index] + other_block
                output[block_index] = block
        return output
