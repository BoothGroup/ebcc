"""Damping and DIIS control."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from ebcc import numpy as np
from ebcc import util
from ebcc.core.precision import astype
from ebcc.backend import _put, ensure_scalar

if TYPE_CHECKING:
    from typing import Any, Optional

    from numpy import float64
    from numpy.typing import NDArray

    from ebcc.util import _BaseOptions

    T = float64


class BaseDamping(ABC):
    """Base class for damping."""

    _arrays: dict[int, NDArray[T]]
    _errors: dict[int, NDArray[T]]
    _counter: int

    def __init__(self, *args: Any, options: Optional[_BaseOptions] = None, **kwargs: Any) -> None:
        """Initialise the damping object."""
        self._arrays = {}
        self._errors = {}
        self._counter = 0

    def __call__(self, array: NDArray[T], error: Optional[NDArray[T]] = None) -> NDArray[T]:
        """Apply damping to the array.

        Args:
            array: The array to damp.
            error: The error array.

        Returns:
            The damped array.
        """
        self.push(array, error=error)
        return self.extrapolate()

    @abstractmethod
    def push(self, array: NDArray[T], error: Optional[NDArray[T]] = None) -> None:
        """Push the array and error into the damping object.

        Args:
            array: The array to push.
            error: The error array to push.
        """
        pass

    @abstractmethod
    def extrapolate(self) -> NDArray[T]:
        """Extrapolate the next array.

        Returns:
            The extrapolated array.
        """
        pass

    def reset(self) -> None:
        """Reset the damping object."""
        self._arrays = {}
        self._errors = {}
        self._counter = 0

    def __len__(self) -> int:
        """Get the number of arrays stored in the damping object."""
        return len(self._arrays)


class NoDamping(BaseDamping):
    """No damping."""

    def __init__(self, *args: Any, options: Optional[_BaseOptions] = None, **kwargs: Any) -> None:
        """Apply damping to the array.

        Args:
            array: The array to damp.
            error: The error array.

        Returns:
            The damped array.
        """
        pass

    def __call__(self, array: NDArray[T], error: Optional[NDArray[T]] = None) -> NDArray[T]:
        """Apply damping to the array.

        Args:
            array: The array to damp.
            error: The error array.

        Returns:
            The damped array.
        """
        return array

    def push(self, array: NDArray[T], error: Optional[NDArray[T]] = None) -> None:
        """Push the array and error into the damping object.

        Args:
            array: The array to push.
            error: The error array to push.
        """
        raise NotImplementedError

    def extrapolate(self) -> NDArray[T]:
        """Extrapolate the next array.

        Returns:
            The extrapolated array.
        """
        raise NotImplementedError


class LinearDamping(BaseDamping):
    """Linear damping."""

    def __init__(
        self, *args: Any, factor: Optional[float] = None, options: Optional[_BaseOptions] = None
    ) -> None:
        """Initialise the damping object.

        Args:
            factor: The damping factor. If `None`, use `options.damping`, if available. Otherwise,
                use `0.5`.
            options: The options object.
        """
        super().__init__()
        if factor is None:
            factor = getattr(options, "damping", 0.5)
        self.factor = factor

    def push(self, array: NDArray[T], error: Optional[NDArray[T]] = None) -> None:
        """Push the array and error into the damping object.

        Args:
            array: The array to push.
            error: The error array to push.
        """
        if len(self) == 2:
            self._arrays.pop(min(self._arrays.keys()))
        self._arrays[self._counter] = array
        self._counter += 1

    def extrapolate(self) -> NDArray[T]:
        """Extrapolate the next array.

        Returns:
            The extrapolated array.
        """
        if len(self) < 2:
            return next(iter(self._arrays.values()))
        (_, previous), (_, current) = sorted(self._arrays.items())
        return self.factor * previous + (1.0 - self.factor) * current


class DIIS(BaseDamping):
    """Direct inversion in the iterative subspace."""

    def __init__(
        self,
        space: Optional[int] = None,
        min_space: Optional[int] = None,
        options: Optional[_BaseOptions] = None,
    ) -> None:
        """Initialize the DIIS object.

        Args:
            space: The number of vectors to store in the DIIS space. If `None`, use
                `options.diis_space`, if available. Otherwise, use `6`.
            min_space: The minimum number of vectors to store in the DIIS space. If `None`, use
                `options.diis_min_space`, if available. Otherwise, use `1`.
            options: The options object.
        """
        super().__init__()
        if space is None:
            space = getattr(options, "diis_space", 6)
        if min_space is None:
            min_space = getattr(options, "diis_min_space", 1)
        self.space = space
        self.min_space = min_space
        self._norm_cache = {}

    def _error_norm(self, i: int, j: int) -> T:
        """Calculate the error norm between two arrays.

        Args:
            i: The first counter.
            j: The second counter.

        Returns:
            The error norm.
        """
        if (i, j) not in self._norm_cache:
            self._norm_cache[i, j] = ensure_scalar(
                np.dot(np.conj(np.ravel(self._errors[i])), np.ravel(self._errors[j]))
            )
            self._norm_cache[j, i] = self._norm_cache[i, j].conjugate()
        return self._norm_cache[i, j]

    def push(self, array: NDArray[T], error: Optional[NDArray[T]] = None) -> None:
        """Push the array and error into the damping object.

        Args:
            array: The array to push.
            error: The error array to push.
        """
        # Get the error if not provided
        if error is None and -1 in self._arrays:
            error = array - self._arrays[-1]
        elif error is None:
            self._arrays[-1] = array
            return

        # Push the array and error into the DIIS subspace
        self._arrays[self._counter] = array
        self._errors[self._counter] = error
        self._counter += 1

        # Remove an array if the space is exceeded
        if len(self) > self.space:
            errors = {
                counter: self._error_norm(counter, counter) for counter in self._errors.keys()
            }
            counter = max(errors, key=errors.__getitem__)
            self._arrays.pop(counter)
            self._errors.pop(counter)

    def extrapolate(self) -> NDArray[T]:
        """Extrapolate the next array.

        Returns:
            The extrapolated array.
        """
        # Return the last array if the space is less than the minimum space
        counters = sorted(self._errors.keys())
        size = len(counters)
        if size < self.min_space:
            return self._arrays[-1]

        # Build the error matrix
        errors = np.array(
            [
                [self._error_norm(counter_i, counter_j) for counter_j in counters]
                for counter_i in counters
            ]
        )
        zeros = np.zeros((1, 1), dtype=errors.dtype)
        ones = np.ones((size, 1), dtype=errors.dtype)
        matrix = np.block([[errors, -ones], [-np.transpose(ones), zeros]])

        # Build the right-hand side
        zeros = np.zeros((size,), dtype=errors.dtype)
        ones = np.ones((1,), dtype=errors.dtype)
        residual = np.block([zeros, -ones])

        # Solve the linear problem
        try:
            c = np.linalg.solve(matrix, residual)
        except Exception:
            w, v = np.linalg.eigh(matrix)
            if np.any(np.abs(w) < 1e-14):
                # avoiding fancy indexing for compatibility
                for i in range(size + 1):
                    if np.abs(w[i]) < 1e-14:
                        _put(v, np.ix_(np.arange(size + 1), np.array([i])), np.zeros_like(v[:, i]))
            c = util.einsum("pi,qi,i,q->p", v, np.conj(v), w**-1.0, residual)

        # Construct the new array
        array = np.zeros_like(self._arrays[-1])
        for counter, coefficient in zip(counters, c):
            array += self._arrays[counter] * coefficient
        error = np.zeros_like(self._arrays[-1])
        for counter, coefficient in zip(counters, c):
            error += self._errors[counter] * coefficient

        # Replace the previous array with the extrapolated array
        self._arrays[-1] = array
        self._arrays[self._counter] = array
        self._errors[self._counter] = error

        return array

    def reset(self) -> None:
        """Reset the damping object."""
        super().reset()
        self._norm_cache = {}
