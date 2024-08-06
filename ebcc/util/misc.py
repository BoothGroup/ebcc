"""Miscellaneous utilities."""

from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from typing import Any, ItemsView, Iterator, KeysView, ValuesView

T = TypeVar("T")


class InheritedType:
    """Type for an inherited variable."""

    pass


Inherited = InheritedType()


class ModelNotImplemented(NotImplementedError):
    """Error for unsupported models."""

    pass


class Namespace(MutableMapping[str, T], Generic[T]):
    """Namespace class.

    Replacement for SimpleNamespace, which does not trivially allow
    conversion to a dict for heterogenously nested objects.

    Attributes can be added and removed, using either string indexing or
    accessing the attribute directly.
    """

    _members: dict[str, T]

    def __init__(self, **kwargs: T):
        """Initialise the namespace."""
        self.__dict__["_members"] = {}
        for key, val in kwargs.items():
            self.__dict__["_members"][key] = val

    def __setitem__(self, key: str, val: T) -> None:
        """Set an item."""
        self.__dict__["_members"][key] = val

    def __setattr__(self, key: str, val: T) -> None:
        """Set an attribute."""
        return self.__setitem__(key, val)

    def __getitem__(self, key: str) -> Any:
        """Get an item."""
        value: Any = self.__dict__["_members"][key]
        return value

    def __getattr__(self, key: str) -> Any:
        """Get an attribute."""
        if key in self.__dict__:
            return self.__dict__[key]
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError(f"Namespace object has no attribute {key}")

    def __delitem__(self, key: str) -> None:
        """Delete an item."""
        self._members.pop(key)

    def __delattr__(self, key: str) -> None:
        """Delete an attribute."""
        return self.__delitem__(key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the namespace as a dictionary."""
        yield from self._members

    def __eq__(self, other: Any) -> bool:
        """Check equality."""
        if not isinstance(other, Namespace):
            return False
        return dict(self) == dict(other)

    def __ne__(self, other: Any) -> bool:
        """Check inequality."""
        return not self == other

    def __contains__(self, key: Any) -> bool:
        """Check if an attribute exists."""
        return key in self._members

    def __len__(self) -> int:
        """Get the number of attributes."""
        return len(self._members)

    def keys(self) -> KeysView[str]:
        """Get keys of the namespace as a dictionary."""
        return self._members.keys()

    def values(self) -> ValuesView[T]:
        """Get values of the namespace as a dictionary."""
        return self._members.values()

    def items(self) -> ItemsView[str, T]:
        """Get items of the namespace as a dictionary."""
        return self._members.items()

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"Namespace({self._members})"


class Timer:
    """Timer class."""

    def __init__(self) -> None:
        """Initialise the timer."""
        self.t_init = time.perf_counter()
        self.t_prev = time.perf_counter()
        self.t_curr = time.perf_counter()

    def lap(self) -> float:
        """Return the time since the last call to `lap`."""
        self.t_prev, self.t_curr = self.t_curr, time.perf_counter()
        return self.t_curr - self.t_prev

    __call__ = lap

    def total(self) -> float:
        """Return the total time since initialization."""
        return time.perf_counter() - self.t_init

    @staticmethod
    def format_time(seconds: float, precision: int = 2) -> str:
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
