"""Miscellaneous utilities."""

import time


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
