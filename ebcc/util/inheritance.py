"""Tools for class inheritance."""


class InheritedType:
    """Type for an inherited variable in `Options` classes."""

    pass


Inherited = InheritedType()


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
