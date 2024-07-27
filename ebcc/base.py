"""Base classes."""

from abc import ABC, abstractmethod

from ebcc import util


class EBCC(ABC):
    """Base class for electron-boson coupled cluster."""

    pass


class EOM(ABC):
    """Base class for equation-of-motion methods."""

    pass


class BruecknerEBCC(ABC):
    """Base class for Brueckner orbital methods."""

    pass


class ERIs(ABC, util.Namespace):
    """Base class for electronic repulsion integrals."""

    pass


class Fock(ABC, util.Namespace):
    """Base class for Fock matrices."""

    pass
