"""Generalised equation-of-motion solver."""

from ebcc import reom, util


class GEOM(reom.REOM, metaclass=util.InheritDocstrings):
    """Generalised equation-of-motion base class."""

    @property
    @util.has_docstring
    def name(self):
        return self.excitation_type.upper() + "-GEOM-" + self.ebcc.name


class IP_GEOM(GEOM, reom.IP_REOM):
    """Generalised equation-of-motion class for ionisation potentials."""

    pass


class EA_GEOM(GEOM, reom.EA_REOM):
    """Generalised equation-of-motion class for electron affinities."""

    pass


class EE_GEOM(GEOM, reom.EE_REOM):
    """Generalised equation-of-motion class for neutral excitations."""

    pass
