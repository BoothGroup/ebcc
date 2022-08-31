"""Generalised equation-of-motion solver.
"""

from ebcc import reom, util


@util.inherit_docstrings
class GEOM(reom.REOM):
    """Generalised equation-of-motion base class."""

    @property
    def name(self):
        return self.excitation_type.upper() + "-GEOM-" + self.ebcc.name


@util.inherit_docstrings
class IP_GEOM(GEOM, reom.IP_REOM):
    """Generalised equation-of-motion class for ionisation potentials."""

    pass


@util.inherit_docstrings
class EA_GEOM(GEOM, reom.EA_REOM):
    """Generalised equation-of-motion class for electron affinities."""

    pass


@util.inherit_docstrings
class EE_GEOM(GEOM, reom.EE_REOM):
    """Generalised equation-of-motion class for neutral excitations."""

    pass
