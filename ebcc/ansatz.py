"""Ansatz definition.
"""

import importlib

import numpy as np

from ebcc import util


named_ansatzes = {
        "CCSD": ("CCSD", "", 0, 0),
        "CCSDT": ("CCSDT", "", 0, 0),
        "CCSD(T)": ("CCSD(T)", "", 0, 0),
        "CC2": ("CC2", "", 0, 0),
        "CC3": ("CC3", "", 0, 0),
        "CCSD-S-1-1": ("CCSD", "S", 1, 1),
        "CCSD-SD-1-1": ("CCSD", "SD", 1, 1),
        "CCSD-SD-1-2": ("CCSD", "SD", 1, 2),
}


class Ansatz:
    """Ansatz class.
    """

    def __init__(
            self,
            fermion_ansatz: str = "CCSD",
            boson_ansatz: str = "",
            fermion_coupling_rank: int = 0,
            boson_coupling_rank: int = 0,
    ):
        self.fermion_ansatz = fermion_ansatz
        self.boson_ansatz = boson_ansatz
        self.fermion_coupling_rank = fermion_coupling_rank
        self.boson_coupling_rank = boson_coupling_rank

    def _get_eqns(self, prefix):
        """Get the module which contains the generated equations for
        the current model.
        """
        name = prefix + self.name.replace("-", "_")
        name = name.replace("(", "_").replace(")", "")
        eqns = importlib.import_module("ebcc.codegen.%s" % name)
        return eqns

    @classmethod
    def from_string(cls, string):
        """Build an Ansatz from a string for the default ansatzes.

        Parameters
        ----------
        input : str
            Input string

        Returns
        -------
        ansatz : Ansatz
            Ansatz object
        """

        if string not in named_ansatzes:
            raise util.ModelNotImplemented(string)

    @property
    def name(self):
        """Get a string with the name of the method.

        Returns
        -------
        name : str
            Name of the method.
        """
        name = self.fermion_ansatz
        if self.boson_ansatz:
            name += "-%s" % self.boson_ansatz
        if self.fermion_coupling_rank or self.boson_coupling_rank:
            name += "-%d" % self.fermion_coupling_rank
            name += "-%d" % self.boson_coupling_rank
        return name

    @property
    def has_perturbative_correction(self):
        """Return a boolean indicating whether the ansatz includes a
        perturbative correction e.g. CCSD(T).

        Returns
        -------
        perturbative : bool
            Boolean indicating if the ansatz is perturbatively
            corrected.
        """
        return any(
                "(" in ansatz and ")" in ansatz
                for ansatz in (self.fermion_ansatz, self.boson_ansatz)
        )
