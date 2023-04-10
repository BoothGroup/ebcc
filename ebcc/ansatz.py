"""Ansatz definition.
"""

import importlib

import numpy as np

from ebcc import METHOD_TYPES, util

named_ansatzes = {
    "MP2": ("MP2", "", 0, 0),
    "MP3": ("MP3", "", 0, 0),
    "CCD": ("CCD", "", 0, 0),
    "CCSD": ("CCSD", "", 0, 0),
    "CCSDT": ("CCSDT", "", 0, 0),
    "CCSDTQ": ("CCSDTQ", "", 0, 0),
    "CCSD(T)": ("CCSD(T)", "", 0, 0),
    "CC2": ("CC2", "", 0, 0),
    "CC3": ("CC3", "", 0, 0),
    "QCISD": ("QCISD", "", 0, 0),
    "CCSD-S-1-1": ("CCSD", "S", 1, 1),
    "CCSD-SD-1-1": ("CCSD", "SD", 1, 1),
    "CCSD-SD-1-2": ("CCSD", "SD", 1, 2),
}


def name_to_identifier(name):
    """Convert an ansatz name to an identifer that can be used for
    variable and file names.

    >>> identifier_to_name("CCSD(T)")
    CCSDxTx
    >>> identifier_to_name("CCSD-SD-1-2")
    CCSD_SD_1_2
    """

    iden = name.replace("(", "x").replace(")", "x")
    iden = iden.replace("[", "y").replace("]", "y")
    iden = iden.replace("-", "_")

    return iden


def identifity_to_name(iden):
    """Convert an ansatz identifier to a name.

    >>> identifier_to_name("CCSDxTx")
    CCSD(T)
    >>> identifier_to_name("CCSD_SD_1_2")
    CCSD-SD-1-2
    """

    name = iden.replace("-", "_")
    while "x" in name:
        name = name.replace("x", "(", 1).replace("x", ")", 1)
    while "y" in name:
        name = name.replace("y", "(", 1).replace("y", ")", 1)

    return name


class Ansatz:
    """Ansatz class.

    Parameters
    ----------
    fermion_ansatz : str, optional
        Fermionic ansatz. Default value is "CCSD".
    boson_ansatz : str, optional
        Rank of bosonic excitations. Default is "".
    fermion_coupling_rank : int, optional
        Rank of fermionic term in coupling. Default is 0.
    boson_coupling_rank : int, optional
        Rank of bosonic term in coupling. Default is 0.
    """

    def __init__(
        self,
        fermion_ansatz: str = "CCSD",
        boson_ansatz: str = "",
        fermion_coupling_rank: int = 0,
        boson_coupling_rank: int = 0,
        module_name: str = None,
    ):
        self.fermion_ansatz = fermion_ansatz
        self.boson_ansatz = boson_ansatz
        self.fermion_coupling_rank = fermion_coupling_rank
        self.boson_coupling_rank = boson_coupling_rank
        self.module_name = module_name

    def _get_eqns(self, prefix):
        """Get the module which contains the generated equations for
        the current model.
        """

        if self.module_name is None:
            name = prefix + name_to_identifier(self.name)
        else:
            name = prefix + self.module_name

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

        return cls(*named_ansatzes[string])

    def __repr__(self):
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
    def name(self):
        return repr(self)

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
            "(" in ansatz and ")" in ansatz for ansatz in (self.fermion_ansatz, self.boson_ansatz)
        )

    @property
    def is_one_shot(self):
        """Return a boolean indicating whether the ansatz is simply
        a one-shot energy calculation e.g. MP2.

        Returns
        -------
        one_shot : bool
            Boolean indicating if the ansatz is a one-shot energy
            calculation.
        """
        return all(
            ansatz.startswith("MP") or ansatz == ""
            for ansatz in (self.fermion_ansatz, self.boson_ansatz)
        )

    @property
    def correlated_cluster_ranks(self):
        """Get a list of cluster operator rank numbers for each of
        the fermionic, bosonic, and coupling ansatzes, for the
        correlated space (see space.py).

        Returns
        -------
        ranks : tuple of tuple of int
            Cluster operator ranks for the fermionic, bosonic, and
            coupling ansatzes, for the correlated space.
        """

        ranks = []

        notations = {
            "S": [1],
            "D": [2],
            "T": [3],
            "Q": [4],
            "2": [1, 2],
            "3": [1, 2, 3],
            "4": [1, 2, 3, 4],
        }

        for i, op in enumerate([self.fermion_ansatz, self.boson_ansatz]):
            # Remove any perturbative corrections
            while "(" in op:
                start = op.index("(")
                end = op.index(")")
                op = op[:start]
                if (end + 1) < len(op):
                    op += op[end + 1 :]

            # Check in order of longest -> shortest string in case
            # one method name starts with a substring equal to the
            # name of another method
            if i == 0:
                for method_type in sorted(METHOD_TYPES, key=len)[::-1]:
                    if op.startswith(method_type):
                        op = op.lstrip(method_type)
                        break

            # If it's Moller-Plesset perturbation theory, we only
            # need to initialise second-order amplitudes
            if method_type == "MP":
                op = "D"

            # Remove any lower case characters, as these correspond
            # to active space
            op = "".join([char for char in op if char.isupper() or char.isnumeric()])

            # Determine the ranks
            ranks_entry = set()
            for char in op:
                for rank in notations[char]:
                    ranks_entry.add(rank)
            ranks.append(tuple(sorted(list(ranks_entry))))

        # Get the coupling ranks
        for op in [self.fermion_coupling_rank, self.boson_coupling_rank]:
            ranks.append(tuple(range(1, op + 1)))

        return tuple(ranks)

    @property
    def active_cluster_ranks(self):
        """Get a list of cluster operator rank numbers for each of
        the fermionic, bosonic, and coupling ansatzes, for the
        active space (see space.py).

        Returns
        -------
        ranks : tuple of tuple of int
            Cluster operator ranks for the fermionic, bosonic, and
            coupling ansatzes, for the active space.
        """

        ranks = []

        notations = {
            "s": [1],
            "d": [2],
            "t": [3],
            "q": [4],
        }

        for i, op in enumerate([self.fermion_ansatz, self.boson_ansatz]):
            # Remove any perturbative corrections
            while "(" in op:
                start = op.index("(")
                end = op.index(")")
                op = op[:start]
                if (end + 1) < len(op):
                    op += op[end + 1 :]

            # Check in order of longest -> shortest string in case
            # one method name starts with a substring equal to the
            # name of another method
            if i == 0:
                for method_type in sorted(METHOD_TYPES, key=len)[::-1]:
                    if op.startswith(method_type):
                        op = op.lstrip(method_type)
                        break

            # Remove any lower case characters, as these correspond
            # to active space
            op = "".join([char for char in op if char.islower()])

            # Determine the ranks
            ranks_entry = set()
            for char in op:
                for rank in notations[char]:
                    ranks_entry.add(rank)
            ranks.append(tuple(sorted(list(ranks_entry))))

        # Get the coupling ranks
        # FIXME how to handle? if it's ever supported
        ranks.append(tuple())

        return tuple(ranks)
