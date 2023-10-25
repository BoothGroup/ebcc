"""Ansatz definition."""

import importlib

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
    "DCD": ("DCD", "", 0, 0),
    "DCSD": ("DCSD", "", 0, 0),
    "CCSDt": ("CCSDt", "", 0, 0),
    "CCSDt'": ("CCSDt'", "", 0, 0),
    "CCSD-S-1-1": ("CCSD", "S", 1, 1),
    "CCSD-SD-1-1": ("CCSD", "SD", 1, 1),
    "CCSD-SD-1-2": ("CCSD", "SD", 1, 2),
}


def name_to_identifier(name):
    """
    Convert an ansatz name to an identifier. The identifier is used as for
    the filename of the module containing the generated equations, where
    the name may contain illegal characters.

    Parameters
    ----------
    name : str
        Name of the ansatz.

    Returns
    -------
    iden : str
        Identifier for the ansatz.

    Examples
    --------
    >>> identifier_to_name("CCSD(T)")
    CCSDxTx
    >>> identifier_to_name("CCSD-SD-1-2")
    CCSD_SD_1_2
    """

    iden = name.replace("(", "x").replace(")", "x")
    iden = iden.replace("[", "y").replace("]", "y")
    iden = iden.replace("-", "_")
    iden = iden.replace("'", "p")

    return iden


def identifity_to_name(iden):
    """
    Convert an ansatz identifier to a name. Inverse operation of
    `name_to_identifier`.

    Parameters
    ----------
    iden : str
        Identifier for the ansatz.

    Returns
    -------
    name : str
        Name of the ansatz.

    Examples
    --------
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
    name = name.replace("p", "'")

    return name


class Ansatz:
    """
    Ansatz class.

    Parameters
    ----------
    fermion_ansatz : str, optional
        Fermionic ansatz. Default value is `"CCSD"`.
    boson_ansatz : str, optional
        Rank of bosonic excitations. Default value is `""`.
    fermion_coupling_rank : int, optional
        Rank of fermionic term in coupling. Default value is `0`.
    boson_coupling_rank : int, optional
        Rank of bosonic term in coupling. Default value is `0`.
    density_fitting : bool, optional
        Use density fitting. Default value is `False`.
    module_name : str, optional
        Name of the module containing the generated equations. If `None`,
        the module name is generated from the ansatz name. Default value is
        `None`.
    """

    def __init__(
        self,
        fermion_ansatz: str = "CCSD",
        boson_ansatz: str = "",
        fermion_coupling_rank: int = 0,
        boson_coupling_rank: int = 0,
        density_fitting: bool = False,
        module_name: str = None,
    ):
        self.fermion_ansatz = fermion_ansatz
        self.boson_ansatz = boson_ansatz
        self.fermion_coupling_rank = fermion_coupling_rank
        self.boson_coupling_rank = boson_coupling_rank
        self.density_fitting = density_fitting
        self.module_name = module_name

    def _get_eqns(self, prefix):
        """Get the module containing the generated equations."""

        if self.module_name is None:
            name = prefix + name_to_identifier(self.name)
        else:
            name = self.module_name

        eqns = importlib.import_module("ebcc.codegen.%s" % name)

        return eqns

    @classmethod
    def from_string(cls, string, density_fitting=False):
        """
        Build an Ansatz from a string for the default ansatzes.

        Parameters
        ----------
        input : str
            Input string
        density_fitting : bool, optional
            Use density fitting. Default value is `False`.

        Returns
        -------
        ansatz : Ansatz
            Ansatz object
        """

        if string not in named_ansatzes:
            raise util.ModelNotImplemented(string)

        return cls(*named_ansatzes[string], density_fitting=density_fitting)

    def __repr__(self):
        """
        Get a string with the name of the method.

        Returns
        -------
        name : str
            Name of the method.
        """
        name = ""
        if self.density_fitting:
            name += "DF"
        name += self.fermion_ansatz
        if self.boson_ansatz:
            name += "-%s" % self.boson_ansatz
        if self.fermion_coupling_rank or self.boson_coupling_rank:
            name += "-%d" % self.fermion_coupling_rank
            name += "-%d" % self.boson_coupling_rank
        return name

    @property
    def name(self):
        """Get the name of the ansatz."""
        return repr(self)

    @property
    def has_perturbative_correction(self):
        """
        Return a boolean indicating whether the ansatz includes a
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
        """
        Return a boolean indicating whether the ansatz is simply a one-shot
        energy calculation e.g. MP2.

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

    def fermionic_cluster_ranks(self, spin_type="G"):
        """
        Get a list of cluster operator ranks for the fermionic space.

        Parameters
        ----------
        spin_type : str, optional
            Spin type of the cluster operator. Default value is `"G"`.

        Returns
        -------
        ranks : list of tuples
            List of cluster operator ranks, each element is a tuple
            containing the name, the slices and the rank.
        """

        ranks = []
        if not self.fermion_ansatz:
            return ranks

        notations = {
            "S": [("t1", "ov", 1)],
            "D": [("t2", "oovv", 2)],
            "T": [("t3", "ooovvv", 3)],
            "t": [("t3", "ooOvvV", 3)],
            "t'": [("t3", "OOOVVV", 3)],
        }
        if spin_type == "R":
            notations["Q"] = [("t4a", "oooovvvv", 4), ("t4b", "oooovvvv", 4)]
        else:
            notations["Q"] = [("t4", "oooovvvv", 4)]
        notations["2"] = notations["S"] + notations["D"]
        notations["3"] = notations["2"] + notations["T"]
        notations["4"] = notations["3"] + notations["Q"]

        # Remove any perturbative corrections
        op = self.fermion_ansatz
        while "(" in op:
            start = op.index("(")
            end = op.index(")")
            op = op[:start]
            if (end + 1) < len(op):
                op += op[end + 1 :]

        # Check in order of longest to shortest string in case one
        # method name starts with a substring equal to the name of
        # another method
        for method_type in sorted(METHOD_TYPES, key=len)[::-1]:
            if op.startswith(method_type):
                op = op.replace(method_type, "", 1)
                break

        # If it's MP we only ever need to initialise second-order
        # amplitudes
        if method_type == "MP":
            op = "D"

        # Determine the ranks
        for key in sorted(notations.keys(), key=len)[::-1]:
            if key in op:
                ranks += notations[key]
                op = op.replace(key, "")

        # Check there are no duplicates
        if len(ranks) != len(set(ranks)):
            raise util.ModelNotImplemented("Duplicate ranks in %s" % self.fermion_ansatz)

        # Sort the ranks by the cluster operator dimension
        ranks = sorted(ranks, key=lambda x: x[2])

        return ranks

    def bosonic_cluster_ranks(self, spin_type="G"):
        """
        Get a list of cluster operator ranks for the bosonic space.

        Parameters
        ----------
        spin_type : str, optional
            Spin type of the cluster operator. Default value is `"G"`.

        Returns
        -------
        ranks : list of tuples
            List of cluster operator ranks, each element is a tuple
            containing the name, the slices and the rank.
        """

        ranks = []
        if not self.boson_ansatz:
            return ranks

        notations = {
            "S": [("s1", "b", 1)],
            "D": [("s2", "bb", 2)],
            "T": [("s3", "bbb", 3)],
        }
        notations["2"] = notations["S"] + notations["D"]
        notations["3"] = notations["2"] + notations["T"]

        # Remove any perturbative corrections
        op = self.boson_ansatz
        while "(" in op:
            start = op.index("(")
            end = op.index(")")
            op = op[:start]
            if (end + 1) < len(op):
                op += op[end + 1 :]

        # Determine the ranks
        for key in sorted(notations.keys(), key=len)[::-1]:
            if key in op:
                ranks += notations[key]
                op = op.replace(key, "")

        # Check there are no duplicates
        if len(ranks) != len(set(ranks)):
            raise util.ModelNotImplemented("Duplicate ranks in %s" % self.boson_ansatz)

        # Sort the ranks by the cluster operator dimension
        ranks = sorted(ranks, key=lambda x: x[2])

        return ranks

    def coupling_cluster_ranks(self, spin_type="G"):
        """
        Get a list of cluster operator ranks for the coupling between
        fermionic and bosonic spaces.

        Parameters
        ----------
        spin_type : str, optional
            Spin type of the cluster operator. Default value is `"G"`.

        Returns
        -------
        ranks : list of tuple
            List of cluster operator ranks, each element is a tuple
            containing the name, slice, fermionic rank and bosonic rank.
        """

        ranks = []

        for fermion_rank in range(1, self.fermion_coupling_rank + 1):
            for boson_rank in range(1, self.boson_coupling_rank + 1):
                name = f"u{fermion_rank}{boson_rank}"
                key = "b" * boson_rank + "o" * fermion_rank + "v" * fermion_rank
                ranks.append((name, key, fermion_rank, boson_rank))

        return ranks
