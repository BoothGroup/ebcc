"""Ansatz definition."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ebcc import METHOD_TYPES, util

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Literal, Optional

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
    "CCSDt'": ("CCSDt'", "", 0, 0),
    "CCSD-S-1-1": ("CCSD", "S", 1, 1),
    "CCSD-SD-1-1": ("CCSD", "SD", 1, 1),
    "CCSD-SD-1-2": ("CCSD", "SD", 1, 2),
}


def name_to_identifier(name: str) -> str:
    """Convert an ansatz name to an identifier.

    The identifier is used as for the filename of the module containing the generated equations,
    where the name may contain illegal characters.

    Args:
        name: Name of the ansatz.

    Returns:
        Identifier for the ansatz.

    Examples:
        >>> name_to_identifier("CCSD(T)")
        'CCSDxTx'
        >>> name_to_identifier("CCSD-SD-1-2")
        'CCSD_SD_1_2'
    """
    iden = name.replace("(", "x").replace(")", "x")
    iden = iden.replace("[", "y").replace("]", "y")
    iden = iden.replace("-", "_")
    iden = iden.replace("'", "p")
    return iden


def identifity_to_name(iden: str) -> str:
    """Convert an ansatz identifier to a name.

    Inverse operation of `name_to_identifier`.

    Args:
        iden: Identifier for the ansatz.

    Returns:
        Name of the ansatz.

    Examples:
        >>> identifier_to_name("CCSDxTx")
        'CCSD(T)'
        >>> identifier_to_name("CCSD_SD_1_2")
        'CCSD-SD-1-2'
    """
    name = iden.replace("-", "_")
    while "x" in name:
        name = name.replace("x", "(", 1).replace("x", ")", 1)
    while "y" in name:
        name = name.replace("y", "(", 1).replace("y", ")", 1)
    name = name.replace("p", "'")
    return name


class Ansatz:
    """Ansatz class."""

    def __init__(
        self,
        fermion_ansatz: str = "CCSD",
        boson_ansatz: str = "",
        fermion_coupling_rank: int = 0,
        boson_coupling_rank: int = 0,
        density_fitting: bool = False,
        module_name: Optional[str] = None,
    ) -> None:
        """Initialise the ansatz.

        Args:
            fermion_ansatz: Fermionic ansatz.
            boson_ansatz: Rank of bosonic excitations.
            fermion_coupling_rank: Rank of fermionic term in coupling.
            boson_coupling_rank: Rank of bosonic term in coupling.
            density_fitting: Use density fitting.
            module_name: Name of the module containing the generated equations.
        """
        self.fermion_ansatz = fermion_ansatz
        self.boson_ansatz = boson_ansatz
        self.fermion_coupling_rank = fermion_coupling_rank
        self.boson_coupling_rank = boson_coupling_rank
        self.density_fitting = density_fitting
        self.module_name = module_name

    def _get_eqns(self, prefix: str) -> ModuleType:
        """Get the module containing the generated equations."""
        if self.module_name is None:
            name = prefix + name_to_identifier(self.name)
        else:
            name = self.module_name
        return importlib.import_module("ebcc.codegen.%s" % name)

    @classmethod
    def from_string(cls, string: str, density_fitting: bool = False) -> Ansatz:
        """Build an `Ansatz` from a string for the default ansatzes.

        Args:
            string: Input string.
            density_fitting: Use density fitting.

        Returns:
            Ansatz object.
        """
        if string not in named_ansatzes:
            raise util.ModelNotImplemented(string)
        return cls(*named_ansatzes[string], density_fitting=density_fitting)

    def __repr__(self) -> str:
        """Get a string with the name of the method."""
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
    def name(self) -> str:
        """Get the name of the ansatz."""
        return repr(self)

    @property
    def has_perturbative_correction(self) -> bool:
        """Get a boolean indicating if the ansatz includes a perturbative correction e.g. CCSD(T).

        Returns:
            perturbative: Boolean indicating if the ansatz is perturbatively corrected.
        """
        return any(
            "(" in ansatz and ")" in ansatz for ansatz in (self.fermion_ansatz, self.boson_ansatz)
        )

    @property
    def is_one_shot(self) -> bool:
        """Get a boolean indicating whether the ansatz is a one-shot energy calculation e.g. MP2.

        Returns:
            one_shot: Boolean indicating if the ansatz is a one-shot energy calculation.
        """
        return all(
            ansatz.startswith("MP") or ansatz == ""
            for ansatz in (self.fermion_ansatz, self.boson_ansatz)
        )

    def fermionic_cluster_ranks(
        self,
        spin_type: str = "G",
        which: Literal["t", "l", "ip", "ea", "ee"] = "t",
    ) -> list[tuple[str, str, int]]:
        """Get a list of cluster operator ranks for the fermionic space.

        Args:
            spin_type: Spin type of the cluster operator.
            which: Type of cluster operator to return.

        Returns:
            List of cluster operator ranks, each element is a tuple containing the name, the slices
            and the rank.
        """
        ranks: list[tuple[str, str, int]] = []
        if not self.fermion_ansatz:
            return ranks

        def _adapt_key(key: str) -> str:
            """Adapt the key to the `which` argument."""
            if which == "ip":
                return key[:-1]
            if which == "ea":
                n = len(key) // 2
                key = key[n:] + key[: n - 1]
            if which == "l":
                n = len(key) // 2
                key = key[n:] + key[:n]
            return key

        symbol = which if which in ("t", "l") else "r"
        notations = {
            "S": [(f"{symbol}1", _adapt_key("ov"), 1)],
            "D": [(f"{symbol}2", _adapt_key("oovv"), 2)],
            "T": [(f"{symbol}3", _adapt_key("ooovvv"), 3)],
            "t": [(f"{symbol}3", _adapt_key("ooOvvV"), 3)],
            "t'": [(f"{symbol}3", _adapt_key("OOOVVV"), 3)],
        }
        if spin_type == "R":
            notations["Q"] = [(f"{symbol}4a", "oooovvvv", 4), (f"{symbol}4b", "oooovvvv", 4)]
        else:
            notations["Q"] = [(f"{symbol}4", "oooovvvv", 4)]
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
        if method_type == "MP" and which in ("t", "l"):
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

    def bosonic_cluster_ranks(
        self,
        spin_type: str = "G",
        which: Literal["t", "l", "ip", "ea", "ee"] = "t",
    ) -> list[tuple[str, str, int]]:
        """Get a list of cluster operator ranks for the bosonic space.

        Args:
            spin_type: Spin type of the cluster operator.
            which: Type of cluster operator.

        Returns:
            List of cluster operator ranks, each element is a tuple containing the name, the slices
            and the rank.
        """
        ranks: list[tuple[str, str, int]] = []
        if not self.boson_ansatz:
            return ranks

        symbol = "s" if which == "t" else "ls"
        notations = {
            "S": [(f"{symbol}1", "b", 1)],
            "D": [(f"{symbol}2", "bb", 2)],
            "T": [(f"{symbol}3", "bbb", 3)],
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

    def coupling_cluster_ranks(
        self,
        spin_type: str = "G",
        which: Literal["t", "l", "ip", "ea", "ee"] = "t",
    ) -> list[tuple[str, str, int, int]]:
        """Get a list of cluster operator ranks for the coupling between fermions and bosons.

        Args:
            spin_type: Spin type of the cluster operator.
            which: Type of cluster operator to return.

        Returns:
            List of cluster operator ranks, each element is a tuple containing the name, the slices
            and the rank.
        """

        def _adapt_key(key: str, fermion_rank: int, boson_rank: int) -> str:
            """Adapt the key to the `which` argument."""
            if which in ("ip", "ea", "ee"):
                raise util.ModelNotImplemented(
                    "Cluster ranks for coupling space not implemented for %s" % which
                )
            if which == "l":
                nf = fermion_rank
                nb = boson_rank
                key[:nb] + key[nb + nf :] + key[nb : nb + nf]
            return key

        symbol = "u" if which == "t" else "lu"

        ranks = []
        for fermion_rank in range(1, self.fermion_coupling_rank + 1):
            for boson_rank in range(1, self.boson_coupling_rank + 1):
                name = f"{symbol}{fermion_rank}{boson_rank}"
                key = _adapt_key(
                    "b" * boson_rank + "o" * fermion_rank + "v" * fermion_rank,
                    fermion_rank,
                    boson_rank,
                )
                ranks.append((name, key, fermion_rank, boson_rank))

        return ranks
