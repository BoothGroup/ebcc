import os

from ebcc import util
from ebcc.rebcc import REBCC
from ebcc.gebcc import GEBCC


def available_methods():
    """List available coupled-cluster models for each of general (G),
    restricted (R) and unrestricted (U) Hartree--Fock references.
    """

    cd = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(cd, "codegen")
    _, _, files = list(os.walk(path))[0]

    rhf = []
    uhf = []
    ghf = []

    for f in files:
        if f.endswith(".py"):
            f = f.rstrip(".py")
            f = f.replace("_", "-")
            if f.startswith("RCC"):
                rhf.append(f)
            elif f.startswith("UCC"):
                uhf.append(f)
            elif f.startswith("GCC"):
                ghf.append(f)

    rhf = sorted(rhf)
    uhf = sorted(uhf)
    ghf = sorted(ghf)

    return tuple(rhf), tuple(uhf), tuple(ghf)
