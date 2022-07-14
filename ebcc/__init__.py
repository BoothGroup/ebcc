"""
************************************************************
ebcc: Coupled cluster calculations on electron-boson systems
************************************************************

Quickstart
----------
`ebcc` builds upon the PySCF quantum chemistry package to expose
various coupled cluster (CC) models to systems consisting of coupled
electrons and bosons. Input to the solvers are PySCF mean-field
objects::

    >>> from pyscf import gto, scf
    >>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="cc-pvdz")
    >>> mf = scf.RHF(mol).run()
    >>> ccsd = ebcc.EBCC(mf)
    >>> ccsd.kernel()

"""

__version__ = "1.0.0a"

import os
import sys
import logging


# --- Logging:

def output(self, msg, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, msg, args, **kwargs)


default_log = logging.getLogger(__name__)
default_log.setLevel(logging.INFO)
default_log.addHandler(logging.StreamHandler(sys.stderr))
logging.addLevelName(25, "OUTPUT")
logging.Logger.output = output


class NullLogger(logging.Logger):
    def __init__(self, *args, **kwargs):
        super().__init__("null")

    def _log(self, level, msg, args, **kwargs):
        pass


# --- General constructor:

from ebcc.rebcc import REBCC
from ebcc.uebcc import UEBCC
from ebcc.gebcc import GEBCC

def EBCC(mf, *args, **kwargs):
    from pyscf import scf

    if isinstance(mf, scf.uhf.UHF):
        return UEBCC(mf)
    elif isinstance(mf, scf.ghf.GHF):
        return GEBCC(mf)
    else:
        return REBCC(mf)

EBCC.__doc__ = REBCC.__doc__


# --- List available methods:

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
