"""
************************************************************
ebcc: Coupled cluster calculations on electron-boson systems
************************************************************

The `ebcc`  package implements various coupled cluster (CC) models
for application to electron-boson systems, with a focus on
generality and model extensibility.


Installation
------------

From the python package index:

    pip install ebcc

From source:

    git clone https://github.com/BoothGroup/ebcc
    pip install .


Usage
-----

The implemented models are built upon the mean-field objects of
`PySCF <https://github.com/pyscf/pyscf>`_:

>>> from pyscf import gto, scf
>>> from ebcc import EBCC
>>> mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="cc-pvdz")
>>> mf = scf.RHF(mol)
>>> mf.kernel()
>>> ccsd = EBCC(mf)
>>> ccsd.kernel()


Code generation
---------------

The models implemented are generated algorithmically from expressions
over second quantized operators. Expressions are generated using
`qwick <https://github.com/obackhouse/qwick>`_ with optimisation of
common subexpressions and contraction order achieved using
`drudge <https://github.com/tschijnmo/drudge>`_ and
`gristmill <https://github.com/tschijnmo/gristmill>`_.

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
        return UEBCC(mf, *args, **kwargs)
    elif isinstance(mf, scf.ghf.GHF):
        return GEBCC(mf, *args, **kwargs)
    else:
        return REBCC(mf, *args, **kwargs)

EBCC.__doc__ = REBCC.__doc__


# --- Constructors for boson-free calculations:

def CCSD(mf, *args, **kwargs):
    from pyscf import scf

    kwargs["fermion_excitations"] = "SD"
    kwargs["boson_excitations"] = "SD"
    kwargs["fermion_coupling_rank"] = 0
    kwargs["boson_coupling_rank"] = 0

    if isinstance(mf, scf.uhf.UHF):
        return UEBCC(mf, *args, **kwargs)
    elif isinstance(mf, scf.ghf.GHF):
        return GEBCC(mf, *args, **kwargs)
    else:
        return REBCC(mf, *args, **kwargs)

CCSD.__doc__ = REBCC.__doc__


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
