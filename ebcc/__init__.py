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
"""

__version__ = "1.4.4"

import os
import sys


# --- Import NumPy here to allow drop-in replacements

import numpy


# --- Logging:

from ebcc.logging import default_log, init_logging, NullLogger


# --- Types of ansatz supporting by the EBCC solvers:

METHOD_TYPES = ["MP", "CC", "LCC", "QCI", "QCC", "DC"]


# --- General constructor:

from ebcc.gebcc import GEBCC
from ebcc.rebcc import REBCC
from ebcc.uebcc import UEBCC


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


def _factory(ansatz):
    def constructor(mf, *args, **kwargs):
        from pyscf import scf

        kwargs["ansatz"] = ansatz

        if isinstance(mf, scf.uhf.UHF):
            cc = UEBCC(mf, *args, **kwargs)
        elif isinstance(mf, scf.ghf.GHF):
            cc = GEBCC(mf, *args, **kwargs)
        else:
            cc = REBCC(mf, *args, **kwargs)

        cc.__doc__ = REBCC.__doc__

        return cc

    return constructor


CCSD = _factory("CCSD")
CCSDT = _factory("CCSDT")
CC2 = _factory("CC2")
CC3 = _factory("CC3")

del _factory


# --- Other imports:

from ebcc.ansatz import Ansatz
from ebcc.brueckner import BruecknerGEBCC, BruecknerREBCC, BruecknerUEBCC
from ebcc.space import Space


# --- List available methods:


def available_models(verbose=True):  # pragma: no cover
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

    if verbose:
        sys.stderr.write("RHF:\n  %s\n" % ", ".join(rhf))
        sys.stderr.write("UHF:\n  %s\n" % ", ".join(uhf))
        sys.stderr.write("GHF:\n  %s\n" % ", ".join(ghf))

    return tuple(rhf), tuple(uhf), tuple(ghf)
