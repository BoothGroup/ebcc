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

from __future__ import annotations

"""Version of the package."""
__version__ = "1.6.1"

"""List of supported ansatz types."""
METHOD_TYPES = ["MP", "CC", "LCC", "QCI", "QCC", "DC"]

import importlib
import os
import sys
from typing import TYPE_CHECKING

"""Backend to use for NumPy operations."""
BACKEND: str = os.environ.get("EBCC_BACKEND", "numpy")

if TYPE_CHECKING:
    # Import NumPy directly for type-checking purposes
    import numpy
else:
    numpy = importlib.import_module(f"ebcc.backend._{BACKEND}")

if BACKEND == "jax" and not TYPE_CHECKING:
    # Try to import pyscfad if JAX is used so we can use automatic differentiation
    try:
        import pyscfad as pyscf

        # See https://github.com/fishjojo/pyscfad/issues/60
        from pyscfad import scf, ao2mo, mp
        from pyscfad.ao2mo import _ao2mo, addons

        pyscf.scf = scf
        pyscf.ao2mo = ao2mo
        pyscf.ao2mo._ao2mo = _ao2mo
        pyscf.ao2mo.addons = addons
        pyscf.mp = mp

    except ImportError:
        import pyscf
else:
    import pyscf

from ebcc.core.logging import NullLogger, default_log, init_logging
from ebcc.cc import GEBCC, REBCC, UEBCC
from ebcc.core import precision
from ebcc.core.ansatz import Ansatz
from ebcc.ham.space import Space
from ebcc.opt import BruecknerGEBCC, BruecknerREBCC, BruecknerUEBCC

if TYPE_CHECKING:
    from typing import Any, Callable

    from pyscf.scf.hf import SCF

    from ebcc.cc.base import BaseEBCC


sys.modules["ebcc.precision"] = precision  # Compatibility with older codegen versions


def EBCC(mf: SCF, *args: Any, **kwargs: Any) -> BaseEBCC:
    """Construct an EBCC object for the given mean-field object."""
    from pyscf import scf

    if isinstance(mf, scf.uhf.UHF):
        return UEBCC(mf, *args, **kwargs)
    elif isinstance(mf, scf.ghf.GHF):
        return GEBCC(mf, *args, **kwargs)
    else:
        return REBCC(mf, *args, **kwargs)


EBCC.__doc__ = REBCC.__doc__


def _factory(ansatz: str) -> Callable[[SCF, Any, Any], BaseEBCC]:
    """Constructor for some specific ansatz."""
    from pyscf import scf

    def constructor(mf: SCF, *args: Any, **kwargs: Any) -> BaseEBCC:
        """Construct an EBCC object for the given mean-field object."""
        kwargs["ansatz"] = ansatz

        if isinstance(mf, scf.uhf.UHF):
            return UEBCC(mf, *args, **kwargs)
        elif isinstance(mf, scf.ghf.GHF):
            return GEBCC(mf, *args, **kwargs)
        else:
            return REBCC(mf, *args, **kwargs)

    return constructor


CCSD = _factory("CCSD")
CCSDT = _factory("CCSDT")
CC2 = _factory("CC2")
CC3 = _factory("CC3")

del _factory


def available_models(
    verbose: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:  # pragma: no cover
    """List available coupled-cluster models for each spin type."""
    cd = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(cd, "codegen")
    _, _, files = list(os.walk(path))[0]

    from ebcc.core.ansatz import identifier_to_name

    rhf = []
    uhf = []
    ghf = []

    for f in files:
        if f.endswith(".py"):
            f = f.rstrip(".py")
            f = f.replace("_", "-")
            f = identifier_to_name(f)
            if f.startswith("R"):
                rhf.append(f)
            elif f.startswith("U"):
                uhf.append(f)
            elif f.startswith("G"):
                ghf.append(f)

    rhf = sorted(rhf)
    uhf = sorted(uhf)
    ghf = sorted(ghf)

    if verbose:
        sys.stderr.write("RHF:\n  %s\n" % ", ".join(rhf))
        sys.stderr.write("UHF:\n  %s\n" % ", ".join(uhf))
        sys.stderr.write("GHF:\n  %s\n" % ", ".join(ghf))

    return tuple(rhf), tuple(uhf), tuple(ghf)
