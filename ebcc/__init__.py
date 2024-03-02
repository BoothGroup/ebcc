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

__version__ = "1.4.3"

import importlib
import logging
import os
import subprocess
import sys


# --- Get the tensor backend

import numpy

TENSOR_BACKEND = None
tensor_backend = None


def set_tensor_backend(backend):
    """
    Set the tensor backend. The desired backend will be imported and
    assigned to the global variable `tensor_backend`.

    Parameters
    ----------
    backend : str
        The name of the tensor backend to use.
    """

    global tensor_backend, TENSOR_BACKEND

    TENSOR_BACKEND = backend

    # Load the backend
    if backend == "":
        backend = "numpy"
    if backend == "jax":
        tensor_backend = importlib.import_module("jax.numpy")
    elif backend == "tblis":
        tensor_backend = importlib.import_module("numpy")
    elif backend in ("numpy", "ctf"):
        tensor_backend = importlib.import_module(backend)
    else:
        raise ValueError("Unsupported tensor backend: %s" % backend)

    # Monkey patch conversion operations
    if backend in ("numpy", "tblis"):
        tensor_backend.asnumpy = lambda array: array
        tensor_backend.astensor = lambda array: array
        tensor_backend.norm = lambda *args, **kwargs: tensor_backend.linalg.norm(*args, **kwargs)
    elif backend == "jax":
        # TODO precision control
        tensor_backend.asnumpy = lambda array: numpy.asarray(array)
        tensor_backend.astensor = lambda array: tensor_backend.asarray(array)
    elif backend == "ctf":
        tensor_backend.asnumpy = lambda array: array.to_nparray()


set_tensor_backend(os.environ.get("EBCC_TENSOR_BACKEND", "numpy"))

# --- Logging:


def output(self, msg, *args, **kwargs):
    """Output a message at the `"OUTPUT"` level."""
    if self.isEnabledFor(25):
        self._log(25, msg, args, **kwargs)


default_log = logging.getLogger(__name__)
default_log.setLevel(logging.INFO)
default_log.addHandler(logging.StreamHandler(sys.stderr))
logging.addLevelName(25, "OUTPUT")
logging.Logger.output = output


class NullLogger(logging.Logger):
    """A logger that does nothing."""

    def __init__(self, *args, **kwargs):
        super().__init__("null")

    def _log(self, level, msg, args, **kwargs):
        pass


HEADER = """        _
       | |
   ___ | |__    ___   ___
  / _ \| '_ \  / __| / __|
 |  __/| |_) || (__ | (__
  \___||_.__/  \___| \___|
%s"""


def init_logging(log):
    """Initialise the logging with a header."""

    if globals().get("_EBCC_LOG_INITIALISED", False):
        return

    # Print header
    header_size = max([len(line) for line in HEADER.split("\n")])
    log.info(HEADER % (" " * (header_size - len(__version__)) + __version__))

    # Print versions of dependencies and ebcc
    def get_git_hash(directory):
        git_directory = os.path.join(directory, ".git")
        cmd = ["git", "--git-dir=%s" % git_directory, "rev-parse", "--short", "HEAD"]
        try:
            git_hash = subprocess.check_output(
                cmd, universal_newlines=True, stderr=subprocess.STDOUT
            ).rstrip()
        except subprocess.CalledProcessError:
            git_hash = "N/A"
        return git_hash

    import numpy
    import pyscf

    log.info("numpy:")
    log.info(" > Version:  %s" % numpy.__version__)
    log.info(" > Git hash: %s" % get_git_hash(os.path.join(os.path.dirname(numpy.__file__), "..")))

    log.info("pyscf:")
    log.info(" > Version:  %s" % pyscf.__version__)
    log.info(" > Git hash: %s" % get_git_hash(os.path.join(os.path.dirname(pyscf.__file__), "..")))

    log.info("ebcc:")
    log.info(" > Version:  %s" % __version__)
    log.info(" > Git hash: %s" % get_git_hash(os.path.join(os.path.dirname(__file__), "..")))

    if TENSOR_BACKEND != "numpy":
        parent = sys.modules[tensor_backend.__name__.split(".")[0]]
        log.info("%s:" % parent.__name__)
        log.info(" > Version:  %s" % getattr(parent, "__version__", "N/A"))
        log.info(
            " > Git hash: %s" % get_git_hash(os.path.join(os.path.dirname(parent.__file__), ".."))
        )

    # Environment variables
    log.info("EBCC_TENSOR_BACKEND = %s" % os.environ.get("EBCC_TENSOR_BACKEND", ""))
    log.info("OMP_NUM_THREADS = %s" % os.environ.get("OMP_NUM_THREADS", ""))

    log.info("")

    globals()["_EBCC_LOG_INITIALISED"] = True


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
