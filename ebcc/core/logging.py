"""Logging."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING, cast

from ebcc import __version__
from ebcc.core.tensor import MPI
from ebcc.util import Namespace

if TYPE_CHECKING:
    from typing import Any

HEADER = """        _
       | |
   ___ | |__    ___   ___
  / _ \| '_ \  / __| / __|
 |  __/| |_) || (__ | (__
  \___||_.__/  \___| \___|
%s"""  # noqa: W605


class Logger(logging.Logger):
    """Logger with a custom output level."""

    def __init__(self, name: str, level: int = logging.INFO) -> None:
        """Initialise the logger."""
        super().__init__(name, level)

    def output(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Output a message at the `"OUTPUT"` level."""
        if self.isEnabledFor(25):
            self._log(25, msg, args, **kwargs)


logging.setLoggerClass(Logger)
logging.addLevelName(25, "OUTPUT")


default_log = Logger("ebcc")
if MPI is None or MPI.COMM_WORLD.rank == 0:
    default_log.setLevel(logging.INFO)
else:
    default_log.setLevel(logging.ERROR)
default_log.addHandler(logging.StreamHandler(sys.stderr))


class NullLogger(Logger):
    """A logger that does nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the logger."""
        super().__init__("null")

    def _log(self, *args, **kwargs):  # type: ignore
        pass


def init_logging(log: Logger) -> None:
    """Initialise the logging with a header."""
    if globals().get("_EBCC_LOG_INITIALISED", False):
        return

    # Print header
    header_size = max([len(line) for line in HEADER.split("\n")])
    space = " " * (header_size - len(__version__))
    log.info(f"{ANSI.B}{HEADER}{ANSI.R}" % f"{space}{ANSI.B}{__version__}{ANSI.R}")

    # Print versions of dependencies and ebcc
    def get_git_hash(directory: str) -> str:
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

    # Environment variables
    log.info("OMP_NUM_THREADS = %s" % os.environ.get("OMP_NUM_THREADS", ""))

    log.debug("")

    globals()["_EBCC_LOG_INITIALISED"] = True


def _check_output(*args: Any, **kwargs: Any) -> bytes:
    """Call a command.

    If the return code is non-zero, an empty `bytes` object is returned.
    """
    try:
        return cast(bytes, subprocess.check_output(*args, **kwargs))
    except subprocess.CalledProcessError:
        return bytes()


ANSI = Namespace(
    B="\x1b[1m",
    H="\x1b[3m",
    R="\x1b[m\x0f",
    U="\x1b[4m",
    b="\x1b[34m",
    c="\x1b[36m",
    g="\x1b[32m",
    k="\x1b[30m",
    m="\x1b[35m",
    r="\x1b[31m",
    w="\x1b[37m",
    y="\x1b[33m",
)
