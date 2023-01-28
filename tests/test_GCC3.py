"""Tests for the GCC3 model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, lib, scf

from ebcc import GEBCC, NullLogger


@pytest.mark.regression
class GCC3_Tests(unittest.TestCase):
    """Test GCC3 against regression.
    """

    # TODO proper tests against some other code

    pass


if __name__ == "__main__":
    print("Tests for GCC3")
    unittest.main()
