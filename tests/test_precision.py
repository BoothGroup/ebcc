"""Tests for precision control.
"""

import unittest
import pytest

import numpy as np
from pyscf import gto, scf

from ebcc import REBCC, UEBCC, GEBCC, NullLogger


@pytest.mark.regression
class REBCC_Precision_Tests(unittest.TestCase):
    """Test single-precision REBCC against double-precision results.
    """

    basis = "cc-pvdz"
    EBCC = REBCC

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = cls.basis
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        cls.mol, cls.mf = mol, mf

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_ansatzes(self):
        for ansatz in [
            "CC2",
            "CCSD",
            "CC3",
            "QCISD",
        ]:
            sp = self.EBCC(self.mf, ansatz=ansatz, log=NullLogger())
            sp.e_tol = 1e-7
            sp.t_tol = 1e-4
            sp.kernel()

            dp = self.EBCC(self.mf, ansatz=ansatz, log=NullLogger())
            dp.e_tol = 1e-8
            dp.t_tol = 1e-5
            dp.kernel()

            self.assertAlmostEqual(sp.e_tot, dp.e_tot, 6)


@pytest.mark.regression
class UEBCC_Precision_Tests(REBCC_Precision_Tests):
    """Test single-precision UEBCC against double-precision results.
    """

    basis = "cc-pvdz"
    EBCC = UEBCC


@pytest.mark.regression
class GEBCC_Precision_Tests(REBCC_Precision_Tests):
    """Test single-precision GEBCC against double-precision results.
    """

    basis = "sto3g"
    EBCC = GEBCC


if __name__ == "__main__":
    print("Tests for precision")
    unittest.main()
