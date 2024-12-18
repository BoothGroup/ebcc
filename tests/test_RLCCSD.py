"""Tests for the RLCCSD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf

from ebcc import REBCC, NullLogger, Space


@pytest.mark.reference
class RLCCSD_Tests(unittest.TestCase):
    """Test RLCCSD against reference values.

    Reference values from PyBEST 2.0.0.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "H 0 0 0; F 0 0 1"
        #mol.unit = "B"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        lccd = REBCC(
                mf,
                ansatz="LCCSD",
                log=NullLogger(),
        )
        lccd.options.e_tol = 1e-10
        lccd.kernel()

        cls.mf, cls.lccd = mf, lccd

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.lccd

    def test_converged(self):
        self.assertTrue(self.lccd.converged)

    def test_energy(self):
        self.assertAlmostEqual(self.lccd.e_corr, -0.21451573243690, 6)
        self.assertAlmostEqual(self.lccd.e_tot, -100.2243857655389, 6)


if __name__ == "__main__":
    print("Tests for RLCCSD")
    unittest.main()
