"""Tests for the ULCCSD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf

from ebcc import REBCC, GEBCC, NullLogger, Space, util


@pytest.mark.regression
class ULCCSD_Tests(unittest.TestCase):
    """Test ULCCSD against RLCCSD.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "N 0 0 0; N 0 0 2.2"
        mol.unit = "B"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        rlccd = REBCC(
                mf,
                ansatz="LCCSD",
                log=NullLogger(),
        )
        rlccd.options.e_tol = 1e-10
        rlccd.kernel()

        rlccd = GEBCC.from_rebcc(rlccd)

        glccd = GEBCC(
                mf,
                ansatz="LCCSD",
                log=NullLogger(),
        )
        glccd.options.e_tol = 1e-10
        glccd.kernel()

        cls.mf, cls.rlccd, cls.glccd= mf, rlccd, glccd

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.rlccd, cls.glccd

    def test_converged(self):
        self.assertTrue(self.rlccd.converged)
        self.assertTrue(self.glccd.converged)

    def test_energy(self):
        self.assertAlmostEqual(self.rlccd.e_tot, self.glccd.e_tot, places=8)


if __name__ == "__main__":
    print("Tests for GLCCSD")
    unittest.main()
