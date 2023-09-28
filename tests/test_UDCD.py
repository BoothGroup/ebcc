"""Tests for the UDCD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf

from ebcc import REBCC, UEBCC, NullLogger, Space


@pytest.mark.regression
class UDCD_Tests(unittest.TestCase):
    """Test UDCD against RDCD.
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

        rdcd = REBCC(
                mf,
                ansatz="DCD",
                log=NullLogger(),
        )
        rdcd.options.e_tol = 1e-10
        rdcd.kernel()

        rdcd = UEBCC.from_rebcc(rdcd)

        udcd = UEBCC(
                mf,
                ansatz="DCD",
                log=NullLogger(),
        )
        udcd.options.e_tol = 1e-10
        udcd.kernel()

        cls.mf, cls.rdcd, cls.udcd= mf, rdcd, udcd

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.rdcd, cls.udcd

    def test_converged(self):
        self.assertTrue(self.rdcd.converged)
        self.assertTrue(self.udcd.converged)

    def test_energy(self):
        self.assertAlmostEqual(self.rdcd.e_tot, self.udcd.e_tot, places=8)

    def test_t2_amplitudes(self):
        a = self.rdcd.t2
        b = self.udcd.t2
        np.testing.assert_allclose(a.aaaa, b.aaaa, atol=1e-7)
        np.testing.assert_allclose(a.bbbb, b.bbbb, atol=1e-7)
        np.testing.assert_allclose(a.abab, b.abab, atol=1e-7)


if __name__ == "__main__":
    print("Tests for UDCD")
    unittest.main()
