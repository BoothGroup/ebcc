"""Tests for the UDCSD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf

from ebcc import REBCC, UEBCC, NullLogger, Space


@pytest.mark.regression
class UDCSD_Tests(unittest.TestCase):
    """Test UDCSD against RDCSD.
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

        rdcsd = REBCC(
                mf,
                ansatz="DCSD",
                log=NullLogger(),
        )
        rdcsd.options.e_tol = 1e-10
        rdcsd.kernel()

        rdcsd = UEBCC.from_rebcc(rdcsd)

        udcsd = UEBCC(
                mf,
                ansatz="DCSD",
                log=NullLogger(),
        )
        udcsd.options.e_tol = 1e-10
        udcsd.kernel()

        cls.mf, cls.rdcsd, cls.udcsd= mf, rdcsd, udcsd

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.rdcsd, cls.udcsd

    def test_converged(self):
        self.assertTrue(self.rdcsd.converged)
        self.assertTrue(self.udcsd.converged)

    def test_energy(self):
        self.assertAlmostEqual(self.rdcsd.e_tot, self.udcsd.e_tot, places=8)

    def test_t1_amplitudes(self):
        a = self.rdcsd.t1
        b = self.udcsd.t1
        np.testing.assert_allclose(a.aa, b.aa, atol=1e-7)
        np.testing.assert_allclose(a.bb, b.bb, atol=1e-7)

    def test_t2_amplitudes(self):
        a = self.rdcsd.t2
        b = self.udcsd.t2
        np.testing.assert_allclose(a.aaaa, b.aaaa, atol=1e-7)
        np.testing.assert_allclose(a.bbbb, b.bbbb, atol=1e-7)
        np.testing.assert_allclose(a.abab, b.abab, atol=1e-7)


if __name__ == "__main__":
    print("Tests for UDCSD")
    unittest.main()
