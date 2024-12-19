"""Tests for the ULCCSD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf

from ebcc import REBCC, UEBCC, NullLogger, Space


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

        rlccd = UEBCC.from_rebcc(rlccd)

        ulccd = UEBCC(
                mf,
                ansatz="LCCSD",
                log=NullLogger(),
        )
        ulccd.options.e_tol = 1e-10
        ulccd.kernel()

        cls.mf, cls.rlccd, cls.ulccd= mf, rlccd, ulccd

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.rlccd, cls.ulccd

    def test_converged(self):
        self.assertTrue(self.rlccd.converged)
        self.assertTrue(self.ulccd.converged)

    def test_energy(self):
        self.assertAlmostEqual(self.rlccd.e_tot, self.ulccd.e_tot, places=8)

    def test_t2_amplitudes(self):
        a = self.rlccd.t2
        b = self.ulccd.t2
        np.testing.assert_allclose(a.aaaa, b.aaaa, atol=1e-7)
        np.testing.assert_allclose(a.bbbb, b.bbbb, atol=1e-7)
        np.testing.assert_allclose(a.abab, b.abab, atol=1e-7)


if __name__ == "__main__":
    print("Tests for ULCCSD")
    unittest.main()
