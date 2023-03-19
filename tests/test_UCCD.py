"""Tests for the UCCD model.
"""

import unittest
import pytest

import numpy as np
from pyscf import cc, gto, lib, scf
from pyscf.cc import ccd as pyscf_ccd
from pyscf.cc import ccsd_rdm

from ebcc import REBCC, UEBCC, NullLogger, Space


@pytest.mark.reference
class UCCD_PySCF_Tests(unittest.TestCase):
    """Test UCCD against PySCF's RCCD values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.4"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccd_ref = pyscf_ccd.CCD(mf)
        ccd_ref.conv_tol = 1e-10
        ccd_ref.conv_tol_normt = 1e-14
        ccd_ref.max_cycle = 200
        ccd_ref.kernel()
        ccd_ref.solve_lambda()

        ccd = UEBCC(
                mf.to_uhf(),
                ansatz="CCD",
                log=NullLogger(),
        )
        ccd.options.e_tol = 1e-10
        eris = ccd.get_eris()
        ccd.kernel(eris=eris)
        ccd.solve_lambda(eris=eris)

        cls.mf, cls.ccd_ref, cls.ccd, cls.eris = mf, ccd_ref, ccd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccd_ref, cls.ccd, cls.eris

    def test_converged(self):
        self.assertTrue(self.ccd.converged)
        self.assertTrue(self.ccd.converged_lambda)
        self.assertTrue(self.ccd_ref.converged)
        self.assertTrue(self.ccd_ref.converged_lambda)

    def test_energy(self):
        a = self.ccd_ref.e_tot
        b = self.ccd.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_t2_amplitudes(self):
        a = self.ccd_ref.t2
        b = self.ccd.t2.abab
        np.testing.assert_almost_equal(a, b, 6)

    def test_l2_amplitudes(self):
        a = self.ccd_ref.l2
        b = self.ccd.l2.abab.transpose(2, 3, 0, 1)
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm1(self):
        a = self.ccd_ref.make_rdm1()
        b = self.ccd.make_rdm1_f(eris=self.eris)
        b = b.aa + b.bb
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_rdm2(self):
        a = self.ccd_ref.make_rdm2()
        b = self.ccd.make_rdm2_f(eris=self.eris)
        b = 2 * b.aabb + b.aaaa + b.bbbb
        np.testing.assert_almost_equal(a, b, 6, verbose=True)


if __name__ == "__main__":
    print("Tests for UCCD")
    unittest.main()
