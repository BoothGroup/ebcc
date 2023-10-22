"""Tests for the UDFCC2 model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf, lib

from ebcc import UEBCC, NullLogger, Space


@pytest.mark.regression
class UDFCC2_Tests(unittest.TestCase):
    """Test UDFCC2 against UCC2.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0.0 0.0 0.11779; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        cc2_ref = UEBCC(
                mf,
                ansatz="CC2",
                log=NullLogger(),
        )
        cc2_ref.options.e_tol = 1e-10
        eris = cc2_ref.get_eris()
        cc2_ref.kernel(eris=eris)
        cc2_ref.solve_lambda(eris=eris)

        mf = mf.density_fit(auxbasis="aug-cc-pvqz-ri")

        cc2 = UEBCC(
                mf,
                ansatz="CC2",
                log=NullLogger(),
        )
        cc2.options.e_tol = 1e-10
        eris = cc2.get_eris()
        cc2.kernel(eris=eris)
        cc2.solve_lambda(eris=eris)

        cls.mf, cls.cc2_ref, cls.cc2 = mf, cc2_ref, cc2

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.cc2_ref, cls.cc2

    def test_converged(self):
        self.assertTrue(self.cc2.converged)
        self.assertTrue(self.cc2.converged_lambda)
        self.assertTrue(self.cc2_ref.converged)
        self.assertTrue(self.cc2_ref.converged_lambda)

    def test_energy(self):
        a = self.cc2_ref.e_tot
        b = self.cc2.e_tot
        self.assertAlmostEqual(a, b, 5)

    def test_t1_amplitudes(self):
        a = self.cc2_ref.t1.aa
        b = self.cc2.t1.aa
        np.testing.assert_almost_equal(a, b, 4)

    def test_t2_amplitudes(self):
        a = self.cc2_ref.t2.aaaa
        b = self.cc2.t2.aaaa
        np.testing.assert_almost_equal(a, b, 4)

    def test_l1_amplitudes(self):
        a = self.cc2_ref.l1.aa
        b = self.cc2.l1.aa
        np.testing.assert_almost_equal(a, b, 4)

    def test_l2_amplitudes(self):
        a = self.cc2_ref.l2.aaaa
        b = self.cc2.l2.aaaa
        np.testing.assert_almost_equal(a, b, 4)

    def test_rdm1(self):
        a = self.cc2_ref.make_rdm1_f().aa
        b = self.cc2.make_rdm1_f().aa
        np.testing.assert_almost_equal(a, b, 4, verbose=True)

    def test_rdm2(self):
        a = self.cc2_ref.make_rdm2_f().aaaa
        b = self.cc2.make_rdm2_f().aaaa
        np.testing.assert_almost_equal(a, b, 4, verbose=True)


if __name__ == "__main__":
    print("Tests for UDFCC2")
    unittest.main()
