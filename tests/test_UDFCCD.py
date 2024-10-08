"""Tests for the UDFCCD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf, lib

from ebcc import UEBCC, NullLogger, Space, BACKEND


@pytest.mark.regression
class UDFCCD_Tests(unittest.TestCase):
    """Test UDFCCD against UCCD.
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

        ccd_ref = UEBCC(
                mf,
                ansatz="CCD",
                log=NullLogger(),
        )
        ccd_ref.options.e_tol = 1e-10
        eris = ccd_ref.get_eris()
        ccd_ref.kernel(eris=eris)
        ccd_ref.solve_lambda(eris=eris)

        mf = mf.density_fit(auxbasis="aug-cc-pvqz-ri")

        ccd = UEBCC(
                mf,
                ansatz="CCD",
                log=NullLogger(),
        )
        ccd.options.e_tol = 1e-10
        eris = ccd.get_eris()
        ccd.kernel(eris=eris)
        ccd.solve_lambda(eris=eris)

        cls.mf, cls.ccd_ref, cls.ccd = mf, ccd_ref, ccd

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.ccd_ref, cls.ccd

    def test_converged(self):
        self.assertTrue(self.ccd.converged)
        self.assertTrue(self.ccd.converged_lambda)
        self.assertTrue(self.ccd_ref.converged)
        self.assertTrue(self.ccd_ref.converged_lambda)

    def test_energy(self):
        a = self.ccd_ref.e_tot
        b = self.ccd.e_tot
        self.assertAlmostEqual(a, b, 5)

    def test_t2_amplitudes(self):
        a = self.ccd_ref.t2.aaaa
        b = self.ccd.t2.aaaa
        self.assertAlmostEqual(np.max(np.abs(a - b)), 0.0, 4)

    def test_l2_amplitudes(self):
        a = self.ccd_ref.l2.aaaa
        b = self.ccd.l2.aaaa
        self.assertAlmostEqual(np.max(np.abs(a - b)), 0.0, 4)

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ip(self):
        e1 = self.ccd.ip_eom(nroots=5).kernel()
        e2 = self.ccd_ref.ip_eom(nroots=5).kernel()
        self.assertAlmostEqual(e1[0], e2[0], 5)

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ea(self):
        e1 = self.ccd.ea_eom(nroots=5).kernel()
        e2 = self.ccd_ref.ea_eom(nroots=5).kernel()
        self.assertAlmostEqual(e1[0], e2[0], 5)


if __name__ == "__main__":
    print("Tests for UDFCCD")
    unittest.main()
