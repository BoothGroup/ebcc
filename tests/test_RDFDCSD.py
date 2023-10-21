"""Tests for the RDFDCSD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf, lib

from ebcc import REBCC, NullLogger, Space


@pytest.mark.regression
class RDFDCSD_Tests(unittest.TestCase):
    """Test RDFDCSD against RDCSD.
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

        ccsd_ref = REBCC(
                mf,
                ansatz="DCSD",
                log=NullLogger(),
        )
        ccsd_ref.options.e_tol = 1e-10
        eris = ccsd_ref.get_eris()
        ccsd_ref.kernel(eris=eris)

        mf = mf.density_fit(auxbasis="aug-cc-pvqz-ri")

        ccsd = REBCC(
                mf,
                ansatz="DCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd = mf, ccsd_ref, ccsd

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd_ref.converged)

    def test_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 5)

    def test_t1_amplitudes(self):
        a = self.ccsd_ref.t1
        b = self.ccsd.t1
        np.testing.assert_almost_equal(a, b, 4)

    def test_t2_amplitudes(self):
        a = self.ccsd_ref.t2
        b = self.ccsd.t2
        np.testing.assert_almost_equal(a, b, 4)


if __name__ == "__main__":
    print("Tests for RDFDCSD")
    unittest.main()
