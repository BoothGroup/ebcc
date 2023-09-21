"""Tests for the RDFCCSD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf, cc, lib

from ebcc import REBCC, NullLogger, Space


@pytest.mark.reference
class RDFCCSD_PySCF_Tests(unittest.TestCase):
    """Test RDFCCSD against PySCF values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0.0 0.0 0.11779; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf = mf.density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd_ref = cc.CCSD(mf)
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.conv_tol_normt = 1e-14
        ccsd_ref.max_cycle = 200
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = REBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd.converged_lambda)
        self.assertTrue(self.ccsd_ref.converged)
        self.assertTrue(self.ccsd_ref.converged_lambda)

    def test_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.ccsd_ref.t1
        b = self.ccsd.t1
        np.testing.assert_almost_equal(a, b, 6)

    def test_t2_amplitudes(self):
        a = self.ccsd_ref.t2
        b = self.ccsd.t2
        np.testing.assert_almost_equal(a, b, 6)

    def test_l1_amplitudes(self):
        a = self.ccsd_ref.l1
        b = self.ccsd.l1.T
        np.testing.assert_almost_equal(a, b, 6)

    def test_l2_amplitudes(self):
        a = self.ccsd_ref.l2
        b = self.ccsd.l2.transpose(2, 3, 0, 1)
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm1(self):
        a = self.ccsd_ref.make_rdm1()
        b = self.ccsd.make_rdm1_f(eris=self.eris)
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2()
        b = self.ccsd.make_rdm2_f(eris=self.eris)
        np.testing.assert_almost_equal(a, b, 6, verbose=True)


@pytest.mark.reference
class RDFCCSD_PySCF_Frozen_Tests(unittest.TestCase):
    """Test RDFCCSD against the PySCF values with frozen orbitals.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0.0 0.0 0.11779; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf = mf.density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()

        frozen = np.zeros_like(mf.mo_occ)
        frozen[:2] = 1
        frozen[-1] = 1
        frozen = frozen.astype(bool)

        ccsd_ref = cc.CCSD(mf)
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.conv_tol_normt = 1e-14
        ccsd_ref.max_cycle = 200
        ccsd_ref.frozen = np.where(frozen)[0]
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        space = Space(
                mf.mo_occ > 0,
                frozen,
                np.zeros_like(mf.mo_occ),
        )

        ccsd = REBCC(
                mf,
                ansatz="CCSD",
                space=space,
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-13
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd.converged_lambda)
        self.assertTrue(self.ccsd_ref.converged)
        self.assertTrue(self.ccsd_ref.converged_lambda)

    def test_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.ccsd_ref.t1
        b = self.ccsd.t1
        np.testing.assert_almost_equal(a, b, 6)

    def test_t2_amplitudes(self):
        a = self.ccsd_ref.t2
        b = self.ccsd.t2
        np.testing.assert_almost_equal(a, b, 6)

    def test_l1_amplitudes(self):
        a = self.ccsd_ref.l1
        b = self.ccsd.l1.T
        np.testing.assert_almost_equal(a, b, 6)

    def test_l2_amplitudes(self):
        a = self.ccsd_ref.l2
        b = self.ccsd.l2.transpose(2, 3, 0, 1)
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm1(self):
        a = self.ccsd_ref.make_rdm1(with_frozen=False)
        b = self.ccsd.make_rdm1_f(eris=self.eris)
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2(with_frozen=False)
        b = self.ccsd.make_rdm2_f(eris=self.eris)
        np.testing.assert_almost_equal(a, b, 6, verbose=True)


if __name__ == "__main__":
    print("Tests for RDFCCSD")
    unittest.main()
