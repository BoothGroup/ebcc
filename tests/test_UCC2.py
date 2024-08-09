"""Tests for the UCC2 model.
"""

import unittest

from pyscf import gto, scf, cc, lib
import numpy as np
import scipy.linalg
import pytest

from ebcc import UEBCC, NullLogger, util


@pytest.mark.reference
class UCC2_PySCF_Tests(unittest.TestCase):
    """Tests UCC2 against the PySCF values.
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

        ccsd_ref = cc.rccsd.RCCSD(mf)
        ccsd_ref.cc2 = True
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.conv_tol_normt = 1e-14
        ccsd_ref.max_cycle = 200
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = UEBCC(
                mf,
                ansatz="CC2",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
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
        np.testing.assert_almost_equal(a, b.aa, 6)
        np.testing.assert_almost_equal(a, b.bb, 6)

    def test_t2_amplitudes(self):
        a = self.ccsd_ref.t2
        b = self.ccsd.t2
        np.testing.assert_almost_equal(a, b.abab, 6)


@pytest.mark.regression
class UCC2_Tests(unittest.TestCase):
    """Test UCC2 against regression.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0.0 0.0 0.11779; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161"
        mol.spin = 2
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd = UEBCC(
                mf,
                ansatz="CC2",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd, cls.eris = mf, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd, cls.eris

    def test_rdm1_f(self):
        dm = self.ccsd.make_rdm1_f()
        ca, cb = self.ccsd.mf.mo_coeff
        dmaa = util.einsum("ij,pi,qj->pq", dm.aa, ca, ca)
        dmbb = util.einsum("ij,pi,qj->pq", dm.bb, cb, cb)
        self.assertAlmostEqual(lib.fp(dmaa), 0.9807657038537687, 6)
        self.assertAlmostEqual(lib.fp(dmbb), 1.7626762326382641, 6)

    def test_rdm2_f(self):
        dm = self.ccsd.make_rdm2_f()
        ca, cb = self.ccsd.mf.mo_coeff
        dmaaaa = util.einsum("ijkl,pi,qj,rk,sl->pqrs", dm.aaaa, ca, ca, ca, ca)
        dmaabb = util.einsum("ijkl,pi,qj,rk,sl->pqrs", dm.aabb, ca, ca, cb, cb)
        dmbbbb = util.einsum("ijkl,pi,qj,rk,sl->pqrs", dm.bbbb, cb, cb, cb, cb)
        self.assertAlmostEqual(lib.fp(dmaaaa), 0.708602503074179, 6)
        self.assertAlmostEqual(lib.fp(dmaabb), 1.617909309590978, 6)
        self.assertAlmostEqual(lib.fp(dmbbbb), 2.231837825282723, 6)


if __name__ == "__main__":
    print("Tests for UCC2")
    unittest.main()
