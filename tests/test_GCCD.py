"""Tests for the GCCD model.
"""

import unittest
import pytest

import numpy as np
from pyscf import cc, gto, lib, scf
from pyscf.cc import ccd as pyscf_ccd
from pyscf.cc import ccsd_rdm

from ebcc import GEBCC, NullLogger, Space, util


@pytest.mark.reference
class GCCD_PySCF_Tests(unittest.TestCase):
    """Test GCCD against PySCF's RCCD values.
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

        ccd = GEBCC(
                mf.to_ghf(),
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
        del cls.mf, cls.ccd_ref, cls.ccd

    def test_converged(self):
        self.assertTrue(self.ccd.converged)
        self.assertTrue(self.ccd.converged_lambda)
        self.assertTrue(self.ccd_ref.converged)
        self.assertTrue(self.ccd_ref.converged_lambda)

    def test_energy(self):
        a = self.ccd_ref.e_tot
        b = self.ccd.e_tot
        self.assertAlmostEqual(a, b, 7)


@pytest.mark.regression
class GCCD_Tests(unittest.TestCase):
    """Test GCCD against regression.
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

        ccd = GEBCC(
                mf.to_ghf(),
                ansatz="CCD",
                log=NullLogger(),
        )
        ccd.options.e_tol = 1e-10
        eris = ccd.get_eris()
        ccd.kernel(eris=eris)
        ccd.solve_lambda(eris=eris)

        cls.mf, cls.ccd, cls.eris = mf, ccd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccd, cls.eris

    def test_converged(self):
        self.assertTrue(self.ccd.converged)
        self.assertTrue(self.ccd.converged_lambda)

    def test_t2(self):
        ci = self.ccd.mo_coeff[:, self.ccd.mo_occ > 0]
        ca = self.ccd.mo_coeff[:, self.ccd.mo_occ == 0]
        t2 = util.einsum("ijab,pi,qj,ra,sb->pqrs", self.ccd.t2, ci, ci, ca, ca)
        self.assertAlmostEqual(lib.fp(t2), -0.0006811985222445749)

    def test_l2(self):
        ci = self.ccd.mo_coeff[:, self.ccd.mo_occ > 0]
        ca = self.ccd.mo_coeff[:, self.ccd.mo_occ == 0]
        l2 = util.einsum("abij,pa,qb,ri,sj->pqrs", self.ccd.l2, ca, ca, ci, ci)
        self.assertAlmostEqual(lib.fp(l2), 0.5045971727461588)

    def test_rdm1(self):
        c = self.ccd.mo_coeff
        rdm1 = util.einsum("ij,pi,qj->pq", self.ccd.make_rdm1_f(), c, c)
        self.assertAlmostEqual(lib.fp(rdm1), 1.9756354423470504)

    def test_rdm2(self):
        c = self.ccd.mo_coeff
        rdm2 = util.einsum("ijkl,pi,qj,rk,sl->pqrs", self.ccd.make_rdm2_f(), c, c, c, c)
        self.assertAlmostEqual(lib.fp(rdm2), 0.21661134391721626)


if __name__ == "__main__":
    print("Tests for GCCD")
    unittest.main()
