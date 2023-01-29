"""Tests for the RCC2 model.
"""

import itertools
import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import cc, gto, lib, scf

from ebcc import REBCC, NullLogger, util


@pytest.mark.reference
class RCC2_PySCF_Tests(unittest.TestCase):
    """Test RCC2 against the PySCF values.
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

        ccsd = REBCC(
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
        del cls.mf, cls.ccsd_ref, cls.ccsd

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


@pytest.mark.regression
class RCC2_Tests(unittest.TestCase):
    """Test RCC2 against regression.
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

        ccsd = REBCC(
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
        c = self.mf.mo_coeff
        dm = util.einsum("ij,pi,qj->pq", dm, c, c)
        self.assertAlmostEqual(lib.fp(dm), 3.572563325863767, 6)

    def test_rdm2_f(self):
        dm = self.ccsd.make_rdm2_f()
        c = self.mf.mo_coeff
        dm = util.einsum("ijkl,pi,qj,rk,sl->pqrs", dm, c, c, c, c)
        self.assertAlmostEqual(lib.fp(dm), 6.475456720894991, 6)


if __name__ == "__main__":
    print("Tests for RCC2")
    unittest.main()
