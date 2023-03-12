"""Tests for the GCCSD(T) model.
"""

import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from types import SimpleNamespace
from pyscf import cc, gto, lib, scf
from pyscf.cc import gccsd_t_lambda as ccsd_t_lambda
from pyscf.cc import gccsd_t_rdm as ccsd_t_rdm

from ebcc import GEBCC, NullLogger


@pytest.mark.reference
class GCCSD_T_PySCF_Tests(unittest.TestCase):
    """Test GCCSD(T) against the PySCF GCCSD(T) values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "He 0 0 0; He 0 0 0.5"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        mf = mf.to_ghf()

        ccsd_ref = cc.GCCSD(mf)
        ccsd_ref.conv_tol = 1e-12
        ccsd_ref.kernel()
        ccsd_ref.e_corr += ccsd_ref.ccsd_t()
        ccsd_ref.converged_lambda, ccsd_ref.l1, ccsd_ref.l2 = ccsd_t_lambda.kernel(
                ccsd_ref,
                ccsd_ref.ao2mo(),
                ccsd_ref.t1,
                ccsd_ref.t2,
                tol=1e-9,
                #verbose=0,
        )
        rdm1_ref = ccsd_t_rdm.make_rdm1(
                ccsd_ref,
                ccsd_ref.t1,
                ccsd_ref.t2,
                ccsd_ref.l1,
                ccsd_ref.l2,
                ccsd_ref.ao2mo(),
        )

        ccsd = GEBCC(
                mf,
                ansatz="CCSD(T)",
                #log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        ccsd.options.t_tol = 1e-9
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris
        cls.rdm1_ref = rdm1_ref

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd
        del cls.rdm1_ref

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd.converged_lambda)
        self.assertTrue(self.ccsd_ref.converged)
        self.assertTrue(self.ccsd_ref.converged_lambda)

    def test_ccsd_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 8)

    #def test_l1(self):
    #    a = self.ccsd_ref.l1.T
    #    b = self.ccsd.l1
    #    np.testing.assert_almost_equal(a, b, 6)

    #def test_rdm1(self):
    #    a = self.rdm1_ref
    #    b = self.ccsd.make_rdm1_f()
    #    np.testing.assert_almost_equal(a, b, 6)


if __name__ == "__main__":
    print("Tests for GCCSD(T)")
    unittest.main()

