"""Tests for the GCCSD-S-1-1 model.
"""

import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import cc, gto, lib, scf

from ebcc import GEBCC, NullLogger


@pytest.mark.reference
class GCCSD_S_1_1_Tests(unittest.TestCase):
    """Test GCCSD-S-1-1 against the legacy GCCSD-S-1-1 values with
    shift=False.
    """

    shift = True

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
            mo_coeff = data["mo_coeff"]
            data = data[(2, 1, 1)]

        mol = gto.Mole()
        mol.atom = "H 0 0 0; F 0 0 1.1"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        mf.mo_coeff = mo_coeff

        nmo = mf.mo_occ.size
        nbos = 5
        np.random.seed(12345)
        g_ = np.random.random((nbos, nmo, nmo)) * 0.02
        g_ = 0.5 * (g_ + g_.transpose(0, 2, 1).conj())
        omega = np.random.random((nbos,)) * 5.0

        orbspin = scf.addons.get_ghf_orbspin(mf.mo_energy, mf.mo_occ, True)
        g = np.zeros((nbos, nmo*2, nmo*2))
        g[np.ix_(range(nbos), orbspin==0, orbspin==0)] = g_
        g[np.ix_(range(nbos), orbspin==1, orbspin==1)] = g_

        ccsd = GEBCC(
                mf.to_ghf(),  # Direct conversion needed for same ordering as reference data
                ansatz="CCSD-S-1-1",
                g=g,
                omega=omega,
                shift=cls.shift,
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        ccsd.options.t_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd, cls.eris, cls.data = mf, ccsd, eris, data

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd, cls.eris, cls.data

    def test_const(self):
        a = self.data[self.shift]["const"]
        b = self.ccsd.const
        self.assertAlmostEqual(a, b, 7)

    def test_xi(self):
        a = self.data[self.shift]["xi"]
        b = self.ccsd.xi
        np.testing.assert_almost_equal(a, b, 7)

    def test_fock(self):
        for tag in ("oo", "ov", "vo", "vv"):
            a = self.data[self.shift]["f"+tag]
            b = getattr(self.ccsd.fock, tag)
            np.testing.assert_almost_equal(a, b, 7)

    def test_g(self):
        for tag in ("oo", "ov", "vo", "vv"):
            a = self.data[self.shift]["gb"+tag]
            b = getattr(self.ccsd.g, "b"+tag)
            np.testing.assert_almost_equal(a, b, 7)

    def test_energy(self):
        a = self.data[self.shift]["e_corr"]
        b = self.ccsd.e_corr
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.data[self.shift]["t1"]
        b = self.ccsd.t1
        np.testing.assert_almost_equal(a, b, 6)

    def test_t2_amplitudes(self):
        a = self.data[self.shift]["t2"]
        b = self.ccsd.t2
        np.testing.assert_almost_equal(a, b, 6)

    def test_s1_amplitudes(self):
        a = self.data[self.shift]["s1"]
        b = self.ccsd.amplitudes["s1"]
        np.testing.assert_almost_equal(a, b, 6)

    def test_u11_amplitudes(self):
        a = self.data[self.shift]["u11"]
        b = self.ccsd.amplitudes["u11"]
        np.testing.assert_almost_equal(a, b, 6)

    def test_l1_amplitudes(self):
        a = self.data[self.shift]["l1"]
        b = self.ccsd.l1
        np.testing.assert_almost_equal(a, b, 6)

    def test_l2_amplitudes(self):
        a = self.data[self.shift]["l2"]
        b = self.ccsd.l2
        np.testing.assert_almost_equal(a, b, 5)  # FIXME low tol

    def test_ls1_amplitudes(self):
        a = self.data[self.shift]["ls1"]
        b = self.ccsd.lambdas["ls1"]
        np.testing.assert_almost_equal(a, b, 6)

    def test_lu11_amplitudes(self):
        a = self.data[self.shift]["lu11"]
        b = self.ccsd.lambdas["lu11"]
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm1_f(self):
        a = self.data[self.shift]["rdm1_f"]
        b = self.ccsd.make_rdm1_f()
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm2_f(self):
        a = self.data[self.shift]["rdm2_f"]
        b = self.ccsd.make_rdm2_f()
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm1_b(self):
        a = self.data[self.shift]["rdm1_b"]
        b = self.ccsd.make_rdm1_b()
        np.testing.assert_almost_equal(a, b, 6)

    def test_dm_b(self):
        a = self.data[self.shift]["dm_b"]
        b = np.array(self.ccsd.make_sing_b_dm())
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm_eb(self):
        a = self.data[self.shift]["rdm_eb"]
        b = self.ccsd.make_eb_coup_rdm()
        np.testing.assert_almost_equal(a, b, 6)


@pytest.mark.reference
class GCCSD_S_1_1_NoShift_Tests(GCCSD_S_1_1_Tests):
    """Test GCCSD-S-1-1 against the legacy GCCSD-S-1-1 values with
    shift=True.
    """

    shift = False



if __name__ == "__main__":
    print("Tests for GCCSD-S-1-1")
    unittest.main()
