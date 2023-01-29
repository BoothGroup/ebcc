"""Tests for the GCCSD-SD-1-1 model.
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
class GCCSD_SD_1_1_Tests(unittest.TestCase):
    """Test GCCSD-SD-1-1 against the legacy GCCSD-SD-1-1 values with
    shift=False.
    """

    shift = True

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
            mo_coeff = data["mo_coeff"]
            data = data[(2, 2, 1)]

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
                ansatz="CCSD-SD-1-1",
                g=g,
                omega=omega,
                shift=cls.shift,
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
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

    def test_s2_amplitudes(self):
        a = self.data[self.shift]["s2"]
        b = self.ccsd.amplitudes["s2"]
        np.testing.assert_almost_equal(a, b, 6)

    def test_u11_amplitudes(self):
        a = self.data[self.shift]["u11"]
        b = self.ccsd.amplitudes["u11"]
        np.testing.assert_almost_equal(a, b, 6)

    @pytest.mark.regression
    def test_lambdas(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"]),   0.0217182273702451, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"]),  -0.0710390081793643, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),  0.0114402535193520, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),  0.0011201841174214, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"]), 0.0133207825661291, 5)

    @pytest.mark.regression
    def test_dms(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_f()),       1.076729770035047, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm2_f()),       0.577229272327733, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_b()),       0.035519685795300, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_sing_b_dm()),    0.020377049291827, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_eb_coup_rdm()), -0.545252297506057, 6)


@pytest.mark.reference
class GCCSD_SD_1_1_NoShift_Tests(GCCSD_SD_1_1_Tests):
    """Test GCCSD-SD-1-1 against the legacy GCCSD-SD-1-1 values with
    shift=True.
    """

    shift = False

    @pytest.mark.regression
    def test_lambdas(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"]),   0.02050459704486253, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"]),  -0.07121141154243935, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),  0.19939161755012555, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),  0.01046120322447800, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"]), 0.01039317107704462, 5)

    @pytest.mark.regression
    def test_dms(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_f()),       1.077941143482945, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm2_f()),       0.581736871456272, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_b()),       0.009425513968899, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_sing_b_dm()),    0.343024702059832, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_eb_coup_rdm()), -0.544727122781793, 6)



if __name__ == "__main__":
    print("Tests for GCCSD-SD-1-1")
    unittest.main()
