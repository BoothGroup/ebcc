"""Tests for the GCCSD-SD-1-2 model.
"""

import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import cc, gto, lib, scf

from ebcc import REBCC, UEBCC, GEBCC, NullLogger


@pytest.mark.reference
class GCCSD_SD_1_2_Tests(unittest.TestCase):
    """Test GCCSD-SD-1-2 against the legacy GCCSD-SD-1-2 values with
    shift=True.
    """

    shift = True

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
            mo_coeff = data["mo_coeff"]
            data = data[(2, 2, 2)]

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
                ansatz="CCSD-SD-1-2",
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
        cls.g, cls.g_rhf, cls.omega = g, g_, omega

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd, cls.eris, cls.data
        del cls.g, cls.g_rhf, cls.omega

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
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"]),   0.0217162716903423, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"]),  -0.0710387889255989, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),  0.0114409407266459, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),  0.0011215148388206, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"]), 0.0133504475951527, 5)

    @pytest.mark.regression
    def test_dms(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_f()),       1.076728368011556, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm2_f()),       0.577217348794959, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_b()),       0.035528094885662, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_sing_b_dm()),    0.020426320880006, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_eb_coup_rdm()), -0.545361301095797, 6)

    @pytest.mark.regression
    def test_from_rebcc(self):
        mf = self.mf

        # Careful with ordering:
        uhf = mf.to_uhf()
        orbspin = scf.addons.get_ghf_orbspin(uhf.mo_energy, uhf.mo_occ, False)
        nmo = self.mf.mo_occ.size
        g = np.zeros((self.omega.size, nmo*2, nmo*2))
        g[np.ix_(range(self.omega.size), orbspin==0, orbspin==0)] = self.g_rhf.copy()
        g[np.ix_(range(self.omega.size), orbspin==1, orbspin==1)] = self.g_rhf.copy()

        gebcc1 = GEBCC(
                mf,
                ansatz="CCSD-SD-1-2",
                g=g,
                omega=self.omega,
                shift=self.shift,
                log=NullLogger(),
        )
        gebcc1.options.e_tol = 1e-12
        eris = gebcc1.get_eris()
        gebcc1.kernel(eris=eris)
        gebcc1.solve_lambda(eris=eris)

        rebcc = REBCC(
                mf,
                ansatz="CCSD-SD-1-2",
                g=self.g_rhf,
                omega=self.omega,
                shift=self.shift,
                log=NullLogger(),
        )
        rebcc.options.e_tol = 1e-12
        eris = rebcc.get_eris()
        rebcc.kernel(eris=eris)
        rebcc.solve_lambda(eris=eris)
        gebcc2 = GEBCC.from_rebcc(rebcc)

        self.assertAlmostEqual(gebcc1.energy(), gebcc2.energy())
        for key in gebcc1.amplitudes.keys():
            np.testing.assert_almost_equal(gebcc1.amplitudes[key], gebcc2.amplitudes[key], 6)
        for key in gebcc1.lambdas.keys():
            np.testing.assert_almost_equal(gebcc1.lambdas[key], gebcc2.lambdas[key], 5)
        np.testing.assert_almost_equal(gebcc1.make_rdm1_f(), gebcc2.make_rdm1_f(), 6)
        np.testing.assert_almost_equal(gebcc1.make_rdm2_f(), gebcc2.make_rdm2_f(), 6)
        np.testing.assert_almost_equal(gebcc1.make_rdm1_b(), gebcc2.make_rdm1_b(), 6)
        np.testing.assert_almost_equal(gebcc1.make_sing_b_dm(), gebcc2.make_sing_b_dm(), 6)
        np.testing.assert_almost_equal(gebcc1.make_eb_coup_rdm(), gebcc2.make_eb_coup_rdm(), 6)

    @pytest.mark.regression
    def test_from_uebcc(self):
        mf = self.mf.to_uhf()

        # Careful with ordering:
        orbspin = scf.addons.get_ghf_orbspin(mf.mo_energy, mf.mo_occ, False)
        nmo = self.mf.mo_occ.size
        g = np.zeros((self.omega.size, nmo*2, nmo*2))
        g[np.ix_(range(self.omega.size), orbspin==0, orbspin==0)] = self.g_rhf.copy()
        g[np.ix_(range(self.omega.size), orbspin==1, orbspin==1)] = self.g_rhf.copy()

        gebcc1 = GEBCC(
                mf,
                ansatz="CCSD-SD-1-2",
                g=g,
                omega=self.omega,
                shift=self.shift,
                log=NullLogger(),
        )
        gebcc1.options.e_tol = 1e-12
        eris = gebcc1.get_eris()
        gebcc1.kernel(eris=eris)
        gebcc1.solve_lambda(eris=eris)

        uebcc = UEBCC(
                mf,
                ansatz="CCSD-SD-1-2",
                g=self.g_rhf,
                omega=self.omega,
                shift=self.shift,
                log=NullLogger(),
        )
        uebcc.options.e_tol = 1e-12
        eris = uebcc.get_eris()
        uebcc.kernel(eris=eris)
        uebcc.solve_lambda(eris=eris)
        gebcc2 = GEBCC.from_uebcc(uebcc)

        self.assertAlmostEqual(gebcc1.energy(), self.data[self.shift]["e_corr"])
        self.assertAlmostEqual(gebcc2.energy(), self.data[self.shift]["e_corr"])

        self.assertAlmostEqual(gebcc1.energy(), gebcc2.energy())
        for key in gebcc1.amplitudes.keys():
            np.testing.assert_almost_equal(gebcc1.amplitudes[key], gebcc2.amplitudes[key], 6)
        for key in gebcc1.lambdas.keys():
            np.testing.assert_almost_equal(gebcc1.lambdas[key], gebcc2.lambdas[key], 5)
        np.testing.assert_almost_equal(gebcc1.make_rdm1_f(), gebcc2.make_rdm1_f(), 6)
        np.testing.assert_almost_equal(gebcc1.make_rdm2_f(), gebcc2.make_rdm2_f(), 6)
        np.testing.assert_almost_equal(gebcc1.make_rdm1_b(), gebcc2.make_rdm1_b(), 6)
        np.testing.assert_almost_equal(gebcc1.make_sing_b_dm(), gebcc2.make_sing_b_dm(), 6)
        np.testing.assert_almost_equal(gebcc1.make_eb_coup_rdm(), gebcc2.make_eb_coup_rdm(), 6)


@pytest.mark.reference
class GCCSD_SD_1_2_NoShift_Tests(GCCSD_SD_1_2_Tests):
    """Test GCCSD-SD-1-2 against the legacy GCCSD-SD-1-2 values with
    shift=False.
    """

    shift = False

    @pytest.mark.regression
    def test_lambdas(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"]),   0.0204957699782574, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"]),  -0.0712052531493835, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),  0.1993970793295078, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),  0.0104693705804119, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"]), 0.0103144422648399, 5)

    @pytest.mark.regression
    def test_dms(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_f()),       1.077968578995936, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm2_f()),       0.581652716437062, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_b()),       0.009432763759286, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_sing_b_dm()),    0.343097462745462, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_eb_coup_rdm()), -0.545233491374837, 6)



if __name__ == "__main__":
    print("Tests for GCCSD-SD-1-2")
    unittest.main()
