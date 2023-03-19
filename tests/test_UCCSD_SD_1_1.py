"""Tests for the UCCSD-SD-1-1 model.
"""

import itertools
import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import cc, gto, lib, scf

from ebcc import GEBCC, UEBCC, NullLogger


@pytest.mark.reference
class UCCSD_SD_1_1_Tests(unittest.TestCase):
    """Test UCCSD-SD-1-1 against the legacy GCCSD-SD-1-1 values with
    shift=True. The system is a singlet.
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
        mf = mf.to_uhf()

        nmo = mf.mo_occ[0].size
        nbos = 5
        np.random.seed(12345)
        g = np.random.random((nbos, nmo, nmo)) * 0.02
        g = 0.5 * (g + g.transpose(0, 2, 1).conj())
        omega = np.random.random((nbos,)) * 5.0

        ccsd = UEBCC(
                mf,
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

        osort = list(itertools.chain(*zip(range(ccsd.nocc[0]), range(ccsd.nocc[0], ccsd.nocc[0]+ccsd.nocc[1]))))
        vsort = list(itertools.chain(*zip(range(ccsd.nvir[1]), range(ccsd.nvir[0], ccsd.nvir[0]+ccsd.nvir[1]))))
        fsort = list(itertools.chain(*zip(range(ccsd.nmo), range(ccsd.nmo, 2*ccsd.nmo))))

        cls.mf, cls.ccsd, cls.eris, cls.data = mf, ccsd, eris, data
        cls.osort, cls.vsort, cls.fsort = osort, vsort, fsort

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd, cls.eris, cls.data
        del cls.osort, cls.vsort

    def test_const(self):
        a = self.data[self.shift]["const"]
        b = self.ccsd.const
        self.assertAlmostEqual(a, b, 7)

    def test_xi(self):
        a = self.data[self.shift]["xi"]
        b = self.ccsd.xi
        np.testing.assert_almost_equal(a, b, 7)

    def test_energy(self):
        a = self.data[self.shift]["e_corr"]
        b = self.ccsd.e_corr
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.data[self.shift]["t1"]
        b = scipy.linalg.block_diag(self.ccsd.t1.aa, self.ccsd.t1.bb)[self.osort][:, self.vsort]
        np.testing.assert_almost_equal(a, b, 6)

    def test_s1_amplitudes(self):
        a = self.data[self.shift]["s1"]
        b = self.ccsd.amplitudes["s1"]
        np.testing.assert_almost_equal(a, b, 6)

    def test_s1_amplitudes(self):
        a = self.data[self.shift]["s2"]
        b = self.ccsd.amplitudes["s2"]
        np.testing.assert_almost_equal(a, b, 6)

    def test_u11_amplitudes(self):
        a = self.data[self.shift]["u11"]
        b = self.ccsd.amplitudes["u11"]
        b = np.array([scipy.linalg.block_diag(x, y) for x, y in zip(b.aa, b.bb)])
        b = b[:, self.osort][:, :, self.vsort]
        np.testing.assert_almost_equal(a, b, 6)

    @pytest.mark.regression
    def test_lambdas(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"].aa),   -0.05224782031081628, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"].bb),   -0.05224782031081629, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].aaaa),  0.02916752477065619, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].abab), -0.05259099799643322, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].bbbb),  0.02916752477065620, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),      0.01144024365028928, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),      0.01144024365028928, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),      0.00112018432431747, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),      0.00112018432431747, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"].aa),  0.02318898262003968, 5)

    @pytest.mark.regression
    def test_dms(self):
        rdm1_f = self.ccsd.make_rdm1_f()
        rdm2_f = self.ccsd.make_rdm2_f()
        rdm1_b = self.ccsd.make_rdm1_b()
        dm_b = self.ccsd.make_sing_b_dm()
        dm_eb = self.ccsd.make_eb_coup_rdm()
        self.assertAlmostEqual(lib.fp(rdm1_f.aa),     1.529256374555281, 6)
        self.assertAlmostEqual(lib.fp(rdm1_f.bb),     1.529256374555281, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.aaaa), -12.786842039450173, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.aabb),  11.890671585335324, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.bbbb), -12.786842039450171, 6)
        self.assertAlmostEqual(lib.fp(rdm1_b),        0.035519686030352, 6)
        self.assertAlmostEqual(lib.fp(dm_b),          0.020377050012922, 6)
        self.assertAlmostEqual(lib.fp(dm_eb.aa),     -1.053690019687700, 6)
        self.assertAlmostEqual(lib.fp(dm_eb.bb),     -1.053690019687700, 6)


@pytest.mark.reference
class UCCSD_SD_1_1_NoShift_Tests(UCCSD_SD_1_1_Tests):
    """Test UCCSD-SD-1-1 against the legacy GCCSD-SD-1-1 values with
    shift=False. The system is a singlet.
    """

    shift = False

    @pytest.mark.regression
    def test_lambdas(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"].aa),   -0.0525114740763063, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"].bb),   -0.0525114740763063, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].aaaa),  0.0296319803931865, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].abab), -0.0524610236810773, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].bbbb),  0.0296319803931865, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),      0.1993916093517041, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),      0.1993916093517041, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),      0.0104612042249723, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),      0.0104612042249723, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"].aa),  0.0325865337806562, 5)

    @pytest.mark.regression
    def test_dms(self):
        rdm1_f = self.ccsd.make_rdm1_f()
        rdm2_f = self.ccsd.make_rdm2_f()
        rdm1_b = self.ccsd.make_rdm1_b()
        dm_b = self.ccsd.make_sing_b_dm()
        dm_eb = self.ccsd.make_eb_coup_rdm()
        self.assertAlmostEqual(lib.fp(rdm1_f.aa),    1.528434038989676, 6)
        self.assertAlmostEqual(lib.fp(rdm1_f.bb),    1.528434038989676, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.aaaa), -12.79625827605020, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.aabb),  11.88890192239292, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.bbbb), -12.79625827605020, 6)
        self.assertAlmostEqual(lib.fp(rdm1_b),        0.00942551407916, 6)
        self.assertAlmostEqual(lib.fp(dm_b),          0.34302470631357, 6)
        self.assertAlmostEqual(lib.fp(dm_eb.aa),     -1.06424997076165, 6)
        self.assertAlmostEqual(lib.fp(dm_eb.bb),     -1.06424997076165, 6)


if __name__ == "__main__":
    print("Tests for UCCSD-SD-1-1")
    unittest.main()
