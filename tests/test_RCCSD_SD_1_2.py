"""Tests for the RCCSD-SD-1-2 model.
"""

import itertools
import os
import pickle
import unittest
import tempfile

import numpy as np
import pytest
import scipy.linalg
from pyscf import cc, gto, lib, scf

from ebcc import REBCC, NullLogger


@pytest.mark.reference
class RCCSD_SD_1_2_Tests(unittest.TestCase):
    """Test RCCSD-SD-1-2 against the legacy GCCSD-SD-1-2 values with
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
        g = np.random.random((nbos, nmo, nmo)) * 0.02
        g = 0.5 * (g + g.transpose(0, 2, 1).conj())
        omega = np.random.random((nbos,)) * 5.0

        ccsd = REBCC(
                mf,
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

        osort = list(itertools.chain(*zip(range(ccsd.nocc), range(ccsd.nocc, 2*ccsd.nocc))))
        vsort = list(itertools.chain(*zip(range(ccsd.nvir), range(ccsd.nvir, 2*ccsd.nvir))))
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
        b = scipy.linalg.block_diag(self.ccsd.t1, self.ccsd.t1)[self.osort][:, self.vsort]
        np.testing.assert_almost_equal(a, b, 6)

    def test_s1_amplitudes(self):
        a = self.data[self.shift]["s1"]
        b = self.ccsd.amplitudes["s1"]
        np.testing.assert_almost_equal(a, b, 6)

    def test_u11_amplitudes(self):
        a = self.data[self.shift]["u11"]
        b = np.array([scipy.linalg.block_diag(x, x) for x in self.ccsd.amplitudes["u11"]])
        b = b[:, self.osort][:, :, self.vsort]
        np.testing.assert_almost_equal(a, b, 6)

    @pytest.mark.regression
    def test_lambdas(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"]),  -0.05224802896432364, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"]),  -0.05259353852072242, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),  0.01144093908452783, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),  0.00112151562710175, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"]), 0.02324025349341592, 5)

    @pytest.mark.regression
    def test_dms(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_f()),       3.058518964389294, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm2_f()),      -1.792334872393240, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_b()),       0.035528094644337, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_sing_b_dm()),    0.020426319078153, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_eb_coup_rdm()), -2.107572966431455, 6)


@pytest.mark.reference
class RCCSD_SD_1_2_NoShift_Tests(RCCSD_SD_1_2_Tests):
    """Test RCCSD-SD-1-2 against the legacy GCCSD-SD-1-2 values with
    shift=False.
    """

    shift = False

    @pytest.mark.regression
    def test_lambdas(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"]),  -0.05251435208717554, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"]),  -0.05247466694425869, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),  0.19939707497970954, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),  0.01046937063225352, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"]), 0.03280064547688709, 5)

    @pytest.mark.regression
    def test_dms(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_f()),       3.056907984935672, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm2_f()),      -1.814797297136204, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_b()),       0.009432763560355, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_sing_b_dm()),    0.343097457676634, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_eb_coup_rdm()), -2.128888460564522, 6)


@pytest.mark.reference
class RCCSD_SD_1_2_Dump_Tests(RCCSD_SD_1_2_Tests):
    """Test RCCSD-SD-1-2 after dumping and loading the data.
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
        g = np.random.random((nbos, nmo, nmo)) * 0.02
        g = 0.5 * (g + g.transpose(0, 2, 1).conj())
        omega = np.random.random((nbos,)) * 5.0

        ccsd = REBCC(
                mf,
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

        osort = list(itertools.chain(*zip(range(ccsd.nocc), range(ccsd.nocc, 2*ccsd.nocc))))
        vsort = list(itertools.chain(*zip(range(ccsd.nvir), range(ccsd.nvir, 2*ccsd.nvir))))
        fsort = list(itertools.chain(*zip(range(ccsd.nmo), range(ccsd.nmo, 2*ccsd.nmo))))

        cls.mf, cls.ccsd, cls.eris, cls.data = mf, ccsd, eris, data
        cls.osort, cls.vsort, cls.fsort = osort, vsort, fsort

        file = "%s/ebcc.h5" % tempfile.gettempdir()
        cls.ccsd.write(file)
        cls.ccsd = cls.ccsd.__class__.read(file, log=cls.ccsd.log)



if __name__ == "__main__":
    print("Tests for RCCSD-SD-1-2")
    unittest.main()
