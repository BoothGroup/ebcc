"""Tests for the RCCSD-SD-1-1 model.
"""

import itertools
import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import cc, gto, lib, scf

from ebcc import REBCC, NullLogger


@pytest.mark.reference
class RCCSD_SD_1_1_Tests(unittest.TestCase):
    """Test RCCSD-SD-1-1 against the legacy GCCSD-SD-1-1 values with
    shift=True.
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
        g = np.random.random((nbos, nmo, nmo)) * 0.02
        g = 0.5 * (g + g.transpose(0, 2, 1).conj())
        omega = np.random.random((nbos,)) * 5.0

        ccsd = REBCC(
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
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"]),  -0.05224789962640912, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"]),  -0.05259102842150937, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),  0.01144025359438191, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),  0.00112018486258491, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"]), 0.02318895479041393, 5)

    @pytest.mark.regression
    def test_dms(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_f()),       3.058512840976993, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm2_f()),      -1.792341170084792, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_b()),       0.035519688475322, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_sing_b_dm()),    0.020377060902172, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_eb_coup_rdm()), -2.107380102954150, 6)


@pytest.mark.reference
class RCCSD_SD_1_1_NoShift_Tests(RCCSD_SD_1_1_Tests):
    """Test RCCSD-SD-1-1 against the legacy GCCSD-SD-1-1 values with
    shift=False.
    """

    shift = False

    @pytest.mark.regression
    def test_lambdas(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"]),  -0.0525114526502243, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"]),  -0.0524610698627949, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),  0.1993916026112773, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),  0.0104612037870549, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"]), 0.0325865795300660, 5)

    @pytest.mark.regression
    def test_dms(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_f()),       3.0568681032194087, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm2_f()),      -1.8147124426634570, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm1_b()),       0.0094255137407322, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_sing_b_dm()),    0.3430246995030957, 6)
        self.assertAlmostEqual(lib.fp(self.ccsd.make_eb_coup_rdm()), -2.1284999094453990, 6)



if __name__ == "__main__":
    print("Tests for RCCSD-SD-1-1")
    unittest.main()
