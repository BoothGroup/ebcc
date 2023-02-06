"""Tests for the RCCSD-S-1-1 model.
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
class RCCSD_S_1_1_Tests(unittest.TestCase):
    """Test RCCSD-S-1-1 against the legacy GCCSD-S-1-1 values with
    shift=True.
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
        g = np.random.random((nbos, nmo, nmo)) * 0.02
        g = 0.5 * (g + g.transpose(0, 2, 1).conj())
        omega = np.random.random((nbos,)) * 5.0

        ccsd = REBCC(
                mf,
                ansatz="CCSD-S-1-1",
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

    def test_l1_amplitudes(self):
        a = self.data[self.shift]["l1"]
        b = scipy.linalg.block_diag(self.ccsd.l1, self.ccsd.l1)[self.vsort][:, self.osort]
        np.testing.assert_almost_equal(a, b, 6)

    def test_ls1_amplitudes(self):
        a = self.data[self.shift]["ls1"]
        b = self.ccsd.lambdas["ls1"]
        np.testing.assert_almost_equal(a, b, 6)

    def test_lu11_amplitudes(self):
        a = self.data[self.shift]["lu11"]
        b = np.array([scipy.linalg.block_diag(x, x) for x in self.ccsd.lambdas["lu11"]])
        b = b[:, self.vsort][:, :, self.osort]
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm1_f(self):
        rdm1_f = self.ccsd.make_rdm1_f()
        a = self.data[self.shift]["rdm1_f"]
        b = scipy.linalg.block_diag(rdm1_f, rdm1_f) / 2
        b = b[self.fsort][:, self.fsort]
        np.testing.assert_almost_equal(a, b, 6)

    @pytest.mark.regression
    def test_rdm2_f(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm2_f()), -1.7916782894206773, 6)

    def test_rdm1_b(self):
        a = self.data[self.shift]["rdm1_b"]
        b = self.ccsd.make_rdm1_b()
        np.testing.assert_almost_equal(a, b, 6)

    def test_dm_b(self):
        a = self.data[self.shift]["dm_b"]
        b = np.array(self.ccsd.make_sing_b_dm())
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm_eb(self):
        a = np.array(self.data[self.shift]["rdm_eb"])
        b = np.array([[scipy.linalg.block_diag(x, x) for x in y] for y in self.ccsd.make_eb_coup_rdm()]) / 2
        b = b[:, :, self.fsort][:, :, :, self.fsort]
        np.testing.assert_almost_equal(a, b, 6)


@pytest.mark.reference
class RCCSD_S_1_1_NoShift_Tests(RCCSD_S_1_1_Tests):
    """Test RCCSD-S-1-1 against the legacy GCCSD-S-1-1 values with
    shift=False.
    """

    shift = False

    @pytest.mark.regression
    def test_rdm2_f(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.make_rdm2_f()), -1.8193615098460836, 6)



if __name__ == "__main__":
    print("Tests for RCCSD-S-1-1")
    unittest.main()
