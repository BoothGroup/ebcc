"""Tests for the UCCSD-SD-SD model.
"""

import itertools
import unittest
import pytest
import pickle
import os

import numpy as np
import scipy.linalg

from pyscf import gto, scf, cc, lib

from ebcc import NullLogger, GEBCC, UEBCC


class UCCSD_SD_SD_Tests(unittest.TestCase):
    """Test UCCSD-SD-SD against the legacy GCCSD-SD-SD values with
    shift=True. The system is a singlet.
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
        mf = mf.to_uhf()

        nmo = mf.mo_occ[0].size
        nbos = 5
        np.random.seed(12345)
        g = np.random.random((nbos, nmo, nmo)) * 0.02
        g = 0.5 * (g + g.transpose(0, 2, 1).conj())
        omega = np.random.random((nbos,)) * 5.0

        ccsd = UEBCC(mf, rank=("SD", "SD", "SD"), g=g, omega=omega, shift=cls.shift, log=NullLogger())
        ccsd.options.e_tol = 1e-12
        ccsd.options.t_tol = 1e-12
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

    #def test_l1_amplitudes(self):
    #    a = self.data[self.shift]["l1"]
    #    b = scipy.linalg.block_diag(self.ccsd.l1.aa, self.ccsd.l1.bb)[self.vsort][:, self.osort]
    #    np.testing.assert_almost_equal(a, b, 6)

    #def test_ls1_amplitudes(self):
    #    a = self.data[self.shift]["ls1"]
    #    b = self.ccsd.lambdas["ls1"]
    #    np.testing.assert_almost_equal(a, b, 6)

    #def test_ls1_amplitudes(self):
    #    a = self.data[self.shift]["ls2"]
    #    b = self.ccsd.lambdas["ls2"]
    #    np.testing.assert_almost_equal(a, b, 6)

    #def test_lu11_amplitudes(self):
    #    a = self.data[self.shift]["lu11"]
    #    b = self.ccsd.lambdas["lu11"]
    #    b = np.array([scipy.linalg.block_diag(x, y) for x, y in zip(b.aa, b.bb)])
    #    b = b[:, self.vsort][:, :, self.osort]
    #    np.testing.assert_almost_equal(a, b, 6)

    #def test_rdm1_f(self):
    #    rdm1_f = self.ccsd.make_rdm1_f()
    #    a = self.data[self.shift]["rdm1_f"]
    #    b = scipy.linalg.block_diag(rdm1_f.aa, rdm1_f.bb)
    #    b = b[self.fsort][:, self.fsort]
    #    np.testing.assert_almost_equal(a, b, 6)

    #def test_rdm1_b(self):
    #    a = self.data[self.shift]["rdm1_b"]
    #    b = self.ccsd.make_rdm1_b()
    #    np.testing.assert_almost_equal(a, b, 6)

    #def test_dm_b(self):
    #    a = self.data[self.shift]["dm_b"]
    #    b = np.array(self.ccsd.make_sing_b_dm())
    #    np.testing.assert_almost_equal(a, b, 6)

    #def test_rdm_eb(self):
    #    a = np.array(self.data[self.shift]["rdm_eb"])
    #    b = self.ccsd.make_eb_coup_rdm()
    #    b = np.array([
    #        [scipy.linalg.block_diag(x, y) for x, y in zip(b.aa[0], b.bb[0])],
    #        [scipy.linalg.block_diag(x, y) for x, y in zip(b.aa[1], b.bb[1])],
    #    ])
    #    b = b[:, :, self.fsort][:, :, :, self.fsort]
    #    np.testing.assert_almost_equal(a, b, 6)


class UCCSD_SD_SD_NoShift_Tests(UCCSD_SD_SD_Tests):
    """Test UCCSD-SD-SD against the legacy GCCSD-SD-SD values with
    shift=False. The system is a singlet.
    """

    shift = False


if __name__ == "__main__":
    print("Tests for UCCSD-SD-SD")
    unittest.main()
