"""Tests for the RCCSD model.
"""

import unittest
import pytest
import itertools
import pickle
import os

import numpy as np
import scipy.linalg

from pyscf import gto, scf, cc, lib

from ebcc import util, REBCC


class RCCSD_Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
            mo_coeff = data["mo_coeff"]
            data = data[(2, 0, 0)]

        mol = gto.Mole()
        mol.atom = "H 0 0 0; F 0 0 1.1"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        mf.mo_coeff = mo_coeff

        orbspin = scf.addons.get_ghf_orbspin(mf.mo_energy, mf.mo_occ, True)

        ccsd = REBCC(mf, rank=("SD", "", ""), log=util.NullLogger())
        ccsd.options.e_tol = 1e-12
        ccsd.options.t_tol = 1e-12
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

    def test_energy(self):
        a = self.data[True]["e_corr"]
        b = self.ccsd.e_corr
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.data[True]["t1"]
        b = scipy.linalg.block_diag(self.ccsd.t1, self.ccsd.t1)[self.osort][:, self.vsort]
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm1_f(self):
        rdm1_f = self.ccsd.make_rdm1_f()
        rdm1_f = 0.5 * (rdm1_f + rdm1_f.T.conj())
        a = self.data[True]["rdm1_f"]
        b = scipy.linalg.block_diag(rdm1_f, rdm1_f) / 2
        b = b[self.fsort][:, self.fsort]
        np.savetxt("tmp1.dat", a)
        np.savetxt("tmp2.dat", b)
        np.testing.assert_almost_equal(a, b, 6)

    def test_l1_amplitudes(self):
        a = self.data[True]["l1"]
        b = scipy.linalg.block_diag(self.ccsd.l1, self.ccsd.l1)[self.vsort][:, self.osort]
        np.testing.assert_almost_equal(a, b, 6)


class RCCSD_PySCF_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0 0 0; O 0 0 1"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd_ref = cc.CCSD(mf)
        ccsd_ref.conv_tol = 1e-12
        ccsd_ref.conv_tol_normt = 1e-12
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = REBCC(mf, rank=("SD", "", ""), log=util.NullLogger())
        ccsd.options.e_tol = 1e-12
        ccsd.options.t_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd

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

    def test_l1_amplitudes(self):
        a = self.ccsd_ref.l1
        b = self.ccsd.l1.T
        np.testing.assert_almost_equal(a, b, 6)

    def test_l2_amplitudes(self):
        a = self.ccsd_ref.l2
        b = self.ccsd.l2.transpose(2, 3, 0, 1)
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm1(self):
        a = self.ccsd_ref.make_rdm1()
        b = self.ccsd.make_rdm1_f(eris=self.eris)
        b = 0.5 * (b + b.T.conj())
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2()
        b = self.ccsd.make_rdm2_f(eris=self.eris)
        #o, v = slice(None, self.ccsd.nocc), slice(self.ccsd.nocc, None)
        #d2 = (
        #    b[o,v,o,v], b[v,v,v,v], b[o,o,o,o], b[o,o,v,v],
        #    b[o,v,v,o], b[v,v,o,v], b[o,v,v,v], b[o,o,o,v],
        #)
        #b = cc.ccsd_rdm._make_rdm2(self.ccsd, None, d2, with_dm1=False, with_frozen=False) * 0.25
        #for s1, k1 in [(o, "o"), (v, "v")]:
        #    for s2, k2 in [(o, "o"), (v, "v")]:
        #        for s3, k3 in [(o, "o"), (v, "v")]:
        #            for s4, k4 in [(o, "o"), (v, "v")]:
        #                print(k1+k2+k3+k4, np.linalg.norm(a[s1,s2,s3,s4]-b[s1,s2,s3,s4]), np.allclose(a[s1,s2,s3,s4], b[s1,s2,s3,s4]))
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_eom_ip(self):
        e1, v1 = self.ccsd.eom_ip(nroots=5)
        e2, v2 = self.ccsd_ref.ipccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 6)

    def test_eom_ea(self):
        e1, v1 = self.ccsd.eom_ea(nroots=5)
        e2, v2 = self.ccsd_ref.eaccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 6)


if __name__ == "__main__":
    print("Tests for RCCSD")
    unittest.main()
