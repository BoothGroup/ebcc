"""Tests for the UCCSD model.
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


class UCCSD_Tests(unittest.TestCase):
    """Test UCCSD against the legacy GCCSD values. The system is a
    singlet.
    """

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
        mf = mf.to_uhf()

        orbspin = scf.addons.get_ghf_orbspin(mf.mo_energy, mf.mo_occ, False)

        ccsd = UEBCC(mf, rank=("SD", "", ""), log=NullLogger())
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

    def test_energy(self):
        a = self.data[True]["e_corr"]
        b = self.ccsd.e_corr
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.data[True]["t1"]
        b = scipy.linalg.block_diag(self.ccsd.t1.aa, self.ccsd.t1.bb)[self.osort][:, self.vsort]
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm1_f(self):
        rdm1_f = self.ccsd.make_rdm1_f()
        a = self.data[True]["rdm1_f"]
        b = scipy.linalg.block_diag(rdm1_f.aa, rdm1_f.bb)
        b = b[self.fsort][:, self.fsort]
        np.testing.assert_almost_equal(a, b, 6)

    def test_l1_amplitudes(self):
        a = self.data[True]["l1"]
        b = scipy.linalg.block_diag(self.ccsd.l1.aa, self.ccsd.l1.bb)[self.vsort][:, self.osort]
        np.testing.assert_almost_equal(a, b, 6)


class UCCSD_PySCF_Tests(unittest.TestCase):
    """Test UCCSD against the PySCF values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0 0 0; O 0 0 1"
        mol.basis = "cc-pvdz"
        mol.spin = 2
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        mf = mf.to_uhf()

        ccsd_ref = cc.UCCSD(mf)
        ccsd_ref.conv_tol = 1e-12
        ccsd_ref.conv_tol_normt = 1e-12
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = UEBCC(mf, rank=("SD", "", ""), log=NullLogger())
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
        np.testing.assert_almost_equal(a[0], b.aa, 6)
        np.testing.assert_almost_equal(a[1], b.bb, 6)

    def test_l1_amplitudes(self):
        a = self.ccsd_ref.l1
        b = self.ccsd.l1
        np.testing.assert_almost_equal(a[0], b.aa.T, 6)
        np.testing.assert_almost_equal(a[1], b.bb.T, 6)

    def test_rdm1(self):
        a = self.ccsd_ref.make_rdm1()
        b = self.ccsd.make_rdm1_f(eris=self.eris)
        np.testing.assert_almost_equal(a[0], b.aa, 6)
        np.testing.assert_almost_equal(a[1], b.bb, 6)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2()
        b = self.ccsd.make_rdm2_f(eris=self.eris)
        np.testing.assert_almost_equal(a[0], b.aaaa, 6)
        np.testing.assert_almost_equal(a[1], b.aabb, 6)
        np.testing.assert_almost_equal(a[2], b.bbbb, 6)


if __name__ == "__main__":
    print("Tests for UCCSD")
    unittest.main()
