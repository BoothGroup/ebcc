"""Tests for the GCCSD model.
"""

import unittest
import pytest
import pickle
import os

import numpy as np
import scipy.linalg

from pyscf import gto, scf, cc, lib

from ebcc import NullLogger, GEBCC


class GCCSD_Tests(unittest.TestCase):
    """Test GCCSD against the legacy GCCSD values.
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

        orbspin = scf.addons.get_ghf_orbspin(mf.mo_energy, mf.mo_occ, True)

        ccsd = GEBCC(mf, rank=("SD", "", ""), log=NullLogger())
        ccsd.options.e_tol = 1e-12
        ccsd.options.t_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd, cls.eris, cls.data = mf, ccsd, eris, data

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd, cls.eris, cls.data

    def test_fock(self):
        for tag in ("oo", "ov", "vo", "vv"):
            a = self.data[True]["f"+tag]
            b = getattr(self.ccsd.fock, tag)
            np.testing.assert_almost_equal(a, b, 7)

    def test_energy(self):
        a = self.data[True]["e_corr"]
        b = self.ccsd.e_corr
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.data[True]["t1"]
        b = self.ccsd.t1
        np.testing.assert_almost_equal(a, b, 6)

    def test_t2_amplitudes(self):
        a = self.data[True]["t2"]
        b = self.ccsd.t2
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm1_f(self):
        a = self.data[True]["rdm1_f"]
        b = self.ccsd.make_rdm1_f()
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm2_f(self):
        a = self.data[True]["rdm2_f"]
        b = self.ccsd.make_rdm2_f()
        np.testing.assert_almost_equal(a, b, 6)

    def test_l1_amplitudes(self):
        a = self.data[True]["l1"]
        b = self.ccsd.l1
        np.testing.assert_almost_equal(a, b, 6)

    def test_l2_amplitudes(self):
        a = self.data[True]["l2"]
        b = self.ccsd.l2
        np.testing.assert_almost_equal(a, b, 6)


class GCCSD_PySCF_Tests(unittest.TestCase):
    """Test GCCSD against the PySCF GCCSD values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
        mol.basis = "sto3g"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd_ref = cc.GCCSD(mf)
        ccsd_ref.conv_tol = 1e-12
        ccsd_ref.conv_tol_normt = 1e-12
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = GEBCC(mf, rank=("SD", "", ""), log=NullLogger())
        ccsd.e_tol = 1e-12
        ccsd.t_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd

    def test_ccsd_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 8)

    def test_ccsd_t1_amplitudes(self):
        a = self.ccsd_ref.t1
        b = self.ccsd.t1
        np.testing.assert_almost_equal(a, b, 6)

    def test_ccsd_t2_amplitudes(self):
        a = self.ccsd_ref.t2
        b = self.ccsd.t2
        np.testing.assert_almost_equal(a, b, 6)

    def test_ccsd_l1_amplitudes(self):
        a = self.ccsd_ref.l1
        b = self.ccsd.l1.T
        np.testing.assert_almost_equal(a, b, 6)

    def test_ccsd_l2_amplitudes(self):
        a = self.ccsd_ref.l2
        b = self.ccsd.l2.transpose(2, 3, 0, 1)
        np.testing.assert_almost_equal(a, b, 6)  # FIXME doesn't reach precision

    def test_rdm1(self):
        a = self.ccsd_ref.make_rdm1()
        b = self.ccsd.make_rdm1_f(eris=self.eris)
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2()
        b = self.ccsd.make_rdm2_f(eris=self.eris)
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_eom_ip(self):
        e1, v1 = self.ccsd.eom_ip(nroots=5)
        e2, v2 = self.ccsd_ref.ipccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 8)

    def test_eom_ea(self):
        e1, v1 = self.ccsd.eom_ea(nroots=5)
        e2, v2 = self.ccsd_ref.eaccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 8)


if __name__ == "__main__":
    print("Tests for GCCSD")
    unittest.main()

