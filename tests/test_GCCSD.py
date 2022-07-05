"""Tests for the GCCSD model.
"""

import unittest

import numpy as np

from pyscf import gto, scf, cc, lib

from ebcc import util, GEBCC


class GCCSDTests(unittest.TestCase):
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

        ccsd = GEBCC(mf, rank=("SD", "", ""), log=util.NullLogger())
        ccsd.e_tol = 1e-12
        ccsd.t_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mol, cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mol, mf, ccsd_ref, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.ccsd_ref, cls.ccsd

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
        b = 0.5 * (b + b.T.conj())
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2()
        b = self.ccsd.make_rdm2_f(eris=self.eris).transpose(0, 2, 1, 3)
        #o, v = slice(None, self.ccsd.nocc), slice(self.ccsd.nocc, None)
        #d2 = [b[tuple({"o":o, "v":v}[k] for k in key)] for key in "ovov, vvvv, oooo, oovv, ovvo, vvov, ovvv, ooov".split(", ")]
        #b = cc.gccsd_rdm._make_rdm2(self.ccsd, None, d2, with_dm1=False, with_frozen=False)
        #b = (
        #    b.transpose(0, 1, 2, 3) + b.transpose(1, 0, 2, 3) + b.transpose(0, 1, 3, 2) + b.transpose(1, 0, 3, 2) +
        #    b.transpose(2, 3, 0, 1) + b.transpose(2, 3, 1, 0) + b.transpose(3, 2, 0, 1) + b.transpose(3, 2, 1, 0)
        #)
        #b /= 8
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

