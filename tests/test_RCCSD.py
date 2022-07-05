"""Tests for the RCCSD model.
"""

import unittest

import numpy as np

from pyscf import gto, scf, cc, lib

from ebcc import util, REBCC


class RCCSDTests(unittest.TestCase):

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
        self.assertAlmostEqual(a, b, 7)

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
