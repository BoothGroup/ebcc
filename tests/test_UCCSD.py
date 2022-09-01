"""Tests for the UCCSD model.
"""

import itertools
import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import cc, gto, lib, scf

from ebcc import REBCC, UEBCC, GEBCC, NullLogger


@pytest.mark.reference
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

        ccsd = UEBCC(
                mf,
                fermion_excitations="SD",
                log=NullLogger(),
        )
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

    def test_from_rebcc(self):
        rebcc = REBCC(
                self.mf,
                fermion_excitations="SD",
                log=NullLogger(),
        )
        rebcc.options.e_tol = 1e-12
        rebcc.kernel()
        uebcc = UEBCC.from_rebcc(rebcc)
        # FIXME seem test_GCCSD.GCCSD_Tests.test_from_uebcc
        self.assertAlmostEqual(self.ccsd.energy(), uebcc.energy(), 8)

    def test_ip_moments(self):
        eom = self.ccsd.ip_eom()
        a = self.data[True]["ip_moms"].transpose(2, 0, 1)
        b = eom.moments(4)
        b = np.array([scipy.linalg.block_diag(x, y) for x, y in zip(b.aa, b.bb)])
        b = b[:, self.fsort][:, :, self.fsort]
        for i, (x, y) in enumerate(zip(a, b)):
            x /= np.max(np.abs(x))
            y /= np.max(np.abs(y))
            np.testing.assert_almost_equal(x, y, 6)

    def test_ea_moments(self):
        eom = self.ccsd.ea_eom()
        a = self.data[True]["ea_moms"].transpose(2, 0, 1)
        b = eom.moments(4)
        b = np.array([scipy.linalg.block_diag(x, y) for x, y in zip(b.aa, b.bb)])
        b = b[:, self.fsort][:, :, self.fsort]
        for i, (x, y) in enumerate(zip(a, b)):
            x /= np.max(np.abs(x))
            y /= np.max(np.abs(y))
            np.testing.assert_almost_equal(x, y, 6)

    def _test_ee_moments_diag(self):
        # FIXME broken
        eom = self.ccsd.ee_eom()
        a = self.data[True]["dd_moms"].transpose(4, 0, 1, 2, 3)
        a = a[:, :eom.nmo, :eom.nmo, :eom.nmo, :eom.nmo]
        a = np.einsum("npqrs,pq,rs->npqrs", a, np.eye(self.ccsd.nmo), np.eye(self.ccsd.nmo))
        b = eom.moments(4, diagonal_only=True)
        print(a[0, 0, 0, 0, 0])
        print(b.aaaa[0, 0, 0, 0, 0])
        print(b.aabb[0, 0, 0, 0, 0])
        print(self.ccsd.make_rdm2_f().aaaa[0, 0, 0, 0])
        print(self.ccsd.make_rdm2_f().aabb[0, 0, 0, 0])
        for x, y in zip(a, b.aabb):
            x /= np.max(np.abs(x))
            y /= np.max(np.abs(y))
            np.testing.assert_almost_equal(x, y, 6)


@pytest.mark.reference
class UCCSD_PySCF_Tests(unittest.TestCase):
    """Test UCCSD against the PySCF values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0 0 0; O 0 0 1"
        mol.basis = "6-31g"
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

        ccsd = UEBCC(
                mf,
                fermion_excitations="SD",
                log=NullLogger(),
        )
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

    def test_eom_ip(self):
        e1 = self.ccsd.ip_eom(nroot=5).kernel()
        e2, v2 = self.ccsd_ref.ipccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 5)

    def test_eom_ea(self):
        e1 = self.ccsd.ea_eom(nroots=5).kernel()
        e2, v2 = self.ccsd_ref.eaccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 5)

    def test_eom_ee(self):
        e1 = self.ccsd.ee_eom(nroots=5).kernel()
        e2, v2 = self.ccsd_ref.eeccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 5)


if __name__ == "__main__":
    print("Tests for UCCSD")
    unittest.main()
