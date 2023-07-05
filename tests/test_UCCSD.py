"""Tests for the UCCSD model.
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

from ebcc import REBCC, UEBCC, GEBCC, NullLogger, Space


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
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
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
                ansatz="CCSD",
                log=NullLogger(),
        )
        rebcc.options.e_tol = 1e-12
        rebcc.kernel()
        rebcc.solve_lambda()
        uebcc = UEBCC.from_rebcc(rebcc)
        self.assertAlmostEqual(self.ccsd.energy(), uebcc.energy(), 8)
        np.testing.assert_almost_equal(self.ccsd.t1.aa, uebcc.t1.aa, 6)
        np.testing.assert_almost_equal(self.ccsd.t1.bb, uebcc.t1.bb, 6)
        np.testing.assert_almost_equal(self.ccsd.t2.aaaa, uebcc.t2.aaaa, 6)
        np.testing.assert_almost_equal(self.ccsd.t2.abab, uebcc.t2.abab, 6)
        np.testing.assert_almost_equal(self.ccsd.t2.bbbb, uebcc.t2.bbbb, 6)
        np.testing.assert_almost_equal(self.ccsd.l1.aa, uebcc.l1.aa, 5)
        np.testing.assert_almost_equal(self.ccsd.l1.bb, uebcc.l1.bb, 5)
        np.testing.assert_almost_equal(self.ccsd.l2.aaaa, uebcc.l2.aaaa, 5)
        np.testing.assert_almost_equal(self.ccsd.l2.abab, uebcc.l2.abab, 5)
        np.testing.assert_almost_equal(self.ccsd.l2.bbbb, uebcc.l2.bbbb, 5)

    def test_from_rebcc_frozen(self):
        mf = self.mf.to_rhf()
        occupied = mf.mo_occ > 0
        frozen = np.zeros_like(mf.mo_occ)
        frozen[0] = True
        active = np.zeros_like(mf.mo_occ)
        space = Space(occupied, frozen, active)

        rebcc = REBCC(
                mf,
                ansatz="CCSD",
                space=space,
                log=NullLogger(),
        )
        rebcc.options.e_tol = 1e-12
        rebcc.kernel()
        rebcc.solve_lambda()

        uebcc_1 = UEBCC.from_rebcc(rebcc)

        mf = self.mf.to_uhf()
        occupied = mf.mo_occ[0] > 0
        frozen = np.zeros_like(mf.mo_occ[0])
        frozen[0] = True
        active = np.zeros_like(mf.mo_occ[0])
        space_a = Space(occupied, frozen, active)
        occupied = mf.mo_occ[1] > 0
        frozen = np.zeros_like(mf.mo_occ[1])
        frozen[0] = True
        active = np.zeros_like(mf.mo_occ[1])
        space_b = Space(occupied, frozen, active)

        uebcc_2 = UEBCC(
                mf,
                ansatz="CCSD",
                space=(space_a, space_b),
                log=NullLogger(),
        )
        uebcc_2.options.e_tol = 1e-12
        uebcc_2.kernel()
        uebcc_2.solve_lambda()

        self.assertAlmostEqual(uebcc_2.energy(), uebcc_1.energy(), 8)
        np.testing.assert_almost_equal(uebcc_2.t1.aa, uebcc_1.t1.aa, 6)
        np.testing.assert_almost_equal(uebcc_2.t1.bb, uebcc_1.t1.bb, 6)
        np.testing.assert_almost_equal(uebcc_2.t2.aaaa, uebcc_1.t2.aaaa, 6)
        np.testing.assert_almost_equal(uebcc_2.t2.abab, uebcc_1.t2.abab, 6)
        np.testing.assert_almost_equal(uebcc_2.t2.bbbb, uebcc_1.t2.bbbb, 6)
        np.testing.assert_almost_equal(uebcc_2.l1.aa, uebcc_1.l1.aa, 5)
        np.testing.assert_almost_equal(uebcc_2.l1.bb, uebcc_1.l1.bb, 5)
        np.testing.assert_almost_equal(uebcc_2.l2.aaaa, uebcc_1.l2.aaaa, 5)
        np.testing.assert_almost_equal(uebcc_2.l2.abab, uebcc_1.l2.abab, 5)
        np.testing.assert_almost_equal(uebcc_2.l2.bbbb, uebcc_1.l2.bbbb, 5)

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
        nmo = self.ccsd.nmo
        a = self.data[True]["dd_moms"].transpose(4, 0, 1, 2, 3)
        a = np.einsum("npqrs,pq,rs->npqrs", a, np.eye(nmo*2), np.eye(nmo*2))
        t = eom.moments(4, diagonal_only=True)
        b = np.zeros_like(a)
        for i in range(a.shape[0]):
            b[i, :nmo, :nmo, :nmo, :nmo] = t.aaaa[i]
            b[i, :nmo, :nmo, nmo:, nmo:] = t.aabb[i]
            b[i, nmo:, nmo:, :nmo, :nmo] = t.aabb.transpose(2, 3, 0, 1)[i]
            b[i, nmo:, nmo:, nmo:, nmo:] = t.bbbb[i]
        b = b[:, self.fsort][:, :, self.fsort][:, :, :, self.fsort][:, :, :, :, self.fsort]
        for x, y in zip(a, b):
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
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

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


@pytest.mark.reference
class UCCSD_PySCF_Frozen_Tests(unittest.TestCase):
    """Test UCCSD against the PySCF values with frozen orbitals.
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

        frozen_a = np.zeros_like(mf.mo_occ[0])
        frozen_a[:2] = 1
        frozen_a[-1] = 1
        frozen_a = frozen_a.astype(bool)

        frozen_b = np.zeros_like(mf.mo_occ[1])
        frozen_b[:1] = 1
        frozen_b[-1] = 1
        frozen_b = frozen_b.astype(bool)

        ccsd_ref = cc.UCCSD(mf)
        ccsd_ref.conv_tol = 1e-12
        ccsd_ref.conv_tol_normt = 1e-12
        ccsd_ref.frozen = (np.where(frozen_a)[0], np.where(frozen_b)[0])
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        space_a = Space(
                mf.mo_occ[0] > 0,
                frozen_a,
                np.zeros_like(mf.mo_occ[0]),
        )

        space_b = Space(
                mf.mo_occ[1] > 0,
                frozen_b,
                np.zeros_like(mf.mo_occ[1]),
        )

        ccsd = UEBCC(
                mf,
                ansatz="CCSD",
                space=(space_a, space_b),
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

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
        a = self.ccsd_ref.make_rdm1(with_frozen=False)
        b = self.ccsd.make_rdm1_f(eris=self.eris)
        np.testing.assert_almost_equal(a[0], b.aa, 6)
        np.testing.assert_almost_equal(a[1], b.bb, 6)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2(with_frozen=False)
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


@pytest.mark.reference
class UCCSD_Dump_Tests(UCCSD_PySCF_Tests):
    """Test UCCSD against the PySCF after dumping and loading.
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
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

        file = "%s/ebcc.h5" % tempfile.gettempdir()
        cls.ccsd.write(file)
        cls.ccsd = cls.ccsd.__class__.read(file, log=cls.ccsd.log)


if __name__ == "__main__":
    print("Tests for UCCSD")
    unittest.main()
