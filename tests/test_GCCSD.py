"""Tests for the GCCSD model.
"""

import os
import pickle
import unittest
import tempfile

import numpy as np
import pytest
import scipy.linalg
from types import SimpleNamespace
from pyscf import cc, gto, lib, scf

from ebcc import REBCC, UEBCC, GEBCC, Space, NullLogger, util

# TODO test more excitations in EOM


@pytest.mark.reference
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

        ccsd = GEBCC(
                mf.to_ghf(),  # Direct conversion needed for same ordering as reference data
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
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

    def test_ip_moments(self):
        eom = self.ccsd.ip_eom()
        a = self.data[True]["ip_moms"].transpose(2, 0, 1)
        b = eom.moments(4)
        for x, y in zip(a, b):
            x /= np.max(np.abs(x))
            y /= np.max(np.abs(y))
            np.testing.assert_almost_equal(x, y, 6)

    def test_ea_moments(self):
        eom = self.ccsd.ea_eom()
        a = self.data[True]["ea_moms"].transpose(2, 0, 1)
        b = eom.moments(4)
        for x, y in zip(a, b):
            x /= np.max(np.abs(x))
            y /= np.max(np.abs(y))
            np.testing.assert_almost_equal(x, y, 6)

    def test_ee_moments_diag(self):
        eom = self.ccsd.ee_eom()
        a = self.data[True]["dd_moms"].transpose(4, 0, 1, 2, 3)
        a = np.einsum("npqrs,pq,rs->npqrs", a, np.eye(self.ccsd.nmo), np.eye(self.ccsd.nmo))
        b = eom.moments(4, diagonal_only=True)
        for x, y in zip(a, b):
            x /= np.max(np.abs(x))
            y /= np.max(np.abs(y))
            np.testing.assert_almost_equal(x, y, 6)

    @pytest.mark.regression
    def test_from_rebcc(self):
        mf = self.mf

        gebcc1 = GEBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        gebcc1.options.e_tol = 1e-12
        eris = gebcc1.get_eris()
        gebcc1.kernel(eris=eris)
        gebcc1.solve_lambda(eris=eris)

        rebcc = REBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        rebcc.options.e_tol = 1e-12
        eris = rebcc.get_eris()
        rebcc.kernel(eris=eris)
        rebcc.solve_lambda(eris=eris)
        gebcc2 = GEBCC.from_rebcc(rebcc)

        self.assertAlmostEqual(gebcc1.energy(), gebcc2.energy())
        np.testing.assert_almost_equal(gebcc1.t1, gebcc2.t1, 6)
        np.testing.assert_almost_equal(gebcc1.t2, gebcc2.t2, 6)
        np.testing.assert_almost_equal(gebcc1.l1, gebcc2.l1, 5)
        np.testing.assert_almost_equal(gebcc1.l2, gebcc2.l2, 5)
        np.testing.assert_almost_equal(gebcc1.make_rdm1_f(), gebcc2.make_rdm1_f(), 6)
        np.testing.assert_almost_equal(gebcc1.make_rdm2_f(), gebcc2.make_rdm2_f(), 6)

    @pytest.mark.regression
    def test_from_uebcc(self):
        mf = self.mf.to_uhf()

        gebcc1 = GEBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        gebcc1.options.e_tol = 1e-12
        eris = gebcc1.get_eris()
        gebcc1.kernel(eris=eris)
        gebcc1.solve_lambda(eris=eris)

        uebcc = UEBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        uebcc.options.e_tol = 1e-12
        eris = uebcc.get_eris()
        uebcc.kernel(eris=eris)
        uebcc.solve_lambda(eris=eris)
        gebcc2 = GEBCC.from_uebcc(uebcc)

        self.assertAlmostEqual(gebcc1.energy(), gebcc2.energy())
        np.testing.assert_almost_equal(gebcc1.t1, gebcc2.t1, 6)
        np.testing.assert_almost_equal(gebcc1.t2, gebcc2.t2, 6)
        np.testing.assert_almost_equal(gebcc1.l1, gebcc2.l1, 5)
        np.testing.assert_almost_equal(gebcc1.l2, gebcc2.l2, 5)
        np.testing.assert_almost_equal(gebcc1.make_rdm1_f(), gebcc2.make_rdm1_f(), 6)
        np.testing.assert_almost_equal(gebcc1.make_rdm2_f(), gebcc2.make_rdm2_f(), 6)

    @pytest.mark.regression
    def test_from_rebcc_frozen(self):
        mf = self.mf
        gmf = mf.to_uhf().to_ghf()

        occupied = gmf.mo_occ > 0
        frozen = np.zeros_like(gmf.mo_occ)
        frozen[:2] = True
        active = np.zeros_like(gmf.mo_occ)
        space = Space(occupied, frozen, active)

        gebcc1 = GEBCC(
                gmf,
                ansatz="CCSD",
                space=space,
                log=NullLogger(),
        )
        gebcc1.options.e_tol = 1e-12
        eris = gebcc1.get_eris()
        gebcc1.kernel(eris=eris)
        gebcc1.solve_lambda(eris=eris)

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
        eris = rebcc.get_eris()
        rebcc.kernel(eris=eris)
        rebcc.solve_lambda(eris=eris)
        gebcc2 = GEBCC.from_rebcc(rebcc)

        self.assertAlmostEqual(gebcc1.energy(), gebcc2.energy())
        np.testing.assert_almost_equal(gebcc1.t1, gebcc2.t1, 6)
        np.testing.assert_almost_equal(gebcc1.t2, gebcc2.t2, 6)
        np.testing.assert_almost_equal(gebcc1.l1, gebcc2.l1, 5)
        np.testing.assert_almost_equal(gebcc1.l2, gebcc2.l2, 5)
        np.testing.assert_almost_equal(gebcc1.make_rdm1_f(), gebcc2.make_rdm1_f(), 6)
        np.testing.assert_almost_equal(gebcc1.make_rdm2_f(), gebcc2.make_rdm2_f(), 6)

    @pytest.mark.regression
    def test_from_uebcc_frozen(self):
        mf = self.mf.to_uhf()
        gmf = mf.to_ghf()

        occupied = gmf.mo_occ > 0
        frozen = np.zeros_like(gmf.mo_occ)
        frozen[:2] = True
        active = np.zeros_like(gmf.mo_occ)
        space = Space(occupied, frozen, active)

        gebcc1 = GEBCC(
                mf,
                ansatz="CCSD",
                space=space,
                log=NullLogger(),
        )
        gebcc1.options.e_tol = 1e-12
        eris = gebcc1.get_eris()
        gebcc1.kernel(eris=eris)
        gebcc1.solve_lambda(eris=eris)

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

        uebcc = UEBCC(
                mf,
                ansatz="CCSD",
                space=(space_a, space_b),
                log=NullLogger(),
        )
        uebcc.options.e_tol = 1e-12
        eris = uebcc.get_eris()
        uebcc.kernel(eris=eris)
        uebcc.solve_lambda(eris=eris)
        gebcc2 = GEBCC.from_uebcc(uebcc)

        self.assertAlmostEqual(gebcc1.energy(), gebcc2.energy())
        np.testing.assert_almost_equal(gebcc1.t1, gebcc2.t1, 6)
        np.testing.assert_almost_equal(gebcc1.t2, gebcc2.t2, 6)
        np.testing.assert_almost_equal(gebcc1.l1, gebcc2.l1, 5)
        np.testing.assert_almost_equal(gebcc1.l2, gebcc2.l2, 5)
        np.testing.assert_almost_equal(gebcc1.make_rdm1_f(), gebcc2.make_rdm1_f(), 6)
        np.testing.assert_almost_equal(gebcc1.make_rdm2_f(), gebcc2.make_rdm2_f(), 6)


@pytest.mark.reference
class GCCSD_PySCF_Tests(unittest.TestCase):
    """Test GCCSD against the PySCF GCCSD values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "sto3g"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd_ref = cc.GCCSD(mf.to_uhf().to_ghf())
        ccsd_ref.conv_tol = 1e-12
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = GEBCC(
                mf.to_uhf().to_ghf(),
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

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd_ref.converged)

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
        np.testing.assert_almost_equal(a, b, 6)

    def test_rdm1(self):
        a = self.ccsd_ref.make_rdm1()
        b = self.ccsd.make_rdm1_f(eris=self.eris)
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2()
        b = self.ccsd.make_rdm2_f(eris=self.eris)
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_eom_ip(self):
        e1 = self.ccsd.ip_eom(nroots=5).kernel()
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

    def test_eom_ip_koopmans(self):
        e1 = self.ccsd.ip_eom(nroots=5, koopmans=True).kernel()
        e2, v2 = self.ccsd_ref.ipccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 5)

    def test_eom_ea_koopmans(self):
        e1 = self.ccsd.ea_eom(nroots=5, koopmans=True).kernel()
        e2, v2 = self.ccsd_ref.eaccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 5)

    def test_rdm_energy(self):
        dm1 = self.ccsd.make_rdm1_f()
        dm2 = self.ccsd.make_rdm2_f()
        c = self.mf.to_ghf().mo_coeff
        h = self.mf.to_ghf().get_hcore()
        h = np.linalg.multi_dot((c.T, h, c))
        v = self.ccsd.get_eris().array
        e_rdm = util.einsum("pq,pq->", h, dm1)
        e_rdm += util.einsum("pqrs,pqrs->", v, dm2) * 0.5
        e_rdm += self.mf.mol.energy_nuc()
        self.assertAlmostEqual(e_rdm, self.ccsd_ref.e_tot)


@pytest.mark.reference
class GCCSD_Dump_Tests(GCCSD_PySCF_Tests):
    """Test GCCSD against PySCF after dumping and loading.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "sto3g"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd_ref = cc.GCCSD(mf.to_uhf().to_ghf())
        ccsd_ref.conv_tol = 1e-12
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = GEBCC(
                mf.to_uhf().to_ghf(),
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
    print("Tests for GCCSD")
    unittest.main()

