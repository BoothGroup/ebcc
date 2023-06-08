"""Tests for the RCCSD model.
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

from ebcc import REBCC, NullLogger, Space


@pytest.mark.reference
class RCCSD_Tests(unittest.TestCase):
    """Test RCCSD against the legacy GCCSD values.
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

        ccsd = REBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
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
        del cls.osort, cls.vsort, cls.fsort

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
        a = self.data[True]["rdm1_f"]
        b = scipy.linalg.block_diag(rdm1_f, rdm1_f) / 2
        b = b[self.fsort][:, self.fsort]
        np.testing.assert_almost_equal(a, b, 6)

    def test_l1_amplitudes(self):
        a = self.data[True]["l1"]
        b = scipy.linalg.block_diag(self.ccsd.l1, self.ccsd.l1)[self.vsort][:, self.osort]
        np.testing.assert_almost_equal(a, b, 6)

    #def test_ip_moments(self):
    #    eom = self.ccsd.ip_eom()
    #    ip_moms = eom.moments(4)
    #    a = self.data[True]["ip_moms"].transpose(2, 0, 1)
    #    b = np.array([scipy.linalg.block_diag(x, x) for x in ip_moms])
    #    b = b[:, self.fsort][:, :, self.fsort]
    #    for x, y in zip(a, b):
    #        print(
    #            np.allclose(x[:self.ccsd.nocc*2, :self.ccsd.nocc*2], y[:self.ccsd.nocc*2, :self.ccsd.nocc*2]),
    #            np.allclose(x[self.ccsd.nocc*2:, :self.ccsd.nocc*2], y[self.ccsd.nocc*2:, :self.ccsd.nocc*2]),
    #            np.allclose(x[:self.ccsd.nocc*2, self.ccsd.nocc*2:], y[:self.ccsd.nocc*2, self.ccsd.nocc*2:]),
    #            np.allclose(x[self.ccsd.nocc*2:, self.ccsd.nocc*2:], y[self.ccsd.nocc*2:, self.ccsd.nocc*2:]),
    #        )
    #        print(
    #            np.sum(x[:self.ccsd.nocc*2, :self.ccsd.nocc*2]), np.sum(y[:self.ccsd.nocc*2, :self.ccsd.nocc*2]), "\n",
    #            np.sum(x[self.ccsd.nocc*2:, :self.ccsd.nocc*2]), np.sum(y[self.ccsd.nocc*2:, :self.ccsd.nocc*2]), "\n",
    #            np.sum(x[:self.ccsd.nocc*2, self.ccsd.nocc*2:]), np.sum(y[:self.ccsd.nocc*2, self.ccsd.nocc*2:]), "\n",
    #            np.sum(x[self.ccsd.nocc*2:, self.ccsd.nocc*2:]), np.sum(y[self.ccsd.nocc*2:, self.ccsd.nocc*2:]),
    #        )
    #        np.set_printoptions(edgeitems=1000, linewidth=1000, precision=4)
    #        print(x[:self.ccsd.nocc*2, :self.ccsd.nocc*2])
    #        print(y[:self.ccsd.nocc*2, :self.ccsd.nocc*2])
    #        x /= np.max(np.abs(x))
    #        y /= np.max(np.abs(y))
    #        np.testing.assert_almost_equal(x, y, 6)

    #def test_ea_moments(self):
    #    eom = self.ccsd.ea_eom()
    #    ea_moms = eom.moments(4)
    #    a = self.data[True]["ea_moms"].transpose(2, 0, 1)
    #    b = np.array([scipy.linalg.block_diag(x, x) for x in ea_moms])
    #    b = b[:, self.fsort][:, :, self.fsort]
    #    for x, y in zip(a, b):
    #        print(
    #            np.allclose(x[:self.ccsd.nocc*2, :self.ccsd.nocc*2], y[:self.ccsd.nocc*2, :self.ccsd.nocc*2]),
    #            np.allclose(x[self.ccsd.nocc*2:, :self.ccsd.nocc*2], y[self.ccsd.nocc*2:, :self.ccsd.nocc*2]),
    #            np.allclose(x[:self.ccsd.nocc*2, self.ccsd.nocc*2:], y[:self.ccsd.nocc*2, self.ccsd.nocc*2:]),
    #            np.allclose(x[self.ccsd.nocc*2:, self.ccsd.nocc*2:], y[self.ccsd.nocc*2:, self.ccsd.nocc*2:]),
    #        )
    #        x /= np.max(np.abs(x))
    #        y /= np.max(np.abs(y))
    #        np.testing.assert_almost_equal(x, y, 6)

    #def test_ip_1mom(self):
    #    ip_1mom = self.ccsd.make_ip_1mom()
    #    a = self.data[True]["ip_1mom"]
    #    b = scipy.linalg.block_diag(ip_1mom, ip_1mom)
    #    np.testing.assert_almost_equal(a, b, 6)

    #def test_ea_1mom(self):
    #    ea_1mom = self.ccsd.make_ea_1mom()
    #    a = self.data[True]["ea_1mom"]
    #    b = sceay.linalg.block_diag(ea_1mom, ea_1mom)
    #    np.testing.assert_almost_equal(a, b, 6)


@pytest.mark.reference
class RCCSD_PySCF_Tests(unittest.TestCase):
    """Test RCCSD against the PySCF values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0.0 0.0 0.11779; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161"
        #mol.atom = "Li 0 0 0; H 0 0 1.4"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd_ref = cc.CCSD(mf)
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.conv_tol_normt = 1e-14
        ccsd_ref.max_cycle = 200
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = REBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd.converged_lambda)
        self.assertTrue(self.ccsd_ref.converged)
        self.assertTrue(self.ccsd_ref.converged_lambda)

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
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2()
        b = self.ccsd.make_rdm2_f(eris=self.eris)
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    #def test_eom_ip(self):
    #    eom = self.ccsd.ip_eom(nroots=5)
    #    e1 = eom.kernel()
    #    e2, v2 = self.ccsd_ref.ipccsd(nroots=5)
    #    self.assertAlmostEqual(e1[0], e2[0], 6)

    #def test_eom_ea(self):
    #    eom = self.ccsd.ea_eom(nroots=5)
    #    e1 = eom.kernel()
    #    e2, v2 = self.ccsd_ref.eaccsd(nroots=5)
    #    self.assertAlmostEqual(e1[0], e2[0], 6)


@pytest.mark.reference
class RCCSD_PySCF_Frozen_Tests(unittest.TestCase):
    """Test RCCSD against the PySCF values with frozen orbitals.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0.0 0.0 0.11779; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        frozen = np.zeros_like(mf.mo_occ)
        frozen[:2] = 1
        frozen[-1] = 1
        frozen = frozen.astype(bool)

        ccsd_ref = cc.CCSD(mf)
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.conv_tol_normt = 1e-14
        ccsd_ref.max_cycle = 200
        ccsd_ref.frozen = np.where(frozen)[0]
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        space = Space(
                mf.mo_occ > 0,
                frozen,
                np.zeros_like(mf.mo_occ),
        )

        ccsd = REBCC(
                mf,
                ansatz="CCSD",
                space=space,
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-13
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd.converged_lambda)
        self.assertTrue(self.ccsd_ref.converged)
        self.assertTrue(self.ccsd_ref.converged_lambda)

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
        a = self.ccsd_ref.make_rdm1(with_frozen=False)
        b = self.ccsd.make_rdm1_f(eris=self.eris)
        np.testing.assert_almost_equal(a, b, 6, verbose=True)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2(with_frozen=False)
        b = self.ccsd.make_rdm2_f(eris=self.eris)
        np.testing.assert_almost_equal(a, b, 6, verbose=True)


@pytest.mark.reference
class RCCSD_Dump_Tests(RCCSD_PySCF_Tests):
    """Test RCCSD against PySCF after dumping and loading.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0.0 0.0 0.11779; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161"
        #mol.atom = "Li 0 0 0; H 0 0 1.4"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd_ref = cc.CCSD(mf)
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.conv_tol_normt = 1e-14
        ccsd_ref.max_cycle = 200
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = REBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

        file = "%s/ebcc.h5" % tempfile.gettempdir()
        cls.ccsd.write(file)
        cls.ccsd = cls.ccsd.__class__.read(file, log=cls.ccsd.log)


if __name__ == "__main__":
    print("Tests for RCCSD")
    unittest.main()
