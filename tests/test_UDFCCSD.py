"""Tests for the UDFCCSD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import cc, gto, lib, scf

from ebcc import UEBCC, NullLogger, Space


@pytest.mark.reference
class UDFCCSD_PySCF_Tests(unittest.TestCase):
    """Test UDFCCSD against the PySCF values.
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
        mf = mf.density_fit()
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


@pytest.mark.reference
class UDFCCSD_PySCF_Frozen_Tests(unittest.TestCase):
    """Test UDFCCSD against the PySCF values with frozen orbitals.
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
        mf = mf.density_fit()
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


if __name__ == "__main__":
    print("Tests for UDFCCSD")
    unittest.main()
