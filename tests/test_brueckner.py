"""Tests for Brueckner orbital calculations.
"""

import unittest
import pytest

import numpy as np
from pyscf import cc, gto, lib, scf
from pyscf.cc.bccd import bccd_kernel_

from ebcc import REBCC, UEBCC, GEBCC, NullLogger, Space


@pytest.mark.reference
class RBCCD_PySCF_Tests(unittest.TestCase):
    """Test RBCCD against the PySCF values.
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

        ccsd_ref = cc.CCSD(mf)
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.max_cycle = 200
        ccsd_ref.kernel()
        ccsd_ref = bccd_kernel_(ccsd_ref, verbose=0)

        ccsd = REBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.brueckner()

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd_ref.converged)

    def test_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 8)

    def test_t2_amplitudes(self):
        a = self.ccsd_ref.t2
        b = self.ccsd.t2
        np.testing.assert_almost_equal(a, b, 6)


@pytest.mark.reference
class RBCCD_Frozen_PySCF_Tests(RBCCD_PySCF_Tests):
    """Test RBCCD against the PySCF values with a frozen approximation.
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

        ccsd_ref = cc.CCSD(mf)
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.max_cycle = 200
        ccsd_ref.frozen = 1
        ccsd_ref.kernel()
        ccsd_ref = bccd_kernel_(ccsd_ref, verbose=0)

        space = Space(
                mf.mo_occ > 0,
                [True] + [False] * (mf.mo_occ.size - 1),
                [False] * mf.mo_occ.size,
        )

        ccsd = REBCC(
                mf,
                ansatz="CCSD",
                space=space,
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.brueckner()

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris


@pytest.mark.reference
class UBCCD_PySCF_Tests(unittest.TestCase):
    """Test UBCCD against the PySCF values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd_ref = cc.CCSD(mf)
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.max_cycle = 200
        ccsd_ref.kernel()
        ccsd_ref = bccd_kernel_(ccsd_ref, verbose=0)

        ccsd = UEBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.brueckner()

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd_ref.converged)

    def test_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 8)


@pytest.mark.reference
class UBCCD_Frozen_PySCF_Tests(unittest.TestCase):
    """Test UBCCD against the PySCF values with a frozen approximation.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd_ref = cc.CCSD(mf)
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.max_cycle = 200
        ccsd_ref.frozen = 1
        ccsd_ref.kernel()
        ccsd_ref = bccd_kernel_(ccsd_ref, verbose=0)

        space = (
            Space(
                    mf.mo_occ[0] > 0,
                    [True] + [False] * (mf.mo_occ[0].size - 1),
                    [False] * mf.mo_occ[0].size,
            ),
            Space(
                    mf.mo_occ[1] > 0,
                    [True] + [False] * (mf.mo_occ[1].size - 1),
                    [False] * mf.mo_occ[1].size,
            ),
        )

        ccsd = UEBCC(
                mf,
                ansatz="CCSD",
                space=space,
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.brueckner()

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd_ref.converged)

    def test_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 8)


@pytest.mark.reference
class GBCCD_PySCF_Tests(unittest.TestCase):
    """Test GBCCD against the PySCF values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        mol.basis = "sto3g"
        mol.verbose = 0
        mol.build()

        mf = scf.GHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd_ref = cc.CCSD(mf)
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.max_cycle = 200
        ccsd_ref.kernel()
        ccsd_ref = bccd_kernel_(ccsd_ref, verbose=0)

        ccsd = GEBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.brueckner()

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd_ref.converged)

    def test_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 8)


if __name__ == "__main__":
    print("Tests for brueckner")
    unittest.main()
