"""Tests for Brueckner orbital calculations.
"""

import unittest
import pytest

import numpy as np
from pyscf import cc, gto, lib, scf
#from pyscf.cc.bccd import bccd_kernel_

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
        # TODO when in pyscf release version
        #ccsd_ref = bccd_kernel_(ccsd_ref, verbose=0)
        cls._pyscf_bccd_results = {
                "e_tot": -7.881447504050691,
        }

        ccsd = REBCC(
                mf,
                ansatz="CCSD",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        ccsd.options.max_iter = 250
        ccsd.options.diis_space = 15
        eris = ccsd.get_eris()
        ccsd.brueckner(max_iter=200, diis_space=15)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris
        del cls._pyscf_bccd_results

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd_ref.converged)

    def test_energy(self):
        #a = self.ccsd_ref.e_tot
        a = self._pyscf_bccd_results["e_tot"]
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 7)

    #def test_t2_amplitudes(self):
    #    a = self.ccsd_ref.t2
    #    b = self.ccsd.t2
    #    np.testing.assert_almost_equal(a, b, 6)


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
        # TODO when in pyscf release version
        #ccsd_ref = bccd_kernel_(ccsd_ref, verbose=0)
        cls._pyscf_bccd_results = {
                "e_tot": -7.881227958827942, 
        }

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
        # TODO when in pyscf release version
        #ccsd_ref = bccd_kernel_(ccsd_ref, verbose=0)
        cls._pyscf_bccd_results = {
                "e_tot": -7.998789085509077, 
        }

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
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris
        del cls._pyscf_bccd_results

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd_ref.converged)

    def test_energy(self):
        #a = self.ccsd_ref.e_tot
        a = self._pyscf_bccd_results["e_tot"]
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
        # TODO when in pyscf release version
        #ccsd_ref = bccd_kernel_(ccsd_ref, verbose=0)
        cls._pyscf_bccd_results = {
                "e_tot": -7.998530643494347, 
        }

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
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris
        del cls._pyscf_bccd_results

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd_ref.converged)

    def test_energy(self):
        #a = self.ccsd_ref.e_tot
        a = self._pyscf_bccd_results["e_tot"]
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 8)


if __name__ == "__main__":
    print("Tests for brueckner")
    unittest.main()
