"""Minimal tests for the module.

These tests simply check the energies for small systems.
"""

import unittest
import pytest

from pyscf import gto, scf

from ebcc import REBCC, UEBCC, GEBCC, util, NullLogger
from ebcc import numpy as np


class _Minimal_Tests:
    """Base class for minimal tests."""

    ENERGY_REBCC: float
    ENERGY_UEBCC: float
    ENERGY_GEBCC: float
    ANSATZ: str
    SOLVE_LAMBDA: bool = True
    CHECK_DMS: bool = True

    def test_rebcc(self):
        rebcc = REBCC(self.mf, ansatz=self.ANSATZ, log=NullLogger())
        rebcc.kernel()
        self.assertTrue(rebcc.converged)
        self.assertAlmostEqual(rebcc.e_tot, self.ENERGY_REBCC, places=7)

        if self.CHECK_DMS:
            if self.SOLVE_LAMBDA:
                rebcc.solve_lambda()
            dm1 = rebcc.make_rdm1_f()
            dm2 = rebcc.make_rdm2_f()
            c = rebcc.mf.mo_coeff
            h = rebcc.mf.get_hcore()
            h = c.T @ h @ c
            v = rebcc.get_eris().xxxx
            e_rdm = util.einsum("pq,pq->", h, dm1)
            e_rdm += util.einsum("pqrs,pqrs->", v, dm2) * 0.5
            e_rdm += rebcc.mf.mol.energy_nuc()
            self.assertAlmostEqual(e_rdm, rebcc.e_tot)

        if callable(self.test_uebcc):
            uebcc = UEBCC.from_rebcc(rebcc)
            uebcc.kernel()
            self.assertAlmostEqual(uebcc.e_tot, self.ENERGY_UEBCC, places=7)

    def test_uebcc(self):
        uebcc = UEBCC(self.mf, ansatz=self.ANSATZ, log=NullLogger())
        uebcc.kernel()
        self.assertTrue(uebcc.converged)
        self.assertAlmostEqual(uebcc.e_tot, self.ENERGY_UEBCC, places=7)

        if callable(self.test_gebcc):
            gebcc = GEBCC.from_uebcc(uebcc)
            gebcc.kernel()
            self.assertAlmostEqual(gebcc.e_tot, self.ENERGY_GEBCC, places=7)

    def test_gebcc(self):
        gebcc = GEBCC(self.mf, ansatz=self.ANSATZ, log=NullLogger())
        gebcc.kernel()
        self.assertTrue(gebcc.converged)
        self.assertAlmostEqual(gebcc.e_tot, self.ENERGY_GEBCC, places=7)

        if self.CHECK_DMS:
            if self.SOLVE_LAMBDA:
                gebcc.solve_lambda()
            dm1 = gebcc.make_rdm1_f()
            dm2 = gebcc.make_rdm2_f()
            c = gebcc.mf.mo_coeff
            h = gebcc.mf.get_hcore()
            h = c.T @ h @ c
            v = gebcc.get_eris().xxxx
            e_rdm = util.einsum("pq,pq->", h, dm1)
            e_rdm += util.einsum("pqrs,pqrs->", v, dm2) * 0.5
            e_rdm += gebcc.mf.mol.energy_nuc()
            self.assertAlmostEqual(e_rdm, gebcc.e_tot)


@pytest.mark.reference
class Minimal_LiH_CCSD_Tests(unittest.TestCase, _Minimal_Tests):
    ENERGY_REBCC = -7.881447660132
    ENERGY_UEBCC = -7.881447660219
    ENERGY_GEBCC = -7.881447660088
    ANSATZ = "CCSD"

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = "Li 0 0 0; H 0 0 1.64"
        cls.mol.basis = "sto3g"
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf


@pytest.mark.reference
class Minimal_LiH_CC2_Tests(unittest.TestCase, _Minimal_Tests):
    ENERGY_REBCC = -7.873708239966
    ENERGY_UEBCC = -7.873708239926
    ENERGY_GEBCC = -7.873708239943
    ANSATZ = "CC2"
    CHECK_DMS = False

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = "Li 0 0 0; H 0 0 1.64"
        cls.mol.basis = "sto3g"
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf


@pytest.mark.reference
class Minimal_H2O_DFDCD_Tests(unittest.TestCase, _Minimal_Tests):
    ENERGY_REBCC = -76.121571466693
    ENERGY_UEBCC = -76.121571463778
    ANSATZ = "DCD"
    CHECK_DMS = False

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = "O 0 0 -0.11779; H 0 -0.755453 0.471161; H 0 0.755453 0.471161"
        cls.mol.basis = "6-31g"
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf = cls.mf.density_fit()
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    test_gebcc = None


@pytest.mark.reference
class Minimal_O2_MP2_Tests(unittest.TestCase, _Minimal_Tests):
    ENERGY_REBCC = -147.675973450322
    ENERGY_UEBCC = -147.675973450322
    ANSATZ = "MP2"
    SOLVE_LAMBDA = False

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = "O 0 0 0; O 0 0 1.207"
        cls.mol.basis = "sto3g"
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    test_gebcc = None


@pytest.mark.reference
class Minimal_LiH_CCSDT_Tests(unittest.TestCase, _Minimal_Tests):
    ENERGY_REBCC = -7.8814585781527
    ENERGY_UEBCC = -7.8814585767463
    ANSATZ = "CCSDT"

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = "Li 0 0 0; H 0 0 1.64"
        cls.mol.basis = "sto3g"
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    test_gebcc = None


if __name__ == "__main__":
    print("Minimal tests.")
    unittest.main()
