"""Tests for the RMP2 model.
"""

import itertools
import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import mp, gto, lib, scf, adc

from ebcc import REBCC, NullLogger, Space


@pytest.mark.reference
class RMP2_PySCF_Tests(unittest.TestCase):
    """Test RMP2 against the PySCF values.
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

        mp2_ref = mp.MP2(mf)
        mp2_ref.kernel()

        mp2 = REBCC(
                mf,
                ansatz="MP2",
                log=NullLogger(),
        )
        mp2.kernel()

        cls.mf, cls.mp2_ref, cls.mp2 = mf, mp2_ref, mp2

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.mp2_ref, cls.mp2

    def test_energy(self):
        a = self.mp2_ref.e_tot
        b = self.mp2.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_eom_ip(self):
        eom = self.mp2.ip_eom(nroots=5, e_tol=1e-10)
        e1 = eom.kernel()
        adc2 = adc.ADC(self.mf)
        adc2.kernel_gs()
        e2, v2 = adc2.ip_adc(nroots=5)[:2]
        self.assertAlmostEqual(e1[0], e2[0], 5)

    def test_eom_ea(self):
        eom = self.mp2.ea_eom(nroots=5, e_tol=1e-12)
        e1 = eom.kernel()
        adc2 = adc.ADC(self.mf)
        adc2.kernel_gs()
        e2, v2 = adc2.ea_adc(nroots=5)[:2]
        self.assertAlmostEqual(e1[0], e2[0], 5)


@pytest.mark.reference
class RMP2_PySCF_Frozen_Tests(unittest.TestCase):
    """Test RMP2 against the PySCF values with frozen orbitals.
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

        mp2_ref = mp.MP2(mf)
        mp2_ref.frozen = np.where(frozen)[0]
        mp2_ref.kernel()

        space = Space(
                mf.mo_occ > 0,
                frozen,
                np.zeros_like(mf.mo_occ),
        )

        mp2 = REBCC(
                mf,
                ansatz="MP2",
                space=space,
                log=NullLogger(),
        )
        mp2.kernel()

        cls.mf, cls.mp2_ref, cls.mp2 = mf, mp2_ref, mp2

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.mp2_ref, cls.mp2

    def test_energy(self):
        a = self.mp2_ref.e_tot
        b = self.mp2.e_tot
        self.assertAlmostEqual(a, b, 7)


if __name__ == "__main__":
    print("Tests for RMP2")
    unittest.main()
