"""Tests for the UMP2 model.
"""

import itertools
import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import mp, gto, lib, scf, adc

from ebcc import UEBCC, NullLogger, Space, BACKEND


@pytest.mark.reference
class UMP2_PySCF_Tests(unittest.TestCase):
    """Test UMP2 against the PySCF values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0 0 0; O 0 0 1"
        mol.basis = "6-31g"
        mol.spin = 2
        mol.verbose = 0
        mol.build()

        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        mp2_ref = mp.MP2(mf)
        mp2_ref.kernel()

        mp2 = UEBCC(
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

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ip(self):
        eom = self.mp2.ip_eom(nroots=5, e_tol=1e-10)
        e1 = eom.kernel()
        adc2 = adc.ADC(self.mf)
        adc2.kernel_gs()
        e2, v2 = adc2.ip_adc(nroots=5)[:2]
        self.assertAlmostEqual(e1[0], e2[0], 5)

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ea(self):
        eom = self.mp2.ea_eom(nroots=5, e_tol=1e-12)
        e1 = eom.kernel()
        adc2 = adc.ADC(self.mf)
        adc2.kernel_gs()
        e2, v2 = adc2.ea_adc(nroots=5)[:2]
        self.assertAlmostEqual(e1[0], e2[0], 5)

    def test_rdm1(self):
        a = self.mp2.make_rdm1_f()
        b = self.mp2_ref.make_rdm1()
        np.testing.assert_allclose(a.aa, b[0], rtol=1e10, atol=1e-8, verbose=True)
        np.testing.assert_allclose(a.bb, b[1], rtol=1e10, atol=1e-8, verbose=True)

    def test_rdm2(self):
        a = self.mp2.make_rdm2_f()
        b = self.mp2_ref.make_rdm2()
        np.testing.assert_allclose(a.aaaa, b[0], rtol=1e10, atol=1e-8, verbose=True)
        np.testing.assert_allclose(a.aabb, b[1], rtol=1e10, atol=1e-8, verbose=True)
        np.testing.assert_allclose(a.bbbb, b[2], rtol=1e10, atol=1e-8, verbose=True)


@pytest.mark.reference
class UMP2_PySCF_Frozen_Tests(unittest.TestCase):
    """Test UMP2 against the PySCF values with frozen orbitals.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0 0 0; O 0 0 1"
        mol.basis = "6-31g"
        mol.spin = 2
        mol.verbose = 0
        mol.build()

        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        frozen_a = np.zeros_like(mf.mo_occ[0])
        frozen_a[:2] = 1
        frozen_a[-1] = 1
        frozen_a = frozen_a.astype(bool)

        frozen_b = np.zeros_like(mf.mo_occ[1])
        frozen_b[:1] = 1
        frozen_b[-1] = 1
        frozen_b = frozen_b.astype(bool)

        mp2_ref = mp.MP2(mf)
        mp2_ref.frozen = (np.where(frozen_a)[0], np.where(frozen_b)[0])
        mp2_ref.kernel()

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

        mp2 = UEBCC(
                mf,
                ansatz="MP2",
                space=(space_a, space_b),
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
    print("Tests for UMP2")
    unittest.main()
