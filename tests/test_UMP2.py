"""Tests for the UMP2 model.
"""

import itertools
import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import mp, gto, lib, scf

from ebcc import UEBCC, NullLogger, Space


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
