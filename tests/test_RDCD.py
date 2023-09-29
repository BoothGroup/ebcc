"""Tests for the RDCD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf

from ebcc import REBCC, NullLogger, Space


@pytest.mark.reference
class RDCD_Tests(unittest.TestCase):
    """Test RDCD against reference values.

    Ref: J. Chem. Phys. 144, 124117 (2016)
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "N 0 0 0; N 0 0 2.2"
        mol.unit = "B"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        frozen = np.zeros_like(mf.mo_occ, dtype=bool)
        frozen[:2] = True
        space = Space(mf.mo_occ > 0, frozen, np.zeros_like(frozen))

        dcd = REBCC(
                mf,
                space=space,
                ansatz="DCD",
                log=NullLogger(),
        )
        dcd.options.e_tol = 1e-10
        dcd.kernel()

        cls.mf, cls.dcd = mf, dcd

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.dcd

    def test_converged(self):
        self.assertTrue(self.dcd.converged)

    def test_energy(self):
        self.assertAlmostEqual(self.dcd.e_corr,  -0.33481, places=5)
        self.assertAlmostEqual(self.dcd.e_tot, -109.26792, places=5)


if __name__ == "__main__":
    print("Tests for RDCD")
    unittest.main()
