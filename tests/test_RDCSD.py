"""Tests for the RDCSD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf

from ebcc import REBCC, NullLogger, Space


@pytest.mark.reference
class RDCSD_Tests(unittest.TestCase):
    """Test RDCSD against reference values.

    Ref: J. Chem. Phys. 144, 124117 (2016)
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "N 0 0 0; N 0 0 2.118"
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

        dcsd = REBCC(
                mf,
                space=space,
                ansatz="DCSD",
                log=NullLogger(),
        )
        dcsd.options.e_tol = 1e-10
        dcsd.kernel()

        cls.mf, cls.dcsd = mf, dcsd

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.dcsd

    def test_converged(self):
        self.assertTrue(self.dcsd.converged)

    def test_energy(self):
        self.assertAlmostEqual(self.dcsd.e_corr, -0.327591, places=6)


if __name__ == "__main__":
    print("Tests for RDCSD")
    unittest.main()
