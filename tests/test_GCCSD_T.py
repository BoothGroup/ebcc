"""Tests for the GCCSD(T) model.
"""

import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from types import SimpleNamespace
from pyscf import cc, gto, lib, scf

from ebcc import GEBCC, NullLogger


@pytest.mark.reference
class GCCSD_T_PySCF_Tests(unittest.TestCase):
    """Test GCCSD(T) against the PySCF GCCSD(T) values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.64"
        #mol.basis = "cc-pvdz"
        mol.basis = "sto3g"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        ccsd_ref = cc.GCCSD(mf)
        ccsd_ref.conv_tol = 1e-12
        ccsd_ref.kernel()
        ccsd_ref.e_corr += ccsd_ref.ccsd_t()

        ccsd = GEBCC(
                mf,
                ansatz="CCSD(T)",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd_ref.converged)

    def test_ccsd_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 8)


if __name__ == "__main__":
    print("Tests for GCCSD(T)")
    unittest.main()

