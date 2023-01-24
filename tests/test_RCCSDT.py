"""Tests for the RCCSDT model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, lib, scf, fci

from ebcc import REBCC, GEBCC, NullLogger

# TODO from http://dx.doi.org/10.1021/acs.jpca.7b10892


@pytest.mark.regression
class RCCSDT_Tests(unittest.TestCase):
    """Test RCCSDT against GCCSDT.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "H 0 0 0; H 0 0 1"
        mol.basis = "6-31g"
        mol.charge = -2
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        rccsdt = REBCC(
                mf,
                ansatz="CCSDT",
                log=NullLogger(),
        )
        rccsdt.options.e_tol = 1e-8
        rccsdt.options.t_tol = 1e-6
        rccsdt.kernel()

        gccsdt = GEBCC(
                mf,
                ansatz="CCSDT",
                log=NullLogger(),
        )
        gccsdt.options.e_tol = 1e-8
        gccsdt.options.t_tol = 1e-6
        gccsdt.kernel()

        cls.mf, cls.rccsdt, cls.gccsdt = mf, rccsdt, gccsdt

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.rccsdt, cls.gccsdt

    def test_energy(self):
        a = self.rccsdt.e_tot
        b = self.gccsdt.e_tot
        self.assertAlmostEqual(a, b, 7)


if __name__ == "__main__":
    print("Tests for RCCSDT")
    unittest.main()
