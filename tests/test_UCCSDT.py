"""Tests for the UCCSDT model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, lib, scf, fci

from ebcc import UEBCC, GEBCC, NullLogger

# TODO from http://dx.doi.org/10.1021/acs.jpca.7b10892


@pytest.mark.regression
class UCCSDT_Tests(unittest.TestCase):
    """Test UCCSDT against GCCSDT.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "H 0 0 0; H 0 0 1"
        mol.basis = "6-31g"
        mol.charge = -2
        mol.verbose = 0
        mol.build()

        mf = scf.UHF(mol)
        mf.kernel()

        uccsdt = UEBCC(
                mf,
                ansatz="CCSDT",
                log=NullLogger(),
        )
        uccsdt.options.e_tol = 1e-8
        uccsdt.options.t_tol = 1e-6
        uccsdt.kernel()

        gccsdt = GEBCC(
                mf,
                ansatz="CCSDT",
                log=NullLogger(),
        )
        gccsdt.options.e_tol = 1e-8
        gccsdt.options.t_tol = 1e-6
        gccsdt.kernel()

        cls.mf, cls.uccsdt, cls.gccsdt = mf, uccsdt, gccsdt

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.uccsdt, cls.gccsdt

    def test_energy(self):
        a = self.uccsdt.e_tot
        b = self.gccsdt.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_3_electron_exact(self):
        mol = gto.M(
                atom="H 0 0 0; H 0 0 1",
                basis="6-31g",
                spin=1,
                charge=-1,
                verbose=0,
        )
        assert mol.nelectron == 3

        mf = scf.UHF(mol)
        mf.kernel()

        ccsdt = UEBCC(
                mf,
                ansatz="CCSDT",
                log=NullLogger(),
        )
        ccsdt.kernel()
        e1 = ccsdt.e_tot

        ci = fci.FCI(mf)
        ci.conv_tol = 1e-10
        e2 = ci.kernel()[0]

        self.assertAlmostEqual(e1, e2, 6)


if __name__ == "__main__":
    print("Tests for UCCSDT")
    unittest.main()
