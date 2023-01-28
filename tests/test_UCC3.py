"""Tests for the UCC3 model.
"""

import itertools
import unittest

import numpy as np
import pytest
from pyscf import gto, lib, scf
import scipy.linalg

from ebcc import UEBCC, GEBCC, NullLogger


@pytest.mark.regression
class UCC3_Tests(unittest.TestCase):
    """Test UCC3 against GCC3.
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

        ucc3 = UEBCC(
                mf,
                ansatz="CC3",
                log=NullLogger(),
        )
        ucc3.options.e_tol = 1e-10
        ucc3.options.t_tol = 1e-8
        ucc3.kernel()

        gcc3 = GEBCC(
                mf,
                ansatz="CC3",
                log=NullLogger(),
        )
        gcc3.options.e_tol = 1e-10
        gcc3.options.t_tol = 1e-8
        gcc3.kernel()

        osort = list(itertools.chain(*zip(range(ucc3.nocc[0], ucc3.nocc[0]+ucc3.nocc[1]), range(ucc3.nocc[0]))))
        vsort = list(itertools.chain(*zip(range(ucc3.nvir[0], ucc3.nvir[0]+ucc3.nvir[1]), range(ucc3.nvir[1]))))

        cls.mf, cls.ucc3, cls.gcc3 = mf, ucc3, gcc3
        cls.osort, cls.vsort = osort, vsort

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ucc3, cls.gcc3
        del cls.osort, cls.vsort

    def test_energy(self):
        a = self.ucc3.e_tot
        b = self.gcc3.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_t1(self):
        a = scipy.linalg.block_diag(self.ucc3.t1.aa, self.ucc3.t1.bb)[self.osort][:, self.vsort]
        b = self.gcc3.t1
        np.testing.assert_almost_equal(a, b, 6)


if __name__ == "__main__":
    print("Tests for UCC3")
    unittest.main()
