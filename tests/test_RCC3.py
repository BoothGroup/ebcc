"""Tests for the RCC3 model.
"""

import itertools
import unittest

import numpy as np
import pytest
from pyscf import gto, lib, scf
import scipy.linalg

from ebcc import REBCC, GEBCC, NullLogger


@pytest.mark.regression
class RCC3_Tests(unittest.TestCase):
    """Test RCC3 against GCC3.
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

        rcc3 = REBCC(
                mf,
                ansatz="CC3",
                log=NullLogger(),
        )
        rcc3.options.e_tol = 1e-10
        rcc3.options.t_tol = 1e-8
        rcc3.kernel()

        gcc3 = GEBCC(
                mf,
                ansatz="CC3",
                log=NullLogger(),
        )
        gcc3.options.e_tol = 1e-10
        gcc3.options.t_tol = 1e-8
        gcc3.kernel()

        osort = list(itertools.chain(*zip(range(rcc3.nocc), range(rcc3.nocc, 2*rcc3.nocc))))
        vsort = list(itertools.chain(*zip(range(rcc3.nvir), range(rcc3.nvir, 2*rcc3.nvir))))
        fsort = list(itertools.chain(*zip(range(rcc3.nmo), range(rcc3.nmo, 2*rcc3.nmo))))

        cls.mf, cls.rcc3, cls.gcc3 = mf, rcc3, gcc3
        cls.osort, cls.vsort, cls.fsort = osort, vsort, fsort

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.rcc3, cls.gcc3
        del cls.osort, cls.vsort, cls.fsort

    def test_energy(self):
        a = self.rcc3.e_tot
        b = self.gcc3.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_t1(self):
        a = scipy.linalg.block_diag(self.rcc3.t1, self.rcc3.t1)[self.osort][:, self.vsort]
        b = self.gcc3.t1
        np.testing.assert_almost_equal(a, b, 6)

    # TODO reference test, too slow
    #def test_h2o_ccpvdz(self):
    #    # https://doi.org/10.1063/1.473322
    #    # https://doi.org/10.1063/1.471518
    #    mol = gto.Mole()
    #    mol.atom = "O 0 0 -0.009; H 0 1.515263 -1.058898; H 0 -1.515263 -1.058898"
    #    mol.basis = "cc-pvdz"
    #    mol.unit = "b"
    #    mol.verbose = 0
    #    mol.build()

    #    mf = scf.RHF(mol)
    #    mf.kernel()

    #    self.assertAlmostEqual(mf.e_tot, -76.024039, 6)

    #    cc3 = REBCC(
    #            mf,
    #            ansatz="CC3",
    #            #log=NullLogger(),
    #    )
    #    cc3.kernel()

    #    self.assertAlmostEqual(cc3.e_tot, -76.241274, 6)


if __name__ == "__main__":
    print("Tests for RCC3")
    unittest.main()
