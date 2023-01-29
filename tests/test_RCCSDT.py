"""Tests for the RCCSDT model.
"""

import itertools
import unittest

import numpy as np
import pytest
from pyscf import gto, lib, scf, fci
import scipy.linalg

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
        rccsdt.options.e_tol = 1e-10
        rccsdt.options.t_tol = 1e-8
        rccsdt.kernel()

        gccsdt = GEBCC(
                mf,
                ansatz="CCSDT",
                log=NullLogger(),
        )
        gccsdt.options.e_tol = 1e-10
        gccsdt.options.t_tol = 1e-8
        gccsdt.kernel()

        osort = list(itertools.chain(*zip(range(rccsdt.nocc), range(rccsdt.nocc, 2*rccsdt.nocc))))
        vsort = list(itertools.chain(*zip(range(rccsdt.nvir), range(rccsdt.nvir, 2*rccsdt.nvir))))
        fsort = list(itertools.chain(*zip(range(rccsdt.nmo), range(rccsdt.nmo, 2*rccsdt.nmo))))

        cls.mf, cls.rccsdt, cls.gccsdt = mf, rccsdt, gccsdt
        cls.osort, cls.vsort, cls.fsort = osort, vsort, fsort

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.rccsdt, cls.gccsdt
        del cls.osort, cls.vsort, cls.fsort

    def test_energy(self):
        a = self.rccsdt.e_tot
        b = self.gccsdt.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_t1(self):
        a = scipy.linalg.block_diag(self.rccsdt.t1, self.rccsdt.t1)[self.osort][:, self.vsort]
        b = self.gccsdt.t1
        np.testing.assert_almost_equal(a, b, 6)


if __name__ == "__main__":
    print("Tests for RCCSDT")
    unittest.main()
