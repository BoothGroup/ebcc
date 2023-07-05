"""Tests for the GCCSDTQQ model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, lib, scf, fci

from ebcc import GEBCC, NullLogger


@pytest.mark.regression
class GCCSDTQ_Tests(unittest.TestCase):
    """Test GCCSDTQ against regression.
    """

    def test_4_electron_exact(self):
        mol = gto.M(
                atom=";".join(["H 0 0 %.10f" % (i*0.75) for i in range(4)]),
                basis="sto3g",
                verbose=0,
        )
        assert mol.nelectron == 4

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.kernel()

        ccsdt = GEBCC(
                mf,
                ansatz="CCSDTQ",
                e_tol=1e-8,
                t_tol=1e-5,
                log=NullLogger(),
        )
        ccsdt.kernel()
        e1 = ccsdt.e_tot

        ci = fci.FCI(mf)
        ci.conv_tol = 1e-8
        e2 = ci.kernel()[0]

        self.assertAlmostEqual(e1, e2, 6)


if __name__ == "__main__":
    print("Tests for GCCSDTQ")
    unittest.main()
