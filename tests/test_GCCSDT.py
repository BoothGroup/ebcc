"""Tests for the GCCSDT model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, lib, scf, fci

from ebcc import GEBCC, NullLogger, util

# TODO from http://dx.doi.org/10.1021/acs.jpca.7b10892


@pytest.mark.regression
class GCCSDT_Tests(unittest.TestCase):
    """Test GCCSDT against regression.
    """

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

        ccsdt = GEBCC(
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

    def test_rdm_energy(self):
        mol = gto.M(
                atom="H 0 0 0; Li 0 0 1",
                basis="sto3g",
                verbose=0,
        )
        assert mol.nelectron > 3

        mf = scf.RHF(mol)
        mf.kernel()

        ccsdt = GEBCC(
                mf,
                ansatz="CCSDT",
                log=NullLogger(),
        )
        ccsdt.options.e_tol = 1e-10
        ccsdt.kernel()
        ccsdt.solve_lambda()
        dm1 = ccsdt.make_rdm1_f()
        dm2 = ccsdt.make_rdm2_f()

        c = mf.to_ghf().mo_coeff
        h = mf.to_ghf().get_hcore()
        h = np.linalg.multi_dot((c.T, h, c))
        v = ccsdt.get_eris().array
        e_rdm = util.einsum("pq,pq->", h, dm1)
        e_rdm += util.einsum("pqrs,pqrs->", v, dm2) * 0.5
        e_rdm += mol.energy_nuc()

        self.assertAlmostEqual(e_rdm, ccsdt.e_tot, 8)


if __name__ == "__main__":
    print("Tests for GCCSDT")
    unittest.main()
