"""Tests for the UCCSDt' model.
"""

import itertools
import unittest

import numpy as np
import pytest
from pyscf import gto, lib, scf, fci
import scipy.linalg

from ebcc import BACKEND
from ebcc import UEBCC, GEBCC, Space, NullLogger, util

# TODO from http://dx.doi.org/10.1021/acs.jpca.7b10892


@pytest.mark.regression
@pytest.mark.skipif(BACKEND != "numpy", reason="Currently requires mutable backend.")
class UCCSDtp_Tests(unittest.TestCase):
    """Test UCCSDt' against GCCSDt'.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "H 0 0 0; Li 0 0 1"
        mol.basis = "6-311g"
        mol.charge = -1
        mol.spin = 1
        mol.verbose = 0
        mol.build()

        mf = scf.UHF(mol)
        mf.kernel()

        active = [np.zeros_like(o) for o in mf.mo_occ]
        for i in range(2):
            active[i][np.sum(mf.mo_occ[i] > 0) - 1] = active[i][np.sum(mf.mo_occ[i] > 0)] = True
        space = tuple(Space(o > 0, np.zeros_like(a), a) for o, a in zip(mf.mo_occ, active))

        uccsdt = UEBCC(
                mf,
                ansatz="CCSDt'",
                space=space,
                log=NullLogger(),
        )
        uccsdt.options.e_tol = 1e-10
        uccsdt.options.t_tol = 1e-9
        uccsdt.kernel()

        gmf = mf.to_uhf().to_ghf()
        active = np.zeros_like(gmf.mo_occ)
        active[np.where(np.isclose(gmf.mo_energy, mf.mo_energy[0][np.sum(mf.mo_occ[0] > 0) - 1]))[0]] = True
        active[np.where(np.isclose(gmf.mo_energy, mf.mo_energy[1][np.sum(mf.mo_occ[1] > 0) - 1]))[0]] = True
        active[np.where(np.isclose(gmf.mo_energy, mf.mo_energy[0][np.sum(mf.mo_occ[0] > 0)]))[0]] = True
        active[np.where(np.isclose(gmf.mo_energy, mf.mo_energy[1][np.sum(mf.mo_occ[1] > 0)]))[0]] = True
        space = Space(gmf.mo_occ > 0, np.zeros_like(active), active)

        gccsdt = GEBCC(
                gmf,
                ansatz="CCSDt'",
                space=space,
                log=NullLogger(),
        )
        gccsdt.options.e_tol = 1e-10
        gccsdt.options.t_tol = 1e-9
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
        mf.conv_tol = 1e-12
        mf.kernel()

        space = tuple(Space(o > 0, np.zeros_like(o), np.ones_like(o)) for o in mf.mo_occ)

        ccsdt = UEBCC(
                mf,
                ansatz="CCSDt'",
                space=space,
                log=NullLogger(),
        )
        ccsdt.options.e_tol = 1e-8
        ccsdt.options.t_tol = 1e-7
        ccsdt.kernel()
        e1 = ccsdt.e_tot

        ci = fci.FCI(mf)
        ci.conv_tol = 1e-10
        e2 = ci.kernel()[0]

        self.assertAlmostEqual(e1, e2, 8)


if __name__ == "__main__":
    print("Tests for UCCSDt'")
    unittest.main()
