"""Tests for the RCCSDt' model.
"""

import itertools
import tempfile
import unittest

import numpy as np
import pytest
import scipy
from pyscf import gto, lib, scf, fci

from ebcc import GEBCC, REBCC, Space, NullLogger, util


@pytest.mark.regression
class RCCSDtp_Tests(unittest.TestCase):
    """Test RCCSDt' against GCCSDt'.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "H 0 0 0; Li 0 0 1"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        active = np.zeros_like(mf.mo_occ, dtype=bool)
        active[np.where(mf.mo_occ > 0)[0][-1]] = True
        active[np.where(mf.mo_occ == 0)[0][0]] = True
        space = Space(
            mf.mo_occ > 0,
            np.zeros_like(mf.mo_occ, dtype=bool),
            active,
        )

        rccsdt = REBCC(
                mf,
                ansatz="CCSDt'",
                space=space,
                log=NullLogger(),
        )
        rccsdt.options.e_tol = 1e-10
        rccsdt.options.t_tol = 1e-8
        rccsdt.kernel()

        gmf = mf.to_ghf()

        active = np.zeros_like(gmf.mo_occ, dtype=bool)
        active[np.isclose(gmf.mo_energy, mf.mo_energy[np.where(mf.mo_occ > 0)[0][-1]])] = True
        active[np.isclose(gmf.mo_energy, mf.mo_energy[np.where(mf.mo_occ == 0)[0][0]])] = True
        space = Space(
            gmf.mo_occ > 0,
            np.zeros_like(gmf.mo_occ, dtype=bool),
            active,
        )

        gccsdt = GEBCC(
                gmf,
                ansatz="CCSDt'",
                space=space,
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


@pytest.mark.regression
class RCCSDtp_Frozen_Tests(unittest.TestCase):
    """Test RCCSDt' against GCCSDt' with a frozen core approximation.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "H 0 0 0; Li 0 0 1"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        active = np.zeros_like(mf.mo_occ, dtype=bool)
        active[np.where(mf.mo_occ > 0)[0][-1]] = True
        active[np.where(mf.mo_occ == 0)[0][0]] = True
        frozen = np.zeros_like(mf.mo_occ, dtype=bool)
        frozen[0] = True
        space = Space(
            mf.mo_occ > 0,
            frozen,
            active,
        )

        rccsdt = REBCC(
                mf,
                ansatz="CCSDt'",
                space=space,
                log=NullLogger(),
        )
        rccsdt.options.e_tol = 1e-10
        rccsdt.options.t_tol = 1e-8
        rccsdt.kernel()

        gmf = mf.to_ghf()

        active = np.zeros_like(gmf.mo_occ, dtype=bool)
        active[np.isclose(gmf.mo_energy, mf.mo_energy[np.where(mf.mo_occ > 0)[0][-1]])] = True
        active[np.isclose(gmf.mo_energy, mf.mo_energy[np.where(mf.mo_occ == 0)[0][0]])] = True
        frozen = np.zeros_like(gmf.mo_occ, dtype=bool)
        frozen[np.isclose(gmf.mo_energy, mf.mo_energy[0])] = True
        space = Space(
            gmf.mo_occ > 0,
            frozen,
            active,
        )

        gccsdt = GEBCC(
                gmf,
                ansatz="CCSDt'",
                space=space,
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


if __name__ == "__main__":
    print("Tests for RCCSDt'")
    unittest.main()
