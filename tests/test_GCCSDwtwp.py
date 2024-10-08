"""Tests for the GCCSDt' model.
"""

import tempfile
import unittest

import numpy as np
import pytest
from pyscf import gto, lib, scf, fci

from ebcc import GEBCC, REBCC, Space, NullLogger, util, BACKEND


@pytest.mark.regression
@pytest.mark.skipif(BACKEND != "numpy", reason="Currently requires mutable backend.")
class GCCSDtp_Tests(unittest.TestCase):
    """Test GCCSDt' against regression.
    """

    def test_3_electron_exact_fully_active(self):
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
        gmf = mf.to_ghf()

        space = Space(
            gmf.mo_occ > 0,
            np.zeros_like(gmf.mo_occ, dtype=bool),
            np.ones_like(gmf.mo_occ, dtype=bool),
        )

        ccsdt = GEBCC(
                gmf,
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

    def test_fully_active(self):
        mol = gto.M(
                atom="H 0 0 0; Li 0 0 1",
                basis="sto3g",
                verbose=0,
        )

        mf = scf.RHF(mol)
        mf.kernel()
        gmf = mf.to_ghf()

        space = Space(
            gmf.mo_occ > 0,
            np.zeros_like(gmf.mo_occ, dtype=bool),
            np.ones_like(gmf.mo_occ, dtype=bool),
        )

        ccsdt = GEBCC(
                gmf,
                ansatz="CCSDt'",
                space=space,
                conv_tol=1e-10,
                log=NullLogger(),
        )
        ccsdt.kernel()
        e1 = ccsdt.e_tot

        ccsdt = GEBCC(
                gmf,
                ansatz="CCSDT",
                space=space,
                conv_tol=1e-10,
                log=NullLogger(),
        )
        ccsdt.kernel()
        e2 = ccsdt.e_tot

        self.assertAlmostEqual(e1, e2, 8)

    def test_none_active(self):
        mol = gto.M(
                atom="H 0 0 0; Li 0 0 1",
                basis="sto3g",
                verbose=0,
        )

        mf = scf.RHF(mol)
        mf.kernel()
        gmf = mf.to_ghf()

        space = Space(
            gmf.mo_occ > 0,
            np.zeros_like(gmf.mo_occ, dtype=bool),
            np.zeros_like(gmf.mo_occ, dtype=bool),
        )

        ccsdt = GEBCC(
                gmf,
                ansatz="CCSDt'",
                space=space,
                conv_tol=1e-10,
                log=NullLogger(),
        )
        ccsdt.kernel()
        e1 = ccsdt.e_tot

        ccsdt = GEBCC(
                gmf,
                ansatz="CCSD",
                space=space,
                conv_tol=1e-10,
                log=NullLogger(),
        )
        ccsdt.kernel()
        e2 = ccsdt.e_tot

        self.assertAlmostEqual(e1, e2, 8)

    def test_dump(self):
        mol = gto.M(
                atom="H 0 0 0; H 0 0 1",
                basis="6-31g",
                spin=1,
                charge=-1,
                verbose=0,
        )

        mf = scf.UHF(mol)
        mf.kernel()
        gmf = mf.to_ghf()

        active = np.zeros_like(gmf.mo_occ, dtype=bool)
        active[np.where(gmf.mo_occ > 0)[0][-1]] = True
        active[np.where(gmf.mo_occ == 0)[0][0]] = True
        space = Space(
            gmf.mo_occ > 0,
            np.zeros_like(gmf.mo_occ, dtype=bool),
            active,
        )

        ccsdt = GEBCC(
                gmf,
                ansatz="CCSDt'",
                space=space,
                conv_tol=1e-10,
                log=NullLogger(),
        )
        ccsdt.kernel()

        file = "%s/ebcc.h5" % tempfile.gettempdir()
        ccsdt.write(file)
        ccsdt_load = GEBCC.read(file, log=NullLogger())

        np.testing.assert_almost_equal(ccsdt_load.t1, ccsdt.t1)
        np.testing.assert_almost_equal(ccsdt_load.t2, ccsdt.t2)
        np.testing.assert_almost_equal(ccsdt_load.t3, ccsdt.t3)

    def test_from_rebcc(self):
        mol = gto.M(
                atom="H 0 0 0; Li 0 0 1",
                basis="sto3g",
                verbose=0,
        )

        mf = scf.RHF(mol)
        mf.kernel()
        gmf = mf.to_uhf().to_ghf()

        active = np.zeros_like(mf.mo_occ, dtype=bool)
        active[np.where(mf.mo_occ > 0)[0][-1]] = True
        active[np.where(mf.mo_occ == 0)[0][0]] = True
        frozen = np.zeros_like(mf.mo_occ, dtype=bool)
        frozen[-1] = True
        rspace = Space(
            mf.mo_occ > 0,
            frozen,
            active,
        )

        active = np.zeros_like(gmf.mo_occ, dtype=bool)
        active[np.isclose(gmf.mo_energy, mf.mo_energy[np.where(mf.mo_occ > 0)[0][-1]])] = True
        active[np.isclose(gmf.mo_energy, mf.mo_energy[np.where(mf.mo_occ == 0)[0][0]])] = True
        frozen = np.zeros_like(gmf.mo_occ, dtype=bool)
        frozen[-1] = frozen[-2] = True
        gspace = Space(
            gmf.mo_occ > 0,
            frozen,
            active,
        )

        gccsdt = GEBCC(
                gmf,
                ansatz="CCSDt'",
                space=gspace,
                conv_tol=1e-10,
                log=NullLogger(),
        )
        gccsdt.kernel()

        rccsdt = REBCC(
                mf,
                ansatz="CCSDt'",
                space=rspace,
                conv_tol=1e-10,
                log=NullLogger(),
        )
        rccsdt.kernel()

        gccsdt_load = GEBCC.from_rebcc(rccsdt)

        self.assertAlmostEqual(gccsdt.e_tot, gccsdt_load.e_tot, 8)
        np.testing.assert_almost_equal(gccsdt_load.t1, gccsdt.t1)
        np.testing.assert_almost_equal(gccsdt_load.t2, gccsdt.t2)
        np.testing.assert_almost_equal(gccsdt_load.t3, gccsdt.t3)



if __name__ == "__main__":
    print("Tests for GCCSDt'")
    unittest.main()
