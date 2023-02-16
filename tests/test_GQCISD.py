"""Tests for the GQCISD model.
"""

import unittest
import pytest

import numpy as np
from pyscf import cc, gto, lib, scf
from pyscf.cc import qcisd as pyscf_qcisd
from pyscf.cc import ccsd_rdm

from ebcc import GEBCC, NullLogger, Space, util


@pytest.mark.reference
class GQCISD_PySCF_Tests(unittest.TestCase):
    """Test GQCISD against PySCF's RQCISD values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.4"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        qcisd_ref = pyscf_qcisd.QCISD(mf)
        qcisd_ref.conv_tol = 1e-10
        qcisd_ref.conv_tol_normt = 1e-14
        qcisd_ref.max_cycle = 200
        qcisd_ref.kernel()

        qcisd = GEBCC(
                mf.to_ghf(),
                ansatz="QCISD",
                log=NullLogger(),
        )
        qcisd.options.e_tol = 1e-10
        eris = qcisd.get_eris()
        qcisd.kernel(eris=eris)

        cls.mf, cls.qcisd_ref, cls.qcisd, cls.eris = mf, qcisd_ref, qcisd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.qcisd_ref, cls.qcisd

    def test_converged(self):
        self.assertTrue(self.qcisd.converged)
        self.assertTrue(self.qcisd_ref.converged)

    def test_energy(self):
        a = self.qcisd_ref.e_tot
        b = self.qcisd.e_tot
        self.assertAlmostEqual(a, b, 7)


@pytest.mark.regression
class GQCISD_Tests(unittest.TestCase):
    """Test GQCISD against regression.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 1.4"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        qcisd = GEBCC(
                mf.to_ghf(),
                ansatz="QCISD",
                log=NullLogger(),
        )
        qcisd.options.e_tol = 1e-10
        eris = qcisd.get_eris()
        qcisd.kernel(eris=eris)

        cls.mf, cls.qcisd, cls.eris = mf, qcisd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.qcisd, cls.eris

    def test_converged(self):
        self.assertTrue(self.qcisd.converged)

    def test_t1(self):
        ci = self.qcisd.mo_coeff[:, self.qcisd.mo_occ > 0]
        ca = self.qcisd.mo_coeff[:, self.qcisd.mo_occ == 0]
        t1 = util.einsum("ia,pi,qa->pq", self.qcisd.t1, ci, ca)
        self.assertAlmostEqual(lib.fp(t1), 0.002669284765052053)

    def test_t2(self):
        ci = self.qcisd.mo_coeff[:, self.qcisd.mo_occ > 0]
        ca = self.qcisd.mo_coeff[:, self.qcisd.mo_occ == 0]
        t2 = util.einsum("ijab,pi,qj,ra,sb->pqrs", self.qcisd.t2, ci, ci, ca, ca)
        self.assertAlmostEqual(lib.fp(t2), -0.0008836462499253184)


if __name__ == "__main__":
    print("Tests for GQCISD")
    unittest.main()
