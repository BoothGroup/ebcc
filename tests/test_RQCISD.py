"""Tests for the RQCISD model.
"""

import itertools
import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import cc, gto, lib, scf
from pyscf.cc import qcisd as pyscf_qcisd

from ebcc import REBCC, NullLogger, Space


@pytest.mark.reference
class RQCISD_PySCF_Tests(unittest.TestCase):
    """Test RQCISD against the PySCF values.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0.0 0.0 0.11779; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161"
        #mol.atom = "Li 0 0 0; H 0 0 1.4"
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

        qcisd = REBCC(
                mf,
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

    def test_t1_amplitudes(self):
        a = self.qcisd_ref.t1
        b = self.qcisd.t1
        np.testing.assert_almost_equal(a, b, 6)

    def test_t2_amplitudes(self):
        a = self.qcisd_ref.t2
        b = self.qcisd.t2
        np.testing.assert_almost_equal(a, b, 6)


@pytest.mark.reference
class RQCISD_PySCF_Frozen_Tests(unittest.TestCase):
    """Test RQCISD against the PySCF values with frozen orbitals.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0.0 0.0 0.11779; H 0.0 0.755453 -0.471161; H 0.0 -0.755453 -0.471161"
        mol.basis = "cc-pvdz"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        frozen = np.zeros_like(mf.mo_occ)
        frozen[:2] = 1
        frozen[-1] = 1
        frozen = frozen.astype(bool)

        qcisd_ref = pyscf_qcisd.QCISD(mf)
        qcisd_ref.conv_tol = 1e-10
        qcisd_ref.conv_tol_normt = 1e-14
        qcisd_ref.max_cycle = 200
        qcisd_ref.frozen = np.where(frozen)[0]
        qcisd_ref.kernel()

        space = Space(
                mf.mo_occ > 0,
                frozen,
                np.zeros_like(mf.mo_occ),
        )

        qcisd = REBCC(
                mf,
                ansatz="QCISD",
                space=space,
                log=NullLogger(),
        )
        qcisd.options.e_tol = 1e-13
        eris = qcisd.get_eris()
        qcisd.kernel(eris=eris)

        cls.mf, cls.qcisd_ref, cls.qcisd, cls.eris = mf, qcisd_ref, qcisd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.qcisd_ref, cls.qcisd, cls.eris

    def test_converged(self):
        self.assertTrue(self.qcisd.converged)
        self.assertTrue(self.qcisd_ref.converged)

    def test_energy(self):
        a = self.qcisd_ref.e_tot
        b = self.qcisd.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.qcisd_ref.t1
        b = self.qcisd.t1
        np.testing.assert_almost_equal(a, b, 6)

    def test_t2_amplitudes(self):
        a = self.qcisd_ref.t2
        b = self.qcisd.t2
        np.testing.assert_almost_equal(a, b, 6)


if __name__ == "__main__":
    print("Tests for RQCISD")
    unittest.main()
