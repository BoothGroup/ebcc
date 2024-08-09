"""Tests for the RDFQCISD model.
"""

import unittest

import numpy as np
import pytest
from pyscf import gto, scf, lib

from ebcc import REBCC, NullLogger, Space


@pytest.mark.regression
class RDFQCISD_Tests(unittest.TestCase):
    """Test RDFQCISD against RQCISD.
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

        qcisd_ref = REBCC(
                mf,
                ansatz="QCISD",
                log=NullLogger(),
        )
        qcisd_ref.options.e_tol = 1e-10
        eris = qcisd_ref.get_eris()
        qcisd_ref.kernel(eris=eris)

        mf = mf.density_fit(auxbasis="aug-cc-pvqz-ri")

        qcisd = REBCC(
                mf,
                ansatz="QCISD",
                log=NullLogger(),
        )
        qcisd.options.e_tol = 1e-10
        eris = qcisd.get_eris()
        qcisd.kernel(eris=eris)

        cls.mf, cls.qcisd_ref, cls.qcisd = mf, qcisd_ref, qcisd

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.qcisd_ref, cls.qcisd

    def test_converged(self):
        self.assertTrue(self.qcisd.converged)
        self.assertTrue(self.qcisd_ref.converged)

    def test_energy(self):
        a = self.qcisd_ref.e_tot
        b = self.qcisd.e_tot
        self.assertAlmostEqual(a, b, 5)

    def test_t1_amplitudes(self):
        a = self.qcisd_ref.t1
        b = self.qcisd.t1
        np.testing.assert_almost_equal(a, b, 4)

    def test_t2_amplitudes(self):
        a = self.qcisd_ref.t2
        b = self.qcisd.t2
        np.testing.assert_almost_equal(a, b, 4)


if __name__ == "__main__":
    print("Tests for RDFQCISD")
    unittest.main()
