"""Tests for the RMP3 model.
"""

import itertools
import os
import pickle
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import adc, gto, lib, scf

from ebcc import REBCC, NullLogger, Space


@pytest.mark.reference
class RMP3_PySCF_Tests(unittest.TestCase):
    """Test RMP3 against the PySCF values.
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

        mp3_ref = adc.ADC(mf)
        mp3_ref.method = "adc(3)"
        mp3_ref.kernel_gs()

        mp3 = REBCC(
                mf,
                ansatz="MP3",
                log=NullLogger(),
        )
        mp3.kernel()

        cls.mf, cls.mp3_ref, cls.mp3 = mf, mp3_ref, mp3

    @classmethod
    def teardownclass(cls):
        del cls.mf, cls.mp3_ref, cls.mp3

    def test_energy(self):
        a = self.mp3_ref.e_corr
        b = self.mp3.e_corr
        self.assertAlmostEqual(a, b, 7)


if __name__ == "__main__":
    print("Tests for RMP3")
    unittest.main()
