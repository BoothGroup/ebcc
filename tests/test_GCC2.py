"""Tests for the GCC2 model.
"""

import unittest
import itertools

from pyscf import gto, scf, cc, lib
import numpy as np
import scipy.linalg
import pytest

from ebcc import GEBCC, NullLogger, util


@pytest.mark.reference
class GCC2_PySCF_Tests(unittest.TestCase):
    """Test GCC2 against the PySCF values.
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

        ccsd_ref = cc.rccsd.RCCSD(mf)
        ccsd_ref.cc2 = True
        ccsd_ref.conv_tol = 1e-10
        ccsd_ref.conv_tol_normt = 1e-14
        ccsd_ref.max_cycle = 200
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = GEBCC(
                mf.to_ghf(),  # Direct conversion needed for same ordering as reference data
                ansatz="CC2",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        nmo, nocc, nvir = ccsd_ref.nmo, ccsd_ref.nocc, ccsd_ref.nmo-ccsd_ref.nocc
        osort = list(itertools.chain(*zip(range(nocc), range(nocc, 2*nocc))))
        vsort = list(itertools.chain(*zip(range(nvir), range(nvir, 2*nvir))))
        fsort = list(itertools.chain(*zip(range(nmo), range(nmo, 2*nmo))))

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris
        cls.osort, cls.vsort, cls.fsort = osort, vsort, fsort

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris
        del cls.osort, cls.vsort, cls.fsort

    def test_converged(self):
        self.assertTrue(self.ccsd.converged)
        self.assertTrue(self.ccsd.converged_lambda)
        self.assertTrue(self.ccsd_ref.converged)
        self.assertTrue(self.ccsd_ref.converged_lambda)

    def test_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = scipy.linalg.block_diag(self.ccsd_ref.t1, self.ccsd_ref.t1)[self.osort][:, self.vsort]
        b = self.ccsd.t1
        np.testing.assert_almost_equal(a, b, 6)

    # This fails:
    #def test_rdm_energy(self):
    #    dm1 = self.ccsd.make_rdm1_f()
    #    dm2 = self.ccsd.make_rdm2_f()
    #    c = self.mf.to_ghf().mo_coeff
    #    h = self.mf.to_ghf().get_hcore()
    #    h = np.linalg.multi_dot((c.T, h, c))
    #    v = self.ccsd.get_eris().array
    #    e_rdm = util.einsum("pq,pq->", h, dm1)
    #    e_rdm += util.einsum("pqrs,pqrs->", v, dm2) * 0.5
    #    e_rdm += self.mf.mol.energy_nuc()
    #    self.assertAlmostEqual(e_rdm, self.ccsd_ref.e_tot)


@pytest.mark.regression
class GCC2_Tests(unittest.TestCase):
    """Test GCC2 against regression.
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

        ccsd = GEBCC(
                mf,
                ansatz="CC2",
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd, cls.eris = mf, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd, cls.eris

    def test_rdm1_f(self):
        dm = self.ccsd.make_rdm1_f()
        c = self.ccsd.mf.mo_coeff
        dm = util.einsum("ij,pi,qj->pq", dm, c, c)
        self.assertAlmostEqual(lib.fp(dm), 1.672795023689995, 6)

    def test_rdm2_f(self):
        dm = self.ccsd.make_rdm2_f()
        c = self.ccsd.mf.mo_coeff
        dm = util.einsum("ijkl,pi,qj,rk,sl->pqrs", dm, c, c, c, c)
        self.assertAlmostEqual(lib.fp(dm), 2.491733293012602, 6)


if __name__ == "__main__":
    print("Tests for GCC2")
    unittest.main()
