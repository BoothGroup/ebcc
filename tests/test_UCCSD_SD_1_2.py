"""Tests for the UCCSD-SD-1-2 model.
"""

import itertools
import os
import pickle
import unittest
import tempfile

import numpy as np
import pytest
import scipy.linalg
from pyscf import cc, gto, lib, scf

from ebcc import REBCC, UEBCC, GEBCC, NullLogger


@pytest.mark.reference
class UCCSD_SD_1_2_Tests(unittest.TestCase):
    """Test UCCSD-SD-1-2 against the legacy GCCSD-SD-SD values with
    shift=True. The system is a singlet.
    """

    shift = True

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
            mo_coeff = data["mo_coeff"]
            data = data[(2, 2, 2)]

        mol = gto.Mole()
        mol.atom = "H 0 0 0; F 0 0 1.1"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        mf.mo_coeff = mo_coeff
        mf = mf.to_uhf()

        nmo = mf.mo_occ[0].size
        nbos = 5
        np.random.seed(12345)
        g = np.random.random((nbos, nmo, nmo)) * 0.02
        g = 0.5 * (g + g.transpose(0, 2, 1).conj())
        omega = np.random.random((nbos,)) * 5.0

        ccsd = UEBCC(
                mf,
                ansatz="CCSD-SD-1-2",
                g=g,
                omega=omega,
                shift=cls.shift,
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        osort = list(itertools.chain(*zip(range(ccsd.nocc[0]), range(ccsd.nocc[0], ccsd.nocc[0]+ccsd.nocc[1]))))
        vsort = list(itertools.chain(*zip(range(ccsd.nvir[1]), range(ccsd.nvir[0], ccsd.nvir[0]+ccsd.nvir[1]))))
        fsort = list(itertools.chain(*zip(range(ccsd.nmo), range(ccsd.nmo, 2*ccsd.nmo))))

        cls.mf, cls.ccsd, cls.eris, cls.data = mf, ccsd, eris, data
        cls.osort, cls.vsort, cls.fsort = osort, vsort, fsort
        cls.g, cls.omega = g, omega

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd, cls.eris, cls.data
        del cls.osort, cls.vsort, cls.fsort
        del cls.g, cls.omega

    def test_const(self):
        a = self.data[self.shift]["const"]
        b = self.ccsd.const
        self.assertAlmostEqual(a, b, 7)

    def test_xi(self):
        a = self.data[self.shift]["xi"]
        b = self.ccsd.xi
        np.testing.assert_almost_equal(a, b, 7)

    def test_energy(self):
        a = self.data[self.shift]["e_corr"]
        b = self.ccsd.e_corr
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.data[self.shift]["t1"]
        b = scipy.linalg.block_diag(self.ccsd.t1.aa, self.ccsd.t1.bb)[self.osort][:, self.vsort]
        np.testing.assert_almost_equal(a, b, 6)

    def test_s1_amplitudes(self):
        a = self.data[self.shift]["s1"]
        b = self.ccsd.amplitudes["s1"]
        np.testing.assert_almost_equal(a, b, 6)

    def test_s1_amplitudes(self):
        a = self.data[self.shift]["s2"]
        b = self.ccsd.amplitudes["s2"]
        np.testing.assert_almost_equal(a, b, 6)

    def test_u11_amplitudes(self):
        a = self.data[self.shift]["u11"]
        b = self.ccsd.amplitudes["u11"]
        b = np.array([scipy.linalg.block_diag(x, y) for x, y in zip(b.aa, b.bb)])
        b = b[:, self.osort][:, :, self.vsort]
        np.testing.assert_almost_equal(a, b, 6)

    @pytest.mark.regression
    def test_lambdas(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"].aa),   -0.05224794749064107, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"].bb),   -0.05224794749064056, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].aaaa),  0.02916600243912396, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].abab), -0.05259350959254097, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].bbbb),  0.02916600243912393, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),      0.01144092890558732, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),      0.01144092890558732, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),      0.00112151519068901, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),      0.00112151519068901, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"].aa),  0.02324028581663984, 5)

    @pytest.mark.regression
    def test_dms(self):
        rdm1_f = self.ccsd.make_rdm1_f()
        rdm2_f = self.ccsd.make_rdm2_f()
        rdm1_b = self.ccsd.make_rdm1_b()
        dm_b = self.ccsd.make_sing_b_dm()
        dm_eb = self.ccsd.make_eb_coup_rdm()
        self.assertAlmostEqual(lib.fp(rdm1_f.aa),     1.529259451075522, 6)
        self.assertAlmostEqual(lib.fp(rdm1_f.bb),     1.529259451075522, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.aaaa), -12.786827602840134, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.aabb),  11.890660368984713, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.bbbb), -12.786827602840136, 6)
        self.assertAlmostEqual(lib.fp(rdm1_b),        0.035528092227729, 6)
        self.assertAlmostEqual(lib.fp(dm_b),          0.020426308220055, 6)
        self.assertAlmostEqual(lib.fp(dm_eb.aa),     -1.053786451784421, 6)
        self.assertAlmostEqual(lib.fp(dm_eb.bb),     -1.053786451784395, 6)

    @pytest.mark.regression
    def test_from_rebcc(self):
        uebcc1 = UEBCC(
                self.mf,
                ansatz="CCSD-SD-1-2",
                g=self.g,
                omega=self.omega,
                shift=self.shift,
                log=NullLogger(),
        )
        uebcc1.options.e_tol = 1e-12
        eris = uebcc1.get_eris()
        uebcc1.kernel(eris=eris)
        uebcc1.solve_lambda(eris=eris)

        rebcc = REBCC(
                self.mf,
                ansatz="CCSD-SD-1-2",
                g=self.g,
                omega=self.omega,
                shift=self.shift,
                log=NullLogger(),
        )
        rebcc.options.e_tol = 1e-12
        eris = rebcc.get_eris()
        rebcc.kernel(eris=eris)
        rebcc.solve_lambda(eris=eris)
        uebcc2 = UEBCC.from_rebcc(rebcc)

        np.testing.assert_almost_equal(uebcc1.amplitudes["t1"].aa,   uebcc2.amplitudes["t1"].aa,   5)
        np.testing.assert_almost_equal(uebcc1.amplitudes["t1"].bb,   uebcc2.amplitudes["t1"].bb,   5)
        np.testing.assert_almost_equal(uebcc1.amplitudes["t2"].aaaa, uebcc2.amplitudes["t2"].aaaa, 5)
        np.testing.assert_almost_equal(uebcc1.amplitudes["t2"].abab, uebcc2.amplitudes["t2"].abab, 5)
        np.testing.assert_almost_equal(uebcc1.amplitudes["t2"].bbbb, uebcc2.amplitudes["t2"].bbbb, 5)
        np.testing.assert_almost_equal(uebcc1.amplitudes["s1"],      uebcc2.amplitudes["s1"],      5)
        np.testing.assert_almost_equal(uebcc1.amplitudes["s1"],      uebcc2.amplitudes["s1"],      5)
        np.testing.assert_almost_equal(uebcc1.amplitudes["s2"],      uebcc2.amplitudes["s2"],      5)
        np.testing.assert_almost_equal(uebcc1.amplitudes["s2"],      uebcc2.amplitudes["s2"],      5)
        np.testing.assert_almost_equal(uebcc1.amplitudes["u11"].aa,  uebcc2.amplitudes["u11"].aa,  5)
        np.testing.assert_almost_equal(uebcc1.amplitudes["u11"].bb,  uebcc2.amplitudes["u11"].bb,  5)

        np.testing.assert_almost_equal(uebcc1.lambdas["l1"].aa,   uebcc2.lambdas["l1"].aa,   5)
        np.testing.assert_almost_equal(uebcc1.lambdas["l1"].bb,   uebcc2.lambdas["l1"].bb,   5)
        np.testing.assert_almost_equal(uebcc1.lambdas["l2"].aaaa, uebcc2.lambdas["l2"].aaaa, 5)
        np.testing.assert_almost_equal(uebcc1.lambdas["l2"].abab, uebcc2.lambdas["l2"].abab, 5)
        np.testing.assert_almost_equal(uebcc1.lambdas["l2"].bbbb, uebcc2.lambdas["l2"].bbbb, 5)
        np.testing.assert_almost_equal(uebcc1.lambdas["ls1"],     uebcc2.lambdas["ls1"],     5)
        np.testing.assert_almost_equal(uebcc1.lambdas["ls1"],     uebcc2.lambdas["ls1"],     5)
        np.testing.assert_almost_equal(uebcc1.lambdas["ls2"],     uebcc2.lambdas["ls2"],     5)
        np.testing.assert_almost_equal(uebcc1.lambdas["ls2"],     uebcc2.lambdas["ls2"],     5)
        np.testing.assert_almost_equal(uebcc1.lambdas["lu11"].aa, uebcc2.lambdas["lu11"].aa, 5)
        np.testing.assert_almost_equal(uebcc1.lambdas["lu11"].bb, uebcc2.lambdas["lu11"].bb, 5)

        uebcc1_rdm1_f = uebcc1.make_rdm1_f()
        uebcc1_rdm2_f = uebcc1.make_rdm2_f()
        uebcc1_rdm1_b = uebcc1.make_rdm1_b()
        uebcc1_dm_b = uebcc1.make_sing_b_dm()
        uebcc1_dm_eb = uebcc1.make_eb_coup_rdm()
        uebcc2_rdm1_f = uebcc2.make_rdm1_f()
        uebcc2_rdm2_f = uebcc2.make_rdm2_f()
        uebcc2_rdm1_b = uebcc2.make_rdm1_b()
        uebcc2_dm_b = uebcc2.make_sing_b_dm()
        uebcc2_dm_eb = uebcc2.make_eb_coup_rdm()

        np.testing.assert_almost_equal(uebcc1_rdm1_f.aa,   uebcc2_rdm1_f.aa,   6)
        np.testing.assert_almost_equal(uebcc1_rdm1_f.bb,   uebcc2_rdm1_f.bb,   6)
        np.testing.assert_almost_equal(uebcc1_rdm2_f.aaaa, uebcc2_rdm2_f.aaaa, 6)
        np.testing.assert_almost_equal(uebcc1_rdm2_f.aabb, uebcc2_rdm2_f.aabb, 6)
        np.testing.assert_almost_equal(uebcc1_rdm2_f.bbbb, uebcc2_rdm2_f.bbbb, 6)
        np.testing.assert_almost_equal(uebcc1_rdm1_b,      uebcc2_rdm1_b,      6)
        np.testing.assert_almost_equal(uebcc1_dm_b,        uebcc2_dm_b,        6)
        np.testing.assert_almost_equal(uebcc1_dm_eb.aa,    uebcc2_dm_eb.aa,    6)
        np.testing.assert_almost_equal(uebcc1_dm_eb.bb,    uebcc2_dm_eb.bb,    6)


@pytest.mark.reference
class UCCSD_SD_1_2_NoShift_Tests(UCCSD_SD_1_2_Tests):
    """Test UCCSD-SD-1-2 against the legacy GCCSD-SD-SD values with
    shift=False. The system is a singlet.
    """

    shift = False

    @pytest.mark.regression
    def test_lambdas(self):
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"].aa),   -0.052514383878910095, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l1"].bb),   -0.052514383878909804, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].aaaa),  0.029627372142994383, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].abab), -0.052474614602499456, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["l2"].bbbb),  0.029627372142994424, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),      0.199397081277302200, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls1"]),      0.199397081277302200, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),      0.010469370633854065, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["ls2"]),      0.010469370633854065, 5)
        self.assertAlmostEqual(lib.fp(self.ccsd.lambdas["lu11"].aa),  0.032800603529878666, 5)

    @pytest.mark.regression
    def test_dms(self):
        rdm1_f = self.ccsd.make_rdm1_f()
        rdm2_f = self.ccsd.make_rdm2_f()
        rdm1_b = self.ccsd.make_rdm1_b()
        dm_b = self.ccsd.make_sing_b_dm()
        dm_eb = self.ccsd.make_eb_coup_rdm()
        self.assertAlmostEqual(lib.fp(rdm1_f.aa),     1.528453985001563, 6)
        self.assertAlmostEqual(lib.fp(rdm1_f.bb),     1.528453985001563, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.aaaa), -12.796158496372168, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.aabb),  11.888759742067261, 6)
        self.assertAlmostEqual(lib.fp(rdm2_f.bbbb), -12.796158496372172, 6)
        self.assertAlmostEqual(lib.fp(rdm1_b),        0.009432763864565, 6)
        self.assertAlmostEqual(lib.fp(dm_b),          0.343097464069136, 6)
        self.assertAlmostEqual(lib.fp(dm_eb.aa),     -1.064444250615477, 6)
        self.assertAlmostEqual(lib.fp(dm_eb.bb),     -1.064444250615478, 6)


@pytest.mark.reference
class UCCSD_SD_1_2_Dump_Tests(UCCSD_SD_1_2_Tests):
    """Test UCCSD-SD-1-2 after dumping and loading the data.
    """

    shift = True

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
            mo_coeff = data["mo_coeff"]
            data = data[(2, 2, 2)]

        mol = gto.Mole()
        mol.atom = "H 0 0 0; F 0 0 1.1"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        mf.mo_coeff = mo_coeff
        mf = mf.to_uhf()

        nmo = mf.mo_occ[0].size
        nbos = 5
        np.random.seed(12345)
        g = np.random.random((nbos, nmo, nmo)) * 0.02
        g = 0.5 * (g + g.transpose(0, 2, 1).conj())
        omega = np.random.random((nbos,)) * 5.0

        ccsd = UEBCC(
                mf,
                ansatz="CCSD-SD-1-2",
                g=g,
                omega=omega,
                shift=cls.shift,
                log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        osort = list(itertools.chain(*zip(range(ccsd.nocc[0]), range(ccsd.nocc[0], ccsd.nocc[0]+ccsd.nocc[1]))))
        vsort = list(itertools.chain(*zip(range(ccsd.nvir[1]), range(ccsd.nvir[0], ccsd.nvir[0]+ccsd.nvir[1]))))
        fsort = list(itertools.chain(*zip(range(ccsd.nmo), range(ccsd.nmo, 2*ccsd.nmo))))

        cls.mf, cls.ccsd, cls.eris, cls.data = mf, ccsd, eris, data
        cls.osort, cls.vsort, cls.fsort = osort, vsort, fsort
        cls.g, cls.omega = g, omega

        file = "%s/ebcc.h5" % tempfile.gettempdir()
        cls.ccsd.write(file)
        cls.ccsd = cls.ccsd.__class__.read(file, log=cls.ccsd.log)


if __name__ == "__main__":
    print("Tests for UCCSD-SD-1-2")
    unittest.main()
