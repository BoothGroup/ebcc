"""Tests for the UCCSD model.
"""

import itertools
import os
import pickle
import tempfile
import unittest

import numpy as np
import pytest
import scipy.linalg
from pyscf import cc, gto, scf, fci

from ebcc import REBCC, UEBCC, GEBCC, NullLogger, Space, BACKEND, util
from ebcc.ext.fci import (
    fci_to_amplitudes_unrestricted,
    _amplitudes_to_ci_vector_unrestricted,
    _ci_vector_to_amplitudes_unrestricted,
    _tn_addrs_signs,
)


@pytest.mark.reference
class UCCSD_Tests(unittest.TestCase):
    """Test UCCSD against the legacy GCCSD values. The system is a
    singlet.
    """

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_data.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
            mo_coeff = data["mo_coeff"]
            data = data[(2, 0, 0)]

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

        ccsd = UEBCC(
            mf,
            ansatz="CCSD",
            log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        osort = list(
            itertools.chain(
                *zip(range(ccsd.nocc[0]), range(ccsd.nocc[0], ccsd.nocc[0] + ccsd.nocc[1]))
            )
        )
        vsort = list(
            itertools.chain(
                *zip(range(ccsd.nvir[1]), range(ccsd.nvir[0], ccsd.nvir[0] + ccsd.nvir[1]))
            )
        )
        fsort = list(itertools.chain(*zip(range(ccsd.nmo), range(ccsd.nmo, 2 * ccsd.nmo))))

        cls.mf, cls.ccsd, cls.eris, cls.data = mf, ccsd, eris, data
        cls.osort, cls.vsort, cls.fsort = osort, vsort, fsort

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd, cls.eris, cls.data
        del cls.osort, cls.vsort

    def test_energy(self):
        a = self.data[True]["e_corr"]
        b = self.ccsd.e_corr
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.data[True]["t1"]
        b = scipy.linalg.block_diag(self.ccsd.t1.aa, self.ccsd.t1.bb)[self.osort][:, self.vsort]
        self.assertAlmostEqual(np.max(np.abs(a - b)), 0.0, 6)

    def test_rdm1_f(self):
        rdm1_f = self.ccsd.make_rdm1_f()
        a = self.data[True]["rdm1_f"]
        b = scipy.linalg.block_diag(rdm1_f.aa, rdm1_f.bb)
        b = b[self.fsort][:, self.fsort]
        self.assertAlmostEqual(np.max(np.abs(a - b)), 0.0, 6)

    def test_l1_amplitudes(self):
        a = self.data[True]["l1"]
        b = scipy.linalg.block_diag(self.ccsd.l1.aa, self.ccsd.l1.bb)[self.vsort][:, self.osort]
        self.assertAlmostEqual(np.max(np.abs(a - b)), 0.0, 6)

    def test_from_rebcc(self):
        rebcc = REBCC(
            self.mf,
            ansatz="CCSD",
            log=NullLogger(),
        )
        rebcc.options.e_tol = 1e-12
        rebcc.kernel()
        rebcc.solve_lambda()
        uebcc = UEBCC.from_rebcc(rebcc)
        self.assertAlmostEqual(self.ccsd.energy(), uebcc.energy(), 8)
        self.assertAlmostEqual(np.max(np.abs(self.ccsd.t1.aa - uebcc.t1.aa)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(self.ccsd.t1.bb - uebcc.t1.bb)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(self.ccsd.t2.aaaa - uebcc.t2.aaaa)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(self.ccsd.t2.abab - uebcc.t2.abab)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(self.ccsd.t2.bbbb - uebcc.t2.bbbb)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(self.ccsd.l1.aa - uebcc.l1.aa)), 0.0, 5)
        self.assertAlmostEqual(np.max(np.abs(self.ccsd.l1.bb - uebcc.l1.bb)), 0.0, 5)
        self.assertAlmostEqual(np.max(np.abs(self.ccsd.l2.aaaa - uebcc.l2.aaaa)), 0.0, 5)
        self.assertAlmostEqual(np.max(np.abs(self.ccsd.l2.abab - uebcc.l2.abab)), 0.0, 5)
        self.assertAlmostEqual(np.max(np.abs(self.ccsd.l2.bbbb - uebcc.l2.bbbb)), 0.0, 5)

    def test_from_rebcc_frozen(self):
        mf = self.mf.to_rhf()
        occupied = mf.mo_occ > 0
        frozen = np.zeros_like(mf.mo_occ)
        frozen[0] = True
        active = np.zeros_like(mf.mo_occ)
        space = Space(occupied, frozen, active)

        rebcc = REBCC(
            mf,
            ansatz="CCSD",
            space=space,
            log=NullLogger(),
        )
        rebcc.options.e_tol = 1e-12
        rebcc.kernel()
        rebcc.solve_lambda()

        uebcc_1 = UEBCC.from_rebcc(rebcc)

        mf = self.mf.to_uhf()
        occupied = mf.mo_occ[0] > 0
        frozen = np.zeros_like(mf.mo_occ[0])
        frozen[0] = True
        active = np.zeros_like(mf.mo_occ[0])
        space_a = Space(occupied, frozen, active)
        occupied = mf.mo_occ[1] > 0
        frozen = np.zeros_like(mf.mo_occ[1])
        frozen[0] = True
        active = np.zeros_like(mf.mo_occ[1])
        space_b = Space(occupied, frozen, active)

        uebcc_2 = UEBCC(
            mf,
            ansatz="CCSD",
            space=(space_a, space_b),
            log=NullLogger(),
        )
        uebcc_2.options.e_tol = 1e-12
        uebcc_2.kernel()
        uebcc_2.solve_lambda()

        self.assertAlmostEqual(uebcc_2.energy(), uebcc_1.energy(), 8)
        self.assertAlmostEqual(np.max(np.abs(uebcc_2.t1.aa - uebcc_1.t1.aa)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(uebcc_2.t1.bb - uebcc_1.t1.bb)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(uebcc_2.t2.aaaa - uebcc_1.t2.aaaa)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(uebcc_2.t2.abab - uebcc_1.t2.abab)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(uebcc_2.t2.bbbb - uebcc_1.t2.bbbb)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(uebcc_2.l1.aa - uebcc_1.l1.aa)), 0.0, 5)
        self.assertAlmostEqual(np.max(np.abs(uebcc_2.l1.bb - uebcc_1.l1.bb)), 0.0, 5)
        self.assertAlmostEqual(np.max(np.abs(uebcc_2.l2.aaaa - uebcc_1.l2.aaaa)), 0.0, 5)
        self.assertAlmostEqual(np.max(np.abs(uebcc_2.l2.abab - uebcc_1.l2.abab)), 0.0, 5)
        self.assertAlmostEqual(np.max(np.abs(uebcc_2.l2.bbbb - uebcc_1.l2.bbbb)), 0.0, 5)


@pytest.mark.reference
class UCCSD_PySCF_Tests(unittest.TestCase):
    """Test UCCSD against the PySCF values."""

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0 0 0; O 0 0 1"
        mol.basis = "6-31g"
        mol.spin = 2
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        mf = mf.to_uhf()

        ccsd_ref = cc.UCCSD(mf)
        ccsd_ref.conv_tol = 1e-12
        ccsd_ref.conv_tol_normt = 1e-12
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = UEBCC(
            mf,
            ansatz="CCSD",
            log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

    def test_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.ccsd_ref.t1
        b = self.ccsd.t1
        self.assertAlmostEqual(np.max(np.abs(a[0] - b.aa)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(a[1] - b.bb)), 0.0, 6)

    def test_l1_amplitudes(self):
        a = self.ccsd_ref.l1
        b = self.ccsd.l1
        self.assertAlmostEqual(np.max(np.abs(a[0] - b.aa.T)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(a[1] - b.bb.T)), 0.0, 6)

    def test_rdm1(self):
        a = self.ccsd_ref.make_rdm1()
        b = self.ccsd.make_rdm1_f(eris=self.eris)
        self.assertAlmostEqual(np.max(np.abs(a[0] - b.aa)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(a[1] - b.bb)), 0.0, 6)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2()
        b = self.ccsd.make_rdm2_f(eris=self.eris)
        self.assertAlmostEqual(np.max(np.abs(a[0] - b.aaaa)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(a[1] - b.aabb)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(a[2] - b.bbbb)), 0.0, 6)

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ip(self):
        e1 = np.asarray(self.ccsd.ip_eom(nroot=5).kernel())
        e2, v2 = self.ccsd_ref.ipccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 5)

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ea(self):
        e1 = np.asarray(self.ccsd.ea_eom(nroots=5).kernel())
        e2, v2 = self.ccsd_ref.eaccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 5)

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ip_left(self):
        e1 = np.asarray(self.ccsd.ip_eom(nroot=5, left=True).kernel())
        e2, v2 = self.ccsd_ref.ipccsd(nroots=5)  # No left EE-EOM in PySCF
        self.assertAlmostEqual(e1[0], e2[0], 5)

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ea_left(self):
        e1 = np.asarray(self.ccsd.ea_eom(nroots=5, left=True).kernel())
        e2, v2 = self.ccsd_ref.eaccsd(nroots=5)  # No left EE-EOM in PySCF
        self.assertAlmostEqual(e1[0], e2[0], 5)


# Disabled until PySCF fix bug  # TODO
# @pytest.mark.reference
# class FNOUCCSD_PySCF_Tests(UCCSD_PySCF_Tests):
#    """Test FNO-UCCSD against the PySCF values.
#    """
#
#    @classmethod
#    def setUpClass(cls):
#        mol = gto.Mole()
#        mol.atom = "O 0 0 0; O 0 0 1"
#        mol.basis = "6-31g"
#        mol.spin = 2
#        mol.verbose = 0
#        mol.build()
#
#        mf = scf.RHF(mol)
#        mf.conv_tol = 1e-12
#        mf.kernel()
#        mf = mf.to_uhf()
#
#        ccsd_ref = cc.FNOCCSD(mf)
#        ccsd_ref.conv_tol = 1e-12
#        ccsd_ref.conv_tol_normt = 1e-12
#        ccsd_ref.kernel()
#        ccsd_ref.solve_lambda()
#
#        no_coeff, no_occ, no_space = construct_fno_space(mf, occ_tol=1e-3)
#
#        ccsd = UEBCC(
#                mf,
#                mo_coeff=no_coeff,
#                mo_occ=no_occ,
#                space=no_space,
#                ansatz="CCSD",
#                log=NullLogger(),
#        )
#        ccsd.options.e_tol = 1e-10
#        eris = ccsd.get_eris()
#        ccsd.kernel(eris=eris)
#        ccsd.solve_lambda(eris=eris)
#
#        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris
#
#    def test_rdm1(self):
#        a = self.ccsd_ref.make_rdm1(with_frozen=False)
#        b = self.ccsd.make_rdm1_f(eris=self.eris)
#        self.assertAlmostEqual(np.max(np.abs(a[0] - b.aa)), 0.0, 6)
#        self.assertAlmostEqual(np.max(np.abs(a[1] - b.bb)), 0.0, 6)
#
#    def test_rdm2(self):
#        a = self.ccsd_ref.make_rdm2(with_frozen=False)
#        b = self.ccsd.make_rdm2_f(eris=self.eris)
#        self.assertAlmostEqual(np.max(np.abs(a[0] - b.aaaa)), 0.0, 6)
#        self.assertAlmostEqual(np.max(np.abs(a[1] - b.aabb)), 0.0, 6)
#        self.assertAlmostEqual(np.max(np.abs(a[2] - b.bbbb)), 0.0, 6)


@pytest.mark.reference
class UCCSD_PySCF_Frozen_Tests(unittest.TestCase):
    """Test UCCSD against the PySCF values with frozen orbitals."""

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0 0 0; O 0 0 1"
        mol.basis = "6-31g"
        mol.spin = 2
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        mf = mf.to_uhf()

        frozen_a = np.zeros_like(mf.mo_occ[0])
        frozen_a[:2] = 1
        frozen_a[-1] = 1
        frozen_a = frozen_a.astype(bool)

        frozen_b = np.zeros_like(mf.mo_occ[1])
        frozen_b[:1] = 1
        frozen_b[-1] = 1
        frozen_b = frozen_b.astype(bool)

        ccsd_ref = cc.UCCSD(mf)
        ccsd_ref.conv_tol = 1e-12
        ccsd_ref.conv_tol_normt = 1e-12
        ccsd_ref.frozen = (np.where(frozen_a)[0], np.where(frozen_b)[0])
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        space_a = Space(
            mf.mo_occ[0] > 0,
            frozen_a,
            np.zeros_like(mf.mo_occ[0]),
        )

        space_b = Space(
            mf.mo_occ[1] > 0,
            frozen_b,
            np.zeros_like(mf.mo_occ[1]),
        )

        ccsd = UEBCC(
            mf,
            ansatz="CCSD",
            space=(space_a, space_b),
            log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris

    def test_energy(self):
        a = self.ccsd_ref.e_tot
        b = self.ccsd.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_t1_amplitudes(self):
        a = self.ccsd_ref.t1
        b = self.ccsd.t1
        self.assertAlmostEqual(np.max(np.abs(a[0] - b.aa)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(a[1] - b.bb)), 0.0, 6)

    def test_l1_amplitudes(self):
        a = self.ccsd_ref.l1
        b = self.ccsd.l1
        self.assertAlmostEqual(np.max(np.abs(a[0] - b.aa.T)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(a[1] - b.bb.T)), 0.0, 6)

    def test_rdm1(self):
        a = self.ccsd_ref.make_rdm1(with_frozen=False)
        b = self.ccsd.make_rdm1_f(eris=self.eris)
        self.assertAlmostEqual(np.max(np.abs(a[0] - b.aa)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(a[1] - b.bb)), 0.0, 6)

    def test_rdm2(self):
        a = self.ccsd_ref.make_rdm2(with_frozen=False)
        b = self.ccsd.make_rdm2_f(eris=self.eris)
        self.assertAlmostEqual(np.max(np.abs(a[0] - b.aaaa)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(a[1] - b.aabb)), 0.0, 6)
        self.assertAlmostEqual(np.max(np.abs(a[2] - b.bbbb)), 0.0, 6)

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ip(self):
        e1 = np.asarray(self.ccsd.ip_eom(nroot=5).kernel())
        e2, v2 = self.ccsd_ref.ipccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 5)

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ea(self):
        e1 = np.asarray(self.ccsd.ea_eom(nroots=5).kernel())
        e2, v2 = self.ccsd_ref.eaccsd(nroots=5)
        self.assertAlmostEqual(e1[0], e2[0], 5)

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ip_left(self):
        e1 = np.asarray(self.ccsd.ip_eom(nroot=5, left=True).kernel())
        e2, v2 = self.ccsd_ref.ipccsd(nroots=5)  # No left EE-EOM in PySCF
        self.assertAlmostEqual(e1[0], e2[0], 5)

    @pytest.mark.skipif(BACKEND != "numpy", reason="EOM is currently too slow with non-NumPy backends")
    def test_eom_ea_left(self):
        e1 = np.asarray(self.ccsd.ea_eom(nroots=5, left=True).kernel())
        e2, v2 = self.ccsd_ref.eaccsd(nroots=5)  # No left EE-EOM in PySCF
        self.assertAlmostEqual(e1[0], e2[0], 5)


@pytest.mark.reference
class UCCSD_Dump_Tests(UCCSD_PySCF_Tests):
    """Test UCCSD against the PySCF after dumping and loading."""

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "O 0 0 0; O 0 0 1"
        mol.basis = "6-31g"
        mol.spin = 2
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        mf = mf.to_uhf()

        ccsd_ref = cc.UCCSD(mf)
        ccsd_ref.conv_tol = 1e-12
        ccsd_ref.conv_tol_normt = 1e-12
        ccsd_ref.kernel()
        ccsd_ref.solve_lambda()

        ccsd = UEBCC(
            mf,
            ansatz="CCSD",
            log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-12
        eris = ccsd.get_eris()
        ccsd.kernel(eris=eris)
        ccsd.solve_lambda(eris=eris)

        cls.mf, cls.ccsd_ref, cls.ccsd, cls.eris = mf, ccsd_ref, ccsd, eris

        file = "%s/ebcc.h5" % tempfile.gettempdir()
        cls.ccsd.write(file)
        cls.ccsd = cls.ccsd.__class__.read(file, log=cls.ccsd.log)


class UCCSD_ExtCorr_Tests(unittest.TestCase):
    """Test UCCSD with external correction.
    """

    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.atom = "Ne 0 0 0"
        mol.basis = "6-31g"
        mol.verbose = 0
        mol.build()

        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        space = (
            Space(
                mf.mo_occ[0] > 0,
                np.zeros_like(mf.mo_occ[0]),
                np.ones_like(mf.mo_occ[0]),
            ),
            Space(
                mf.mo_occ[1] > 0,
                np.zeros_like(mf.mo_occ[1]),
                np.ones_like(mf.mo_occ[1]),
            ),
        )

        ci = fci.FCI(mf, mo=(mf.mo_coeff[0][:, space[0].active], mf.mo_coeff[1][:, space[1].active]))
        ci.conv_tol = 1e-12
        ci.kernel()

        cls.mf, cls.ci, cls.space = mf, ci, space

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.ci, cls.space

    def test_conversion(self):
        amps1 = _ci_vector_to_amplitudes_unrestricted(self.ci.ci, self.space, max_order=4)
        ci = _amplitudes_to_ci_vector_unrestricted(amps1, normalise=False, max_order=4)
        amps2 = _ci_vector_to_amplitudes_unrestricted(ci, self.space, max_order=4)

        for order in range(1, 5):
            for spins in util.generate_spin_combinations(order, unique=True):
                i, _ = _tn_addrs_signs(self.space[0].nact, self.space[0].naocc, spins.count("a") // 2)
                j, _ = _tn_addrs_signs(self.space[1].nact, self.space[1].naocc, spins.count("b") // 2)
                if spins.count("a") and spins.count("b"):
                    i, j = np.ix_(i, j)
                assert np.allclose(ci[i, j] * self.ci.ci[0, 0], self.ci.ci[i, j]), (order, spins)

        with pytest.raises(AssertionError):
            # Expect a fail since the excitation space goes beyond fourth order -- we have
            # checked the individual orders above
            assert np.allclose(ci * self.ci.ci[0, 0], self.ci.ci)

        assert np.allclose(amps1.t1.aa, amps2.t1.aa)
        assert np.allclose(amps1.t1.bb, amps2.t1.bb)
        assert np.allclose(amps1.t2.aaaa, amps2.t2.aaaa)
        assert np.allclose(amps1.t2.abab, amps2.t2.abab)
        assert np.allclose(amps1.t2.bbbb, amps2.t2.bbbb)
        assert np.allclose(amps1.t3.aaaaaa, amps2.t3.aaaaaa)
        assert np.allclose(amps1.t3.abaaba, amps2.t3.abaaba)
        assert np.allclose(amps1.t3.babbab, amps2.t3.babbab)
        assert np.allclose(amps1.t3.bbbbbb, amps2.t3.bbbbbb)
        assert np.allclose(amps1.t4.aaaaaaaa, amps2.t4.aaaaaaaa)
        assert np.allclose(amps1.t4.aaabaaab, amps2.t4.aaabaaab)
        assert np.allclose(amps1.t4.abababab, amps2.t4.abababab)
        assert np.allclose(amps1.t4.abbbabbb, amps2.t4.abbbabbb)
        assert np.allclose(amps1.t4.bbbbbbbb, amps2.t4.bbbbbbbb)

    def test_external_correction(self):
        amplitudes = fci_to_amplitudes_unrestricted(self.ci, self.space)
        ccsd = UEBCC(
            self.mf,
            ansatz="CCSD",
            space=self.space,
            log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        ccsd.external_correction(amplitudes, mixed_term_strategy="update")

        self.assertTrue(ccsd.converged)

        a = self.ci.e_tot
        b = ccsd.e_tot
        self.assertAlmostEqual(a, b, 7)

    def test_tailor(self):
        amplitudes = fci_to_amplitudes_unrestricted(self.ci, self.space, max_order=2)
        ccsd = UEBCC(
            self.mf,
            ansatz="CCSD",
            space=self.space,
            log=NullLogger(),
        )
        ccsd.options.e_tol = 1e-10
        ccsd.options.t_tol = 1e-8
        ccsd.tailor(amplitudes)

        self.assertTrue(ccsd.converged)

        a = self.ci.e_tot
        b = ccsd.e_tot
        self.assertAlmostEqual(a, b, 7)


if __name__ == "__main__":
    print("Tests for UCCSD")
    unittest.main()
