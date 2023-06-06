"""Tests for util module.
"""

import unittest
import itertools

from pyscf import gto, scf
import numpy as np
import pytest

import ebcc
from ebcc import util


@pytest.mark.regression
class Util_Tests(unittest.TestCase):
    """Test util module against known values.
    """

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_factorial(self):
        self.assertEqual(util.factorial(0), 1)
        self.assertEqual(util.factorial(1), 1)
        self.assertEqual(util.factorial(2), 2)
        self.assertEqual(util.factorial(3), 6)
        self.assertEqual(util.factorial(4), 24)
        self.assertEqual(util.factorial(5), 120)

    def test_permute_string(self):
        self.assertEqual(util.permute_string("god", (2, 1, 0)), "dog")
        self.assertEqual(util.permute_string("ebcc", (2, 0, 3, 1)), "cecb")

    def test_tril_indices_ndim(self):
        for n in (1, 2, 3, 4):
            for ndim in (1, 2, 3, 4):
                for combinations, include_diagonal in [
                        (itertools.combinations, False),
                        (itertools.combinations_with_replacement, True),
                ]:
                    x = np.zeros((n,) * ndim)
                    y = np.zeros((n,) * ndim)
                    x[util.tril_indices_ndim(n, ndim, include_diagonal=include_diagonal)] = 1
                    for tup in combinations(range(n), r=ndim):
                        y[tuple(sorted(tup)[::-1])] = 1
                    np.testing.assert_equal(x, y)

    def test_ntril_ndim(self):
        for n in (1, 2, 3, 4):
            for ndim in (1, 2, 3, 4):
                for combinations, include_diagonal in [
                        (itertools.combinations, False),
                        (itertools.combinations_with_replacement, True),
                ]:
                    a = sum(1 for tup in combinations(range(n), r=ndim))
                    b = util.ntril_ndim(n, ndim, include_diagonal=include_diagonal)
                    self.assertEqual(a, b)

    def test_generate_spin_combinations(self):
        for n, combs in [
                (1, {"aa", "bb"}),
                (2, {"aaaa", "abab", "baba", "bbbb"}),
                (3, {"aaaaaa", "aabaab", "abaaba", "baabaa", "abbabb", "babbab", "bbabba", "bbbbbb"}),
        ]:
            self.assertEqual(set(util.generate_spin_combinations(n)), combs)
        for n, combs in [
                (1, {"aa", "bb"}),
                (2, {"aaaa", "abab", "bbbb"}),
                (3, {"aaaaaa", "abaaba", "babbab", "bbbbbb"}),
        ]:
            self.assertEqual(set(util.generate_spin_combinations(n, unique=True)), combs)
        for n, combs in [
                (1, {"a", "b"}),
                (2, {"aaa", "aba", "bab", "bbb"}),
                (3, {"aaaaa", "aabaa", "abaab", "baaba", "abbab", "babba", "bbabb", "bbbbb"}),
        ]:
            self.assertEqual(set(util.generate_spin_combinations(n, excited=True)), combs)

    def test_permutations_with_signs(self):
        for seq, res in [
                ([0, 1], (([0, 1], 1), ([1, 0], -1))),
                ([0, 1, 2], tuple(sorted([
                    ([0, 1, 2], 1), ([0, 2, 1], -1), ([1, 0, 2], -1),
                    ([1, 2, 0], 1), ([2, 0, 1], 1), ([2, 1, 0], -1),
                ]))),
        ]:
            self.assertEqual(tuple(sorted(util.permutations_with_signs(seq))), res)

    def test_symmetry_factor(self):
        self.assertAlmostEqual(util.get_symmetry_factor(1, 1), 1.0, 10)
        self.assertAlmostEqual(util.get_symmetry_factor(2, 2), 0.25, 10)
        self.assertAlmostEqual(util.get_symmetry_factor(3, 2, 1), 0.125, 10)

    def test_antisymmetrise_array(self):
        for n in (1, 2, 3, 4):
            for ndim in (1, 2, 3, 4, 5, 6):
                array = np.cos(np.arange(1, n**ndim+1).reshape((n,) * ndim))
                array = util.antisymmetrise_array(array, axes=range(ndim))
                for perm, sign in util.permutations_with_signs(range(ndim)):
                    np.testing.assert_almost_equal(array, sign * array.transpose(perm))

    def test_is_mixed(self):
        self.assertEqual(util.is_mixed_spin("aa"), False)
        self.assertEqual(util.is_mixed_spin("ab"), True)
        self.assertEqual(util.is_mixed_spin("bbb"), False)
        self.assertEqual(util.is_mixed_spin("aba"), True)
        self.assertEqual(util.is_mixed_spin([0, 0, 0, 0]), False)
        self.assertEqual(util.is_mixed_spin([1, 0, 1, 0]), True)

    def test_compress_axes(self):
        pass  # Covered by methods

    def test_decompress_axes(self):
        pass  # Covered by methods

    def test_get_compressed_size(self):
        pass  # Covered by methods

    def test_symmetrise(self):
        for n in (1, 2, 3, 4):
            for ndim in (2, 4, 6):
                subscript = "i" * (ndim//2) + "a" * (ndim//2)
                array = np.cos(np.arange(n**ndim).reshape((n,) * ndim))
                array = util.symmetrise(subscript, array)
                for p1, s1 in util.permutations_with_signs(range(ndim//2)):
                    for p2, s2 in util.permutations_with_signs(range(ndim//2, ndim)):
                        perm = tuple(p1) + tuple(p2)
                        sign = s1 * s2
                        np.testing.assert_almost_equal(array, sign * array.transpose(perm))

    def test_constructors(self):
        # Tests the constructors in the main __init__.py
        mol = gto.M(atom="H 0 0 0; Li 0 0 1.64", basis="6-31g", verbose=0)
        rhf = scf.RHF(mol).run()
        uhf = rhf.to_uhf()
        ghf = uhf.to_ghf()
        log = ebcc.NullLogger()
        e_ccsd_r = ebcc.REBCC(rhf, ansatz="CCSD", e_tol=1e-10, log=log).kernel()
        e_ccsd_u = ebcc.UEBCC(uhf, ansatz="CCSD", e_tol=1e-10, log=log).kernel()
        e_ccsd_g = ebcc.GEBCC(ghf, ansatz="CCSD", e_tol=1e-10, log=log).kernel()
        self.assertAlmostEqual(ebcc.EBCC(rhf, e_tol=1e-10, log=log).kernel(), e_ccsd_r, 8)
        self.assertAlmostEqual(ebcc.EBCC(uhf, e_tol=1e-10, log=log).kernel(), e_ccsd_u, 8)
        self.assertAlmostEqual(ebcc.EBCC(ghf, e_tol=1e-10, log=log).kernel(), e_ccsd_g, 8)
        self.assertAlmostEqual(ebcc.CCSD(rhf, e_tol=1e-10, log=log).kernel(), e_ccsd_r, 8)
        self.assertAlmostEqual(ebcc.CCSD(uhf, e_tol=1e-10, log=log).kernel(), e_ccsd_u, 8)
        self.assertAlmostEqual(ebcc.CCSD(ghf, e_tol=1e-10, log=log).kernel(), e_ccsd_g, 8)
        e_cc2_r = ebcc.REBCC(rhf, ansatz="CC2", e_tol=1e-10, log=log).kernel()
        e_cc2_u = ebcc.UEBCC(uhf, ansatz="CC2", e_tol=1e-10, log=log).kernel()
        e_cc2_g = ebcc.GEBCC(ghf, ansatz="CC2", e_tol=1e-10, log=log).kernel()
        self.assertAlmostEqual(ebcc.CC2(rhf, e_tol=1e-10, log=log).kernel(), e_cc2_r, 8)
        self.assertAlmostEqual(ebcc.CC2(uhf, e_tol=1e-10, log=log).kernel(), e_cc2_u, 8)
        self.assertAlmostEqual(ebcc.CC2(ghf, e_tol=1e-10, log=log).kernel(), e_cc2_g, 8)

    def test_einsum(self):
        # Tests the einsum implementation
        _size = util.NUMPY_EINSUM_SIZE
        util.NUMPY_EINSUM_SIZE = 0

        x = np.random.random((10, 11, 12, 13))
        y = np.random.random((13, 12, 5, 6))
        z = np.random.random((6, 5, 7, 8))

        a = np.einsum("ijkl,lkab->ijab", x, y)
        b = util.einsum("ijkl,lkab->ijab", x, y)
        np.testing.assert_almost_equal(a, b)

        b[:] = 0
        b = util.einsum("ijkl,lkab->ijab", x, y, out=b, alpha=1.0, beta=0.0)
        np.testing.assert_almost_equal(a, b)

        b = util.einsum("ijkl,lkab->ijab", x, y, out=b, alpha=1.0, beta=1.0)
        np.testing.assert_almost_equal(a*2, b)

        b = util.einsum("ijkl,lkab->ijab", x, y, out=b, alpha=2.0, beta=0.0)
        np.testing.assert_almost_equal(a*2, b)

        a = np.einsum("ijkl,lkab,bacd->ijcd", x, y, z)
        b = util.einsum("ijkl,lkab,bacd->ijcd", x, y, z)
        np.testing.assert_almost_equal(a, b)

        x = np.asfortranarray(np.random.random((5, 5, 5)))
        y = np.asfortranarray(np.random.random((5, 5, 5)))
        z = np.asfortranarray(np.random.random((5, 5, 5)))

        a = np.einsum("ijk,jki,kji->ik", x, y, z)
        b = util.einsum("ijk,jki,kji->ik", x, y, z)
        np.testing.assert_almost_equal(a, b)

        a = np.einsum("ijk,jki,kji->ik", x, y, z)
        b = util.einsum("ijk,jki,kji->ik", x, y, z, alpha=0.5, beta=0.5)
        np.testing.assert_almost_equal(a, b)

        a = np.einsum("iik,kjj->ij", x, y)
        b = util.einsum("iik,kjj->ij", x, y)
        np.testing.assert_almost_equal(a, b)

        a = np.einsum("ijk,ijk->", x, y)
        b = util.einsum("ijk,ijk->", x, y)
        np.testing.assert_almost_equal(a, b)

        z = np.random.random((5, 5))

        a = np.einsum("ijk,jk->", x, z)
        b = util.einsum("ijk,jk->", x, z)
        np.testing.assert_almost_equal(a, b)

        a = np.einsum("ijk,kl->il", x, z)
        b = util.einsum("ijk,kl->il", x, z)
        np.testing.assert_almost_equal(a, b)

        x = np.random.random((5, 5, 0))

        a = np.einsum("ikl,jkl->ij", x, x)
        b = util.einsum("ikl,jkl->ij", x, x)
        np.testing.assert_almost_equal(a, b)

        util.NUMPY_EINSUM_SIZE = _size


if __name__ == "__main__":
    print("Tests for utilities")
    unittest.main()
