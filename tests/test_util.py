"""Tests for util module.
"""

import unittest
import itertools

import numpy as np
import pytest

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
            for ndim in (1, 2, 3, 4):
                array = np.arange(n**ndim).reshape((n,) * ndim)
                array = util.antisymmetrise_array(array, axes=range(ndim))
                for perm, sign in util.permutations_with_signs(range(ndim)):
                    np.testing.assert_equal(array, sign * array.transpose(perm))

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
        pass  # Covered by methods



if __name__ == "__main__":
    print("Tests for utilities")
    unittest.main()
