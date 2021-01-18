"""Tests for CC manifold

"""
import unittest

import numpy as np

from coupled_cluster.manifold cimport _exc_on_string
from wave_functions.fci import make_occ

import test


class ExcOnStringTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)

    def test1(self):
        self.assertEqual(make_occ([1, 2, 3, 4, 5, 6, -1]),
                         make_occ(_exc_on_string(
                             0, 6, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 2, 3, 4, 5, 6, 1]),
                         make_occ(_exc_on_string(
                             1, 6, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 3, 4, 5, 6, -1]),
                         make_occ(_exc_on_string(
                             2, 6, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 4, 5, 6, 1]),
                         make_occ(_exc_on_string(
                             3, 6, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 3, 5, 6, -1]),
                         make_occ(_exc_on_string(
                             4, 6, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 3, 4, 6, 1]),
                         make_occ(_exc_on_string(
                             5, 6, make_occ([0, 1, 2, 3, 4, 5]))))

    def test2(self):
        self.assertEqual(make_occ([1, 2, 3, 4, 5, 8, -1]),
                         make_occ(_exc_on_string(
                             0, 8, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 2, 3, 4, 5, 8, 1]),
                         make_occ(_exc_on_string(
                             1, 8, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 3, 4, 5, 8, -1]),
                         make_occ(_exc_on_string(
                             2, 8, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 4, 5, 8, 1]),
                         make_occ(_exc_on_string(
                             3, 8, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 3, 5, 8, -1]),
                         make_occ(_exc_on_string(
                             4, 8, make_occ([0, 1, 2, 3, 4, 5]))))
        self.assertEqual(make_occ([0, 1, 2, 3, 4, 8, 1]),
                         make_occ(_exc_on_string(
                             5, 8, make_occ([0, 1, 2, 3, 4, 5]))))

    def test3(self):
        self.assertEqual(make_occ([1, 2, 4, 6, 7, 8, 1]),
                         make_occ(_exc_on_string(
                             0, 4, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 2, 4, 6, 7, 8, -1]),
                         make_occ(_exc_on_string(
                             1, 4, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 4, 6, 7, 8, 1]),
                         make_occ(_exc_on_string(
                             2, 4, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 4, 7, 8, 1]),
                         make_occ(_exc_on_string(
                             6, 4, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 4, 6, 8, -1]),
                         make_occ(_exc_on_string(
                             7, 4, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 4, 6, 7, 1]),
                         make_occ(_exc_on_string(
                             8, 4, make_occ([0, 1, 2, 6, 7, 8]))))

    def test4(self):
        self.assertEqual(make_occ([1, 2, 6, 7, 8, 9, -1]),
                         make_occ(_exc_on_string(
                             0, 9, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 2, 6, 7, 8, 9, 1]),
                         make_occ(_exc_on_string(
                             1, 9, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 6, 7, 8, 9, -1]),
                         make_occ(_exc_on_string(
                             2, 9, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 7, 8, 9, 1]),
                         make_occ(_exc_on_string(
                             6, 9, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 6, 8, 9, -1]),
                         make_occ(_exc_on_string(
                             7, 9, make_occ([0, 1, 2, 6, 7, 8]))))
        self.assertEqual(make_occ([0, 1, 2, 6, 7, 9, 1]),
                         make_occ(_exc_on_string(
                             8, 9, make_occ([0, 1, 2, 6, 7, 8]))))

    def test5(self):
        self.assertEqual(make_occ([0, 4, 5, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             3, 0, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([0, 3, 5, 10, 11, 12, -1]),
                         make_occ(_exc_on_string(
                             4, 0, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([0, 3, 4, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             5, 0, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([0, 3, 4, 5, 11, 12, -1]),
                         make_occ(_exc_on_string(
                             10, 0, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([0, 3, 4, 5, 10, 12, 1]),
                         make_occ(_exc_on_string(
                             11, 0, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([0, 3, 4, 5, 10, 11, -1]),
                         make_occ(_exc_on_string(
                             12, 0, make_occ([3, 4, 5, 10, 11, 12]))))

    def test6(self):
        self.assertEqual(make_occ([2, 4, 5, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             3, 2, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([2, 3, 5, 10, 11, 12, -1]),
                         make_occ(_exc_on_string(
                             4, 2, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([2, 3, 4, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             5, 2, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([2, 3, 4, 5, 11, 12, -1]),
                         make_occ(_exc_on_string(
                             10, 2, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([2, 3, 4, 5, 10, 12, 1]),
                         make_occ(_exc_on_string(
                             11, 2, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([2, 3, 4, 5, 10, 11, -1]),
                         make_occ(_exc_on_string(
                             12, 2, make_occ([3, 4, 5, 10, 11, 12]))))

    def test7(self):
        self.assertEqual(make_occ([4, 5, 6, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             3, 6, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 5, 6, 10, 11, 12, -1]),
                         make_occ(_exc_on_string(
                             4, 6, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 6, 10, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             5, 6, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 6, 11, 12, 1]),
                         make_occ(_exc_on_string(
                             10, 6, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 6, 10, 12, -1]),
                         make_occ(_exc_on_string(
                             11, 6, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 6, 10, 11, 1]),
                         make_occ(_exc_on_string(
                             12, 6, make_occ([3, 4, 5, 10, 11, 12]))))

    def test8(self):
        self.assertEqual(make_occ([4, 5, 10, 11, 12, 15, -1]),
                         make_occ(_exc_on_string(
                             3, 15, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 5, 10, 11, 12, 15, 1]),
                         make_occ(_exc_on_string(
                             4, 15, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 10, 11, 12, 15, -1]),
                         make_occ(_exc_on_string(
                             5, 15, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 11, 12, 15, 1]),
                         make_occ(_exc_on_string(
                             10, 15, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 10, 12, 15, -1]),
                         make_occ(_exc_on_string(
                             11, 15, make_occ([3, 4, 5, 10, 11, 12]))))
        self.assertEqual(make_occ([3, 4, 5, 10, 11, 15, 1]),
                         make_occ(_exc_on_string(
                             12, 15, make_occ([3, 4, 5, 10, 11, 12]))))
