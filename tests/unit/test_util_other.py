"""Tests for fci

"""
import unittest

import numpy as np

import tests
from util.other import int_array
from util.variables import int_dtype

@tests.category('SHORT', 'ESSENTIAL')
class SlaterDetTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        
    def test_1(self):
        self.assertEqual(int_array(0, 1, 2, 3, 4, 5),
                         np.array([0, 1, 2, 3, 4, 5], dtype=int_dtype))
        self.assertEqual(int_array(4, 6, 12, 34, 6, 1),
                         np.array([4, 6, 12, 34, 6, 1], dtype=int_dtype))

    def test_2(self):
        self.assertEqual(int_array([0, 1, 2, 3, 4, 5]),
                         np.array([0, 1, 2, 3, 4, 5], dtype=int_dtype))
        self.assertEqual(int_array([4, 6, 12, 34, 6, 1]),
                         np.array([4, 6, 12, 34, 6, 1], dtype=int_dtype))

    def test_3(self):
        self.assertEqual(int_array((0, 1, 2, 3, 4, 5)),
                         np.array([0, 1, 2, 3, 4, 5], dtype=int_dtype))
        self.assertEqual(int_array((4, 6, 12, 34, 6, 1)),
                         np.array([4, 6, 12, 34, 6, 1], dtype=int_dtype))

    def test_4(self):
        self.assertEqual(int_array([[0, 1, 2, 3, 4, 5],
                                    [6, 7, 8, 9, 10, 11]]),
                         np.array([[0, 1, 2, 3, 4, 5],
                                   [6, 7, 8, 9, 10, 11]], dtype=int_dtype))
        self.assertEqual(int_array(([[4, 6, 12, 34, 6, 1],
                                     [6, 7, 1, 0, 15, 40]])),
                         np.array([[4, 6, 12, 34, 6, 1],
                                   [6, 7, 1, 0, 15, 40]], dtype=int_dtype))
