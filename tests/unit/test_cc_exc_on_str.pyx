"""Tests for CC manifold

"""
import unittest

import numpy as np

import tests
from util.variables import int_dtype
from util.other import int_array
from coupled_cluster.exc_on_string cimport exc_on_string, annihilates

@tests.category('SHORT', 'ESSENTIAL')
class ExcOnStringTestCase(unittest.TestCase):

    def setUp(self):
        self.I_buf = np.empty(6, dtype=int_dtype)
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test1(self):
        s = exc_on_string(0, 6, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(1, 2, 3, 4, 5, 6), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(1, 6, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(0, 2, 3, 4, 5, 6), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(2, 6, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(0, 1, 3, 4, 5, 6), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(3, 6, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 4, 5, 6), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(4, 6, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 3, 5, 6), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(5, 6, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 3, 4, 6), self.I_buf)
        self.assertEqual(s, 1)

    def test2(self):
        s = exc_on_string(0, 8, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(1, 2, 3, 4, 5, 8), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(1, 8, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(0, 2, 3, 4, 5, 8), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(2, 8, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(0, 1, 3, 4, 5, 8), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(3, 8, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 4, 5, 8), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(4, 8, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 3, 5, 8), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(5, 8, int_array(0, 1, 2, 3, 4, 5), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 3, 4, 8), self.I_buf)
        self.assertEqual(s, 1)

    def test3(self):
        s = exc_on_string(0, 4, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(1, 2, 4, 6, 7, 8), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(1, 4, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(0, 2, 4, 6, 7, 8), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(2, 4, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(0, 1, 4, 6, 7, 8), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(6, 4, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 4, 7, 8), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(7, 4, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 4, 6, 8), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(8, 4, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 4, 6, 7), self.I_buf)
        self.assertEqual(s, 1)

    def test4(self):
        s = exc_on_string(0, 9, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(1, 2, 6, 7, 8, 9), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(1, 9, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(0, 2, 6, 7, 8, 9), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(2, 9, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(0, 1, 6, 7, 8, 9), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(6, 9, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 7, 8, 9), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(7, 9, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 6, 8, 9), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(8, 9, int_array(0, 1, 2, 6, 7, 8), self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 6, 7, 9), self.I_buf)
        self.assertEqual(s, 1)

    def test5(self):
        s = exc_on_string(3, 0, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(0, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(4, 0, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(0, 3, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(5, 0, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(0, 3, 4, 10, 11, 12), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(10, 0, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(0, 3, 4, 5, 11, 12), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(11, 0, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(0, 3, 4, 5, 10, 12), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(12, 0, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(0, 3, 4, 5, 10, 11), self.I_buf)
        self.assertEqual(s, -1)

    def test6(self):
        s = exc_on_string(3, 2, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(2, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(4, 2, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(2, 3, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(5, 2, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(2, 3, 4, 10, 11, 12), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(10, 2, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(2, 3, 4, 5, 11, 12), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(11, 2, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(2, 3, 4, 5, 10, 12), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(12, 2, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(2, 3, 4, 5, 10, 11), self.I_buf)
        self.assertEqual(s, -1)

    def test7(self):
        s = exc_on_string(3, 6, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(4, 5, 6, 10, 11, 12), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(4, 6, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(3, 5, 6, 10, 11, 12), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(5, 6, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(3, 4, 6, 10, 11, 12), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(10, 6, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(3, 4, 5, 6, 11, 12), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(11, 6, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(3, 4, 5, 6, 10, 12), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(12, 6, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(3, 4, 5, 6, 10, 11), self.I_buf)
        self.assertEqual(s, 1)

    def test8(self):
        s = exc_on_string(3, 15, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(4, 5, 10, 11, 12, 15), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(4, 15, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(3, 5, 10, 11, 12, 15), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(5, 15, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(3, 4, 10, 11, 12, 15), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(10, 15, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(3, 4, 5, 11, 12, 15), self.I_buf)
        self.assertEqual(s, 1)
        s = exc_on_string(11, 15, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(3, 4, 5, 10, 12, 15), self.I_buf)
        self.assertEqual(s, -1)
        s = exc_on_string(12, 15, int_array(3, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(int_array(3, 4, 5, 10, 11, 15), self.I_buf)
        self.assertEqual(s, 1)

    def test9(self):
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(3, 15, self.I_buf, self.I_buf)
        self.assertEqual(int_array(4, 5, 10, 11, 12, 15), self.I_buf)
        self.assertEqual(s, -1)
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(4, 15, self.I_buf, self.I_buf)
        self.assertEqual(int_array(3, 5, 10, 11, 12, 15), self.I_buf)
        self.assertEqual(s, 1)
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(5, 15, self.I_buf, self.I_buf)
        self.assertEqual(int_array(3, 4, 10, 11, 12, 15), self.I_buf)
        self.assertEqual(s, -1)
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(10, 15, self.I_buf, self.I_buf)
        self.assertEqual(int_array(3, 4, 5, 11, 12, 15), self.I_buf)
        self.assertEqual(s, 1)
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(11, 15, self.I_buf, self.I_buf)
        self.assertEqual(int_array(3, 4, 5, 10, 12, 15), self.I_buf)
        self.assertEqual(s, -1)
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(12, 15, self.I_buf, self.I_buf)
        self.assertEqual(int_array(3, 4, 5, 10, 11, 15), self.I_buf)
        self.assertEqual(s, 1)

    def test10(self):
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(3, 2, self.I_buf, self.I_buf)
        self.assertEqual(int_array(2, 4, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(s, 1)
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(4, 2, self.I_buf, self.I_buf)
        self.assertEqual(int_array(2, 3, 5, 10, 11, 12), self.I_buf)
        self.assertEqual(s, -1)
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(5, 2, self.I_buf, self.I_buf)
        self.assertEqual(int_array(2, 3, 4, 10, 11, 12), self.I_buf)
        self.assertEqual(s, 1)
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(10, 2, self.I_buf, self.I_buf)
        self.assertEqual(int_array(2, 3, 4, 5, 11, 12), self.I_buf)
        self.assertEqual(s, -1)
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(11, 2, self.I_buf, self.I_buf)
        self.assertEqual(int_array(2, 3, 4, 5, 10, 12), self.I_buf)
        self.assertEqual(s, 1)
        self.I_buf[:] = int_array(3, 4, 5, 10, 11, 12)
        s = exc_on_string(12, 2, self.I_buf, self.I_buf)
        self.assertEqual(int_array(2, 3, 4, 5, 10, 11), self.I_buf)
        self.assertEqual(s, -1)

    def test11(self):
        self.I_buf[:] = int_array(0, 1, 2, 3, 4, 5)
        s = exc_on_string(0, 6, self.I_buf, self.I_buf)
        self.assertEqual(int_array(1, 2, 3, 4, 5, 6), self.I_buf)
        self.assertEqual(s, -1)
        self.I_buf[:] = int_array(0, 1, 2, 3, 4, 5)
        s = exc_on_string(1, 6, self.I_buf, self.I_buf)
        self.assertEqual(int_array(0, 2, 3, 4, 5, 6), self.I_buf)
        self.assertEqual(s, 1)
        self.I_buf[:] = int_array(0, 1, 2, 3, 4, 5)
        s = exc_on_string(2, 6, self.I_buf, self.I_buf)
        self.assertEqual(int_array(0, 1, 3, 4, 5, 6), self.I_buf)
        self.assertEqual(s, -1)
        self.I_buf[:] = int_array(0, 1, 2, 3, 4, 5)
        s = exc_on_string(3, 6, self.I_buf, self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 4, 5, 6), self.I_buf)
        self.assertEqual(s, 1)
        self.I_buf[:] = int_array(0, 1, 2, 3, 4, 5)
        s = exc_on_string(4, 6, self.I_buf, self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 3, 5, 6), self.I_buf)
        self.assertEqual(s, -1)
        self.I_buf[:] = int_array(0, 1, 2, 3, 4, 5)
        s = exc_on_string(5, 6, self.I_buf, self.I_buf)
        self.assertEqual(int_array(0, 1, 2, 3, 4, 6), self.I_buf)
        self.assertEqual(s, 1)


class AnnihilatesTestCase(unittest.TestCase):

    def setUp(self):
        self.I_buf = np.empty(6, dtype=int_dtype)

    def test1(self):
        self.I_buf[:] = int_array(0, 1, 2, 3, 4, 5)
        self.assertTrue(annihilates(6, 6, self.I_buf))
        s = exc_on_string(6, 6, self.I_buf, self.I_buf)
        self.assertEqual(s, 0)
        
        self.I_buf[:] = int_array(0, 1, 2, 3, 4, 5)
        self.assertTrue(annihilates(6, 8, self.I_buf))
        s = exc_on_string(6, 8, self.I_buf, self.I_buf)
        self.assertEqual(s, 0)
        
        self.I_buf[:] = int_array(0, 1, 2, 3, 4, 5)
        self.assertTrue(annihilates(1, 5, self.I_buf))
        s = exc_on_string(1, 5, self.I_buf, self.I_buf)
        self.assertEqual(s, 0)
        
        self.I_buf[:] = int_array(0, 1, 2, 3, 4, 5)
        self.assertTrue(annihilates(6, 8, self.I_buf))
        s = exc_on_string(6, 8, self.I_buf, self.I_buf)
        self.assertEqual(s, 0)

    def test2(self):
        self.I_buf[:] = int_array(0, 1, 2, 6, 7, 8)
        self.assertTrue(annihilates(1, 8, self.I_buf))
        s = exc_on_string(1, 8, self.I_buf, self.I_buf)
        self.assertEqual(s, 0)

        self.I_buf[:] = int_array(0, 1, 2, 6, 7, 8)
        self.assertTrue(annihilates(4, 9, self.I_buf))
        s = exc_on_string(4, 9, self.I_buf, self.I_buf)
        self.assertEqual(s, 0)

        self.I_buf[:] = int_array(0, 1, 2, 6, 7, 8)
        self.assertTrue(annihilates(3, 9, self.I_buf))
        s = exc_on_string(3, 9, self.I_buf, self.I_buf)
        self.assertEqual(s, 0)

        self.I_buf[:] = int_array(0, 1, 2, 6, 7, 8)
        self.assertTrue(annihilates(5, 9, self.I_buf))
        s = exc_on_string(5, 9, self.I_buf, self.I_buf)
        self.assertEqual(s, 0)

        self.I_buf[:] = int_array(0, 1, 2, 6, 7, 8)
        self.assertTrue(annihilates(4, 4, self.I_buf))
        s = exc_on_string(4, 4, self.I_buf, self.I_buf)
        self.assertEqual(s, 0)

    def test3(self):
        self.I_buf[:] = int_array(0, 1, 2, 6, 7, 8)
        self.assertFalse(annihilates(1, 9, self.I_buf))
        s = exc_on_string(1, 9, self.I_buf, self.I_buf)
        self.assertNotEqual(s, 0)

        self.I_buf[:] = int_array(0, 1, 2, 6, 7, 8)
        self.assertFalse(annihilates(1, 5, self.I_buf))
        s = exc_on_string(1, 5, self.I_buf, self.I_buf)
        self.assertNotEqual(s, 0)

        self.I_buf[:] = int_array(0, 1, 2, 6, 7, 8)
        self.assertFalse(annihilates(6, 5, self.I_buf))
        s = exc_on_string(6, 5, self.I_buf, self.I_buf)
        self.assertNotEqual(s, 0)

        self.I_buf[:] = int_array(0, 1, 2, 6, 7, 8)
        self.assertFalse(annihilates(6, 6, self.I_buf))
        s = exc_on_string(6, 6, self.I_buf, self.I_buf)
        self.assertNotEqual(s, 0)