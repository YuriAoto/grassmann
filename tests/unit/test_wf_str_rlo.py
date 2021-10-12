"""Tests for reverse lexical order

"""
import unittest

import numpy as np

import tests
from util.variables import int_dtype
from util.other import int_array
import wave_functions.strings_rev_lexical_order as str_order


@tests.category('SHORT', 'ESSENTIAL')
class SignPutMaxCoincTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test1(self):
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             int_array(0, 1, 2, 3, 4, 5, 7),
                             7))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 4, 5, 7),
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             7))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             int_array(0, 1, 2, 3, 4, 5, 9),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             int_array(0, 1, 2, 3, 4, 6, 9),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 4, 6, 9),
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             7))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 5, 6, 9),
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             7))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 4, 8, 10),
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 5, 8, 10),
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             int_array(0, 1, 2, 3, 5, 8, 10),
                             7))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 4, 5, 8, 10),
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 4, 6, 8, 10),
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 4, 7, 8, 10),
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             int_array(0, 1, 2, 4, 7, 8, 10),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 2, 4, 5, 7, 8, 10),
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 4, 5, 6, 7, 8, 10),
                             int_array(0, 1, 2, 3, 4, 5, 6),
                             7))

    def test2(self):
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 8, 9, 10),
                             int_array(0, 1, 2, 3, 8, 9, 11),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 8, 9, 10),
                             int_array(0, 1, 2, 3, 8, 10, 11),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 8, 9, 10),
                             int_array(0, 1, 2, 5, 8, 10, 11),
                             7))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 8, 9, 10),
                             int_array(0, 1, 3, 5, 9, 11, 13),
                             7))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 8, 9, 10),
                             int_array(0, 1, 3, 9, 11, 13, 14),
                             7))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 8, 9, 10),
                             int_array(0, 2, 3, 9, 11, 13, 14),
                             7))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 8, 9, 10),
                             int_array(0, 2, 3, 4, 5, 9, 11),
                             7))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 2, 3, 4, 5, 9, 11),
                             int_array(0, 1, 2, 3, 8, 9, 10),
                             7))

    def test3(self):
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3, 8, 9, 10, 15),
                             int_array(0, 1, 2, 3, 8, 9, 10, 15),
                             8))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3,    8, 9, 10, 15),
                             int_array(   1, 2, 3, 4, 8, 9, 10, 15),
                             8))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3,    8, 9, 10, 15),
                             int_array(0, 1, 2, 3, 4, 8, 9, 10),
                             8))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3,       8, 9, 10, 15),
                             int_array(   1, 2, 3, 4, 5, 8, 9, 10),
                             8))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3,       8, 9, 10,    15),
                             int_array(   1,    3, 4, 5, 8, 9, 10, 13),
                             8))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1, 2, 3,    8, 9, 10,    15),
                             int_array(   1,    3, 5, 8, 9, 10,    15, 18),
                             8))
    
    def test4(self):
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0),
                             int_array(0),
                             1))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0),
                             int_array(4),
                             1))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(3),
                             int_array(7),
                             1))
    
    def test5(self):
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1),
                             int_array(0, 1),
                             2))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1),
                             int_array(0, 2),
                             2))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1),
                             int_array(0, 4),
                             2))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1),
                             int_array(2, 4),
                             2))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 1),
                             int_array(2, 3),
                             2))
    
    def test6(self):
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(2, 5),
                             int_array(2, 5),
                             2))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(2, 7),
                             int_array(2, 5),
                             2))
        self.assertEqual(-1,
                         str_order.sign_put_max_coincidence(
                             int_array(0, 2),
                             int_array(2, 5),
                             2))
        self.assertEqual(1,
                         str_order.sign_put_max_coincidence(
                             int_array(2, 5),
                             int_array(0, 1),
                             2))



@tests.category('SHORT', 'ESSENTIAL')
class StringGraphsTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
    
    def test_gen_str_graph_0(self):
        with self.assertRaises(ValueError):
            str_order.generate_graph(-1, 2)
        with self.assertRaises(ValueError):
            str_order.generate_graph(3, -3)
        with self.assertRaises(ValueError):
            str_order.generate_graph(-1, -3)
        with self.assertRaises(ValueError):
            str_order.generate_graph(5, 2)
        
    def test_gen_str_graph_1(self):
        nel = 4
        norb = 6
        string_gr = str_order.generate_graph(nel, norb)
        self.assertEqual(string_gr,
                         int_array([[0, 0, 0, 0],
                                    [1, 1, 1, 1],
                                    [2, 3, 4, 5]]))
    
    def test_gen_str_graph_2(self):
        nel = 3
        norb = 5
        string_gr = str_order.generate_graph(nel, norb)
        self.assertEqual(string_gr,
                         int_array([[0, 0, 0],
                                    [1, 1, 1],
                                    [2, 3, 4]]))
    
    def test_gen_str_graph_3(self):
        nel = 4
        norb = 10
        string_gr = str_order.generate_graph(nel, norb)
        self.assertEqual(string_gr,
                         int_array([[0, 0, 0, 0],
                                    [1, 1, 1, 1],
                                    [2, 3, 4, 5],
                                    [3, 6, 10, 15],
                                    [4, 10, 20, 35],
                                    [5, 15, 35, 70],
                                    [6, 21, 56, 126]]))
    
    def test_gen_str_graph_5(self):
        nel = 5
        norb = 11
        string_gr = str_order.generate_graph(nel, norb)
        self.assertEqual(string_gr,
                         int_array([[0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1],
                                    [2, 3, 4, 5, 6],
                                    [3, 6, 10, 15, 21],
                                    [4, 10, 20, 35, 56],
                                    [5, 15, 35, 70, 126],
                                    [6, 21, 56, 126, 252]]))


@tests.category('SHORT', 'ESSENTIAL')
class StringGraphs2TestCase(unittest.TestCase):
        
    def setUp(self):
        self.string_gr_1 = str_order.generate_graph(4, 6)
        self.string_gr_2 = str_order.generate_graph(3, 5)
        self.string_gr_3 = str_order.generate_graph(5, 11)
    
    def test_get_str_index1(self):
        self.assertEqual(str_order.get_index(int_array(0, 1, 2, 3),
                                             self.string_gr_1),
                         0)
        self.assertEqual(str_order.get_index(int_array(0, 1, 2, 4),
                                             self.string_gr_1),
                         1)
        self.assertEqual(str_order.get_index(int_array(1, 2, 3, 4),
                                             self.string_gr_1),
                         4)
        self.assertEqual(str_order.get_index(int_array(1, 2, 4, 5),
                                             self.string_gr_1),
                         11)

    def test_get_str_index2(self):
        self.assertEqual(str_order.get_index(int_array(0, 1, 2),
                                             self.string_gr_2),
                         0)
        self.assertEqual(str_order.get_index(int_array(1, 2, 3),
                                             self.string_gr_2),
                         3)
        self.assertEqual(str_order.get_index(int_array(0, 3, 4),
                                             self.string_gr_2),
                         7)
        self.assertEqual(str_order.get_index(int_array(2, 3, 4),
                                             self.string_gr_2),
                         9)

    def test_get_str_index3(self):
        self.assertEqual(str_order.get_index(int_array(0, 1, 2, 3, 4),
                                             self.string_gr_3),
                         0)
        self.assertEqual(str_order.get_index(int_array(1, 2, 3, 4, 5),
                                             self.string_gr_3),
                         5)
        self.assertEqual(str_order.get_index(int_array(0, 1, 2, 3, 6),
                                             self.string_gr_3),
                         6)
        self.assertEqual(str_order.get_index(int_array(0, 1, 4, 5, 6),
                                             self.string_gr_3),
                         15)
        
        self.assertEqual(str_order.get_index(int_array(0, 1, 2, 4, 7),
                                             self.string_gr_3),
                         22)


@tests.category('SHORT')
class StringGraphs3TestCase(unittest.TestCase):
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.string_gr_1 = str_order.generate_graph(4, 6)
        self.string_gr_2 = str_order.generate_graph(3, 5)
        self.string_gr_3 = str_order.generate_graph(5, 11)

    def test_occ_from_str_ind(self):
        self.assertEqual(
            str_order.occ_from_pos(0, self.string_gr_1),
            int_array(0, 1, 2, 3))
        self.assertEqual(
            str_order.occ_from_pos(1, self.string_gr_1),
            int_array(0, 1, 2, 4))
        self.assertEqual(
            str_order.occ_from_pos(4, self.string_gr_1),
            int_array(1, 2, 3, 4))
        self.assertEqual(
            str_order.occ_from_pos(11, self.string_gr_1),
            int_array(1, 2, 4, 5))

    def test_occ_from_str_ind2(self):
        self.assertEqual(
            str_order.occ_from_pos(0, self.string_gr_2),
            int_array(0, 1, 2))
        self.assertEqual(
            str_order.occ_from_pos(3, self.string_gr_2),
            int_array(1, 2, 3))
        self.assertEqual(
            str_order.occ_from_pos(7, self.string_gr_2),
            int_array(0, 3, 4))
        self.assertEqual(
            str_order.occ_from_pos(9, self.string_gr_2),
            int_array(2, 3, 4))

    def test_occ_from_str_ind3(self):
        self.assertEqual(
            str_order.occ_from_pos(0, self.string_gr_3),
            int_array(0, 1, 2, 3, 4))
        self.assertEqual(
            str_order.occ_from_pos(6, self.string_gr_3),
            int_array(0, 1, 2, 3, 6))
        self.assertEqual(
            str_order.occ_from_pos(17, self.string_gr_3),
            int_array(1, 2, 4, 5, 6))
        self.assertEqual(
            str_order.occ_from_pos(26, self.string_gr_3),
            int_array(0, 1, 2, 5, 7))


@tests.category('SHORT')
class RevLexOrd3TestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_next_3el(self):
        occ = int_array(0, 1, 2)
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(0, 1, 3))
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(0, 2, 3))
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(1, 2, 3))
        occ = int_array(1, 4, 5)
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(2, 4, 5))
        occ = int_array(5, 6, 9)
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(0, 7, 9))
        occ = int_array(1, 6, 9)
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(2, 6, 9))
        
    def test_next_4el(self):
        occ = int_array(0, 1, 2, 3)
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(0, 1, 2, 4))
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(0, 1, 3, 4))
        occ = int_array(0, 1, 2, 5)
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(0, 1, 3, 5))
        occ = int_array(0, 3, 5, 6)
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(1, 3, 5, 6))
        occ = int_array(4, 5, 6, 7)
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(0, 1, 2, 8))


@tests.category('SHORT')
class SignRelRefTestCase(unittest.TestCase):

    def test_sign_rel_ref(self):
        self.assertEqual(str_order.sign_relative_to_ref(
            int_array(5),
            int_array(2),
            int_array(0, 1, 5, 6)),
                         -1)
        self.assertEqual(str_order.sign_relative_to_ref(
            int_array(1, 5),
            int_array(2, 3),
            int_array(0, 1, 5, 6)),
                         1)
        self.assertEqual(str_order.sign_relative_to_ref(
            int_array(5),
            int_array(2),
            int_array(0, 1, 5)),
                         1)
        self.assertEqual(str_order.sign_relative_to_ref(
            int_array(1, 5),
            int_array(2, 3),
            int_array(0, 1, 5)),
                         1)


@tests.category('SHORT', 'ESSENTIAL')
class IniStrTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
    
    def test_1(self):
        occ = np.empty(3, dtype=int_dtype)
        str_order.ini_str(occ)
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(0, 1, 2))

    def test_2(self):
        occ = np.empty(5, dtype=int_dtype)
        str_order.ini_str(occ)
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(0, 1, 2, 3, 4))

    def test_3(self):
        occ = np.empty(10, dtype=int_dtype)
        str_order.ini_str(occ)
        str_order.next_str(occ)
        self.assertEqual(occ, int_array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
