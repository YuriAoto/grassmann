"""Tests for reverse lexical order

"""
import unittest

import numpy as np

import test
from wave_functions.fci import make_occ
import wave_functions.strings_rev_lexical_order as str_order



class StringGraphsTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)


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
                         np.array([[0, 0, 0, 0],
                                   [1, 1, 1, 1],
                                   [2, 3, 4, 5]],
                                  dtype=np.intc))
    
    def test_gen_str_graph_2(self):
        nel = 3
        norb = 5
        string_gr = str_order.generate_graph(nel, norb)
        self.assertEqual(string_gr,
                         np.array([[0, 0, 0],
                                   [1, 1, 1],
                                   [2, 3, 4]],
                                  dtype=np.intc))
    
    def test_gen_str_graph_3(self):
        nel = 4
        norb = 10
        string_gr = str_order.generate_graph(nel, norb)
        self.assertEqual(string_gr,
                         np.array([[0, 0, 0, 0],
                                   [1, 1, 1, 1],
                                   [2, 3, 4, 5],
                                   [3, 6, 10, 15],
                                   [4, 10, 20, 35],
                                   [5, 15, 35, 70],
                                   [6, 21, 56, 126]],
                                  dtype=np.intc))
    
    def test_gen_str_graph_5(self):
        nel = 5
        norb = 11
        string_gr = str_order.generate_graph(nel, norb)
        self.assertEqual(string_gr,
                         np.array([[0, 0, 0, 0, 0],
                                   [1, 1, 1, 1, 1],
                                   [2, 3, 4, 5, 6],
                                   [3, 6, 10, 15, 21],
                                   [4, 10, 20, 35, 56],
                                   [5, 15, 35, 70, 126],
                                   [6, 21, 56, 126, 252]],
                                  dtype=np.intc))


class StringGraphsTestCase2(unittest.TestCase):
        
    def setUp(self):
        self.string_gr_1 = str_order.generate_graph(4, 6)
        self.string_gr_2 = str_order.generate_graph(3, 5)
        self.string_gr_3 = str_order.generate_graph(5, 11)
    
    def test_get_str_index1(self):
        self.assertEqual(str_order.get_index(make_occ([0,1,2,3]),
                                             self.string_gr_1),
                         0)
        self.assertEqual(str_order.get_index(make_occ([0,1,2,4]),
                                             self.string_gr_1),
                         1)
        self.assertEqual(str_order.get_index(make_occ([1,2,3,4]),
                                             self.string_gr_1),
                         4)
        self.assertEqual(str_order.get_index(make_occ([1,2,4,5]),
                                             self.string_gr_1),
                         11)

    def test_get_str_index2(self):
        self.assertEqual(str_order.get_index(make_occ([0,1,2]),
                                             self.string_gr_2),
                         0)
        self.assertEqual(str_order.get_index(make_occ([1,2,3]),
                                             self.string_gr_2),
                         3)
        self.assertEqual(str_order.get_index(make_occ([0,3,4]),
                                             self.string_gr_2),
                         7)
        self.assertEqual(str_order.get_index(make_occ([2,3,4]),
                                             self.string_gr_2),
                         9)


    def test_get_str_index3(self):
        self.assertEqual(str_order.get_index(make_occ([0,1,2,3,4]),
                                             self.string_gr_3),
                         0)
        self.assertEqual(str_order.get_index(make_occ([1,2,3,4,5]),
                                             self.string_gr_3),
                         5)
        self.assertEqual(str_order.get_index(make_occ([0,1,2,3,6]),
                                             self.string_gr_3),
                         6)
        self.assertEqual(str_order.get_index(make_occ([0,1,4,5,6]),
                                             self.string_gr_3),
                         15)
        
        self.assertEqual(str_order.get_index(make_occ([0,1,2,4,7]),
                                             self.string_gr_3),
                         22)


class StringGraphsTestCase3(unittest.TestCase):
    
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)
        self.string_gr_1 = str_order.generate_graph(4, 6)
        self.string_gr_2 = str_order.generate_graph(3, 5)
        self.string_gr_3 = str_order.generate_graph(5, 11)

    def test_occ_from_str_ind(self):
        self.assertEqual(
            str_order.occ_from_pos(0, self.string_gr_1),
            make_occ([0,1,2,3]))
        self.assertEqual(
            str_order.occ_from_pos(1, self.string_gr_1),
            make_occ([0,1,2,4]))
        self.assertEqual(
            str_order.occ_from_pos(4, self.string_gr_1),
            make_occ([1,2,3,4]))
        self.assertEqual(
            str_order.occ_from_pos(11, self.string_gr_1),
                         make_occ([1,2,4,5]))

    def test_occ_from_str_ind2(self):
        self.assertEqual(
            str_order.occ_from_pos(0, self.string_gr_2),
            make_occ([0,1,2]))
        self.assertEqual(
            str_order.occ_from_pos(3, self.string_gr_2),
            make_occ([1,2,3]))
        self.assertEqual(
            str_order.occ_from_pos(7, self.string_gr_2),
            make_occ([0,3,4]))
        self.assertEqual(
            str_order.occ_from_pos(9,self.string_gr_2),
            make_occ([2,3,4]))

    def test_occ_from_str_ind3(self):
        self.assertEqual(
            str_order.occ_from_pos(0, self.string_gr_3),
            make_occ([0,1,2,3,4]))
        self.assertEqual(
            str_order.occ_from_pos(6, self.string_gr_3),
            make_occ([0,1,2,3,6]))
        self.assertEqual(
            str_order.occ_from_pos(17, self.string_gr_3),
            make_occ([1,2,4,5,6]))
        self.assertEqual(
            str_order.occ_from_pos(26, self.string_gr_3),
            make_occ([0,1,2,5,7]))


class RevLexOrdTestCase3(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)

    def test_next_3el(self):
        occ = make_occ([0, 1, 2])
        str_order.next_str(occ)
        self.assertEqual(occ, make_occ([0, 1 ,3]))
        str_order.next_str(occ)
        self.assertEqual(occ, make_occ([0, 2, 3]))
        str_order.next_str(occ)
        self.assertEqual(occ, make_occ([1, 2, 3]))
        occ = make_occ([1, 4, 5])
        str_order.next_str(occ)
        self.assertEqual(occ, make_occ([2, 4, 5]))
        occ = make_occ([5, 6, 9])
        str_order.next_str(occ)
        self.assertEqual(occ, make_occ([0, 7, 9]))
        occ = make_occ([1, 6, 9])
        str_order.next_str(occ)
        self.assertEqual(occ, make_occ([2, 6, 9]))
        
    def test_next_4el(self):
        occ = make_occ([0, 1, 2, 3])
        str_order.next_str(occ)
        self.assertEqual(occ, make_occ([0, 1, 2, 4]))
        str_order.next_str(occ)
        self.assertEqual(occ, make_occ([0, 1, 3, 4]))
        occ = make_occ([0, 1, 2, 5])
        str_order.next_str(occ)
        self.assertEqual(occ, make_occ([0, 1, 3, 5]))
        occ = make_occ([0, 3, 5, 6])
        str_order.next_str(occ)
        self.assertEqual(occ, make_occ([1, 3, 5, 6]))
        occ = make_occ([4, 5, 6, 7])
        str_order.next_str(occ)
        self.assertEqual(occ, make_occ([0, 1, 2, 8]))

class SignRelRefTestCase(unittest.TestCase):

    def test_sign_rel_ref(self):
        self.assertEqual(str_order.sign_relative_to_ref(
            make_occ([5]),
            make_occ([2]),
            make_occ([0, 1, 5, 6])),
                         -1)
        self.assertEqual(str_order.sign_relative_to_ref(
            make_occ([1, 5]),
            make_occ([2, 3]),
            make_occ([0, 1, 5, 6])),
                         1)
        self.assertEqual(str_order.sign_relative_to_ref(
            make_occ([5]),
            make_occ([2]),
            make_occ([0, 1, 5])),
                         1)
        self.assertEqual(str_order.sign_relative_to_ref(
            make_occ([1, 5]),
            make_occ([2, 3]),
            make_occ([0, 1, 5])),
                         1)
