"""Tests for fci

"""
import unittest

import numpy as np
from scipy.special import comb

from wave_functions import fci
import wave_functions.strings_rev_lexical_order as str_order
import tests

@tests.category('SHORT', 'ESSENTIAL')
class StringGraphsTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.norb1 = 6
        self.nel1 = 4
        self.norb2 = 5
        self.nel2 = 3
        self.string_gr_1 = str_order.generate_graph(self.nel1, self.norb1)
        self.string_gr_2 = str_order.generate_graph(self.nel2, self.norb2)

    def test_occ_fromto_strind(self):
        for i in range(comb(self.norb1, self.nel1, exact=True)):
            self.assertEqual(
                str_order.get_index(str_order.occ_from_pos(
                    i, self.string_gr_1), self.string_gr_1),
                i)

    def test_occ_fromto_strind2(self):
        for i in range(comb(self.norb2, self.nel2, exact=True)):
            self.assertEqual(
                str_order.get_index(str_order.occ_from_pos(
                    i, self.string_gr_2), self.string_gr_2),
                i)
