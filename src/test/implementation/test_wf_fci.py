"""Tests for fci

"""
import unittest

import numpy as np
from scipy.special import comb

from wave_functions import fci, general
from wave_functions.fci import make_occ
import test


class StringGraphsTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)
        self.norb1 = 6
        self.nel1 = 4
        self.norb2 = 5
        self.nel2 = 3
        self.string_gr_1 = fci._generate_string_graph(self.nel1, self.norb1)
        self.string_gr_2 = fci._generate_string_graph(self.nel2, self.norb2)

    def test_occ_fromto_strind(self):
        for i in range(comb(self.norb1, self.nel1, exact=True)):
            self.assertEqual(
                fci._get_string_index(fci._occupation_from_string_index(
                    i, self.string_gr_1), self.string_gr_1),
                i)

    def test_occ_fromto_strind2(self):
        for i in range(comb(self.norb2, self.nel2, exact=True)):
            self.assertEqual(
                fci._get_string_index(fci._occupation_from_string_index(
                    i, self.string_gr_2), self.string_gr_2),
                i)
