"""Tests for util.array_indices

"""
import unittest

from util.array_indices import (
    triangular,
    get_ij_from_triang, get_n_from_triang,
    get_ij_from_triang_with_diag, get_n_from_triang_with_diag,
    get_pos_from_rectangular, get_ia_from_rectangular)

class TriangTestCase(unittest.TestCase):

    def test_triang(self):
        self.assertEqual(triangular(0), 0)
        self.assertEqual(triangular(1), 1)
        self.assertEqual(triangular(2), 3)
        self.assertEqual(triangular(3), 6)
        self.assertEqual(triangular(4), 10)
        self.assertEqual(triangular(5), 15)
    
    def test_ij_from_triang(self):
        self.assertEqual(get_ij_from_triang(0), (0, 1))
        self.assertEqual(get_ij_from_triang(1), (0, 2))
        self.assertEqual(get_ij_from_triang(2), (1, 2))
        self.assertEqual(get_ij_from_triang(3), (0, 3))
        self.assertEqual(get_ij_from_triang(4), (1, 3))
        self.assertEqual(get_ij_from_triang(5), (2, 3))
        self.assertEqual(get_ij_from_triang(6), (0, 4))
        self.assertEqual(get_ij_from_triang(7), (1, 4))
        self.assertEqual(get_ij_from_triang(8), (2, 4))
        self.assertEqual(get_ij_from_triang(9), (3, 4))
    
    def test_n_from_triang(self):
        self.assertEqual(get_n_from_triang(0, 1), 0)
        self.assertEqual(get_n_from_triang(0, 2), 1)
        self.assertEqual(get_n_from_triang(1, 2), 2)
        self.assertEqual(get_n_from_triang(0, 3), 3)
        self.assertEqual(get_n_from_triang(1, 3), 4)
        self.assertEqual(get_n_from_triang(2, 3), 5)
        self.assertEqual(get_n_from_triang(0, 4), 6)
        self.assertEqual(get_n_from_triang(1, 4), 7)
        self.assertEqual(get_n_from_triang(2, 4), 8)
        self.assertEqual(get_n_from_triang(3, 4), 9)

    def test_ij_from_triang_with_diag(self):
        self.assertEqual(get_ij_from_triang_with_diag(0), (0, 0))
        self.assertEqual(get_ij_from_triang_with_diag(1), (0, 1))
        self.assertEqual(get_ij_from_triang_with_diag(2), (1, 1))
        self.assertEqual(get_ij_from_triang_with_diag(3), (0, 2))
        self.assertEqual(get_ij_from_triang_with_diag(4), (1, 2))
        self.assertEqual(get_ij_from_triang_with_diag(5), (2, 2))
        self.assertEqual(get_ij_from_triang_with_diag(6), (0, 3))
        self.assertEqual(get_ij_from_triang_with_diag(7), (1, 3))
        self.assertEqual(get_ij_from_triang_with_diag(8), (2, 3))
        self.assertEqual(get_ij_from_triang_with_diag(9), (3, 3))
        self.assertEqual(get_ij_from_triang_with_diag(10), (0, 4))
        self.assertEqual(get_ij_from_triang_with_diag(11), (1, 4))
        self.assertEqual(get_ij_from_triang_with_diag(12), (2, 4))
        self.assertEqual(get_ij_from_triang_with_diag(13), (3, 4))
        self.assertEqual(get_ij_from_triang_with_diag(14), (4, 4))

    def test_n_from_triang_with_diag(self):
        self.assertEqual(get_n_from_triang_with_diag(0, 0), 0 )
        self.assertEqual(get_n_from_triang_with_diag(0, 1), 1 )
        self.assertEqual(get_n_from_triang_with_diag(1, 1), 2 )
        self.assertEqual(get_n_from_triang_with_diag(0, 2), 3 )
        self.assertEqual(get_n_from_triang_with_diag(1, 2), 4 )
        self.assertEqual(get_n_from_triang_with_diag(2, 2), 5 )
        self.assertEqual(get_n_from_triang_with_diag(0, 3), 6 )
        self.assertEqual(get_n_from_triang_with_diag(1, 3), 7 )
        self.assertEqual(get_n_from_triang_with_diag(2, 3), 8 )
        self.assertEqual(get_n_from_triang_with_diag(3, 3), 9 )
        self.assertEqual(get_n_from_triang_with_diag(0, 4), 10)
        self.assertEqual(get_n_from_triang_with_diag(1, 4), 11)
        self.assertEqual(get_n_from_triang_with_diag(2, 4), 12)
        self.assertEqual(get_n_from_triang_with_diag(3, 4), 13)
        self.assertEqual(get_n_from_triang_with_diag(4, 4), 14)

    def test_go_and_back(self):
        for n in range(10, 50):
            with self.subTest(n=n):
                self.assertEqual(get_n_from_triang(*get_ij_from_triang(n)), n)

    def test_go_and_back_with_diag(self):
        for n in range(15, 50):
            with self.subTest(n=n):
                self.assertEqual(get_n_from_triang_with_diag(
                    *get_ij_from_triang_with_diag(n)), n)


class RectTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 5

    def test_get_pos(self):
        self.assertEqual(get_pos_from_rectangular(0, 0, self.n), 0)
        self.assertEqual(get_pos_from_rectangular(0, 1, self.n), 1)
        self.assertEqual(get_pos_from_rectangular(0, 2, self.n), 2)
        self.assertEqual(get_pos_from_rectangular(0, 3, self.n), 3)
        self.assertEqual(get_pos_from_rectangular(0, 4, self.n), 4)
        self.assertEqual(get_pos_from_rectangular(1, 0, self.n), 5)
        self.assertEqual(get_pos_from_rectangular(1, 1, self.n), 6)
        self.assertEqual(get_pos_from_rectangular(1, 2, self.n), 7)
        self.assertEqual(get_pos_from_rectangular(1, 3, self.n), 8)
        self.assertEqual(get_pos_from_rectangular(1, 4, self.n), 9)
        self.assertEqual(get_pos_from_rectangular(2, 0, self.n), 10)
        self.assertEqual(get_pos_from_rectangular(2, 1, self.n), 11)
        self.assertEqual(get_pos_from_rectangular(2, 2, self.n), 12)
        self.assertEqual(get_pos_from_rectangular(2, 3, self.n), 13)
        self.assertEqual(get_pos_from_rectangular(2, 4, self.n), 14)
        self.assertEqual(get_pos_from_rectangular(3, 0, self.n), 15)
        self.assertEqual(get_pos_from_rectangular(3, 1, self.n), 16)
        self.assertEqual(get_pos_from_rectangular(3, 2, self.n), 17)
        self.assertEqual(get_pos_from_rectangular(3, 3, self.n), 18)
        self.assertEqual(get_pos_from_rectangular(3, 4, self.n), 19)
    
    def test_get_ia(self):
        self.assertEqual(get_ia_from_rectangular(0 , self.n), (0, 0))
        self.assertEqual(get_ia_from_rectangular(1 , self.n), (0, 1))
        self.assertEqual(get_ia_from_rectangular(2 , self.n), (0, 2))
        self.assertEqual(get_ia_from_rectangular(3 , self.n), (0, 3))
        self.assertEqual(get_ia_from_rectangular(4 , self.n), (0, 4))
        self.assertEqual(get_ia_from_rectangular(5 , self.n), (1, 0))
        self.assertEqual(get_ia_from_rectangular(6 , self.n), (1, 1))
        self.assertEqual(get_ia_from_rectangular(7 , self.n), (1, 2))
        self.assertEqual(get_ia_from_rectangular(8 , self.n), (1, 3))
        self.assertEqual(get_ia_from_rectangular(9 , self.n), (1, 4))
        self.assertEqual(get_ia_from_rectangular(10, self.n), (2, 0))
        self.assertEqual(get_ia_from_rectangular(11, self.n), (2, 1))
        self.assertEqual(get_ia_from_rectangular(12, self.n), (2, 2))
        self.assertEqual(get_ia_from_rectangular(13, self.n), (2, 3))
        self.assertEqual(get_ia_from_rectangular(14, self.n), (2, 4))
        self.assertEqual(get_ia_from_rectangular(15, self.n), (3, 0))
        self.assertEqual(get_ia_from_rectangular(16, self.n), (3, 1))
        self.assertEqual(get_ia_from_rectangular(17, self.n), (3, 2))
        self.assertEqual(get_ia_from_rectangular(18, self.n), (3, 3))
        self.assertEqual(get_ia_from_rectangular(19, self.n), (3, 4))

    def test_go_and_back(self):
        for n in range(20, 50):
            with self.subTest(n=n):
                self.assertEqual(get_pos_from_rectangular(
                    *get_ia_from_rectangular(n, self.n), self.n), n)
