"""


"""
import unittest

import numpy as np

import tests
from util.other import int_array
from wave_functions.slater_det import SlaterDet
from coupled_cluster.cluster_decomposition import cluster_decompose


class ClusterDecTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.ref_det = SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,2,3,4,5,6),
            beta_occ=int_array(0,1,2,3,4,5,6))
    
    def test_cluster_decompose1(self):
        """t_{0 1}^{7 8}   (all in alpha)
        
        - t_0^7 t_1^8
        + t_0^8 t_1^7
        """
        dec = cluster_decompose(
            (int_array(0, 1), int_array(7, 8)),
            (int_array(), int_array()),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 2)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][2].alpha_occ, int_array(0,2,3,4,5,6,8))
        self.assertEqual(dec[0][2].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][0], 1)
        self.assertEqual(dec[1][1].alpha_occ, int_array(1,2,3,4,5,6,8))
        self.assertEqual(dec[1][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][2].alpha_occ, int_array(0,2,3,4,5,6,7))
        self.assertEqual(dec[1][2].beta_occ, int_array(0,1,2,3,4,5,6))
    
    def test_cluster_decompose2(self):
        """t_{3 5}^{9 12}    (all in alpha)
        
        - t_3^9 t_5^12
        + t_3^12 t_5^9
        """
        dec = cluster_decompose(
            (int_array(3, 5), int_array(9, 12)),
            (int_array(), int_array()),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 2)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(0,1,2,4,5,6,9))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][2].alpha_occ, int_array(0,1,2,3,4,6,12))
        self.assertEqual(dec[0][2].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][0], 1)
        self.assertEqual(dec[1][1].alpha_occ, int_array(0,1,2,4,5,6,12))
        self.assertEqual(dec[1][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][2].alpha_occ, int_array(0,1,2,3,4,6,9))
        self.assertEqual(dec[1][2].beta_occ, int_array(0,1,2,3,4,5,6))
    
    def test_cluster_decompose3(self):
        """t_{0 1 2 3}^{7 8 9 10}  (all in alpha, only D)
        
        - t_{0 1}^{7 8} t_{2 3}^{9 10}
        + t_{0 1}^{7 9} t_{2 3}^{8 10}
        + t_{0 2}^{7 8} t_{1 3}^{9 10}
        
        """
        dec = cluster_decompose(
            (int_array(0, 1, 2, 3), int_array(7, 8, 9, 10)),
            (int_array(), int_array()),
             self.ref_det)
        self.assertEqual(len(dec), 18)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(2,3,4,5,6,7,8))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][2].alpha_occ, int_array(0,1,4,5,6,9,10))
        self.assertEqual(dec[0][2].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][0], 1)
        self.assertEqual(dec[1][1].alpha_occ, int_array(2,3,4,5,6,7,9))
        self.assertEqual(dec[1][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][2].alpha_occ, int_array(0,1,4,5,6,8,10))
        self.assertEqual(dec[1][2].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[6][0], 1)
        self.assertEqual(dec[6][1].alpha_occ, int_array(1,3,4,5,6,7,8))
        self.assertEqual(dec[6][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[6][2].alpha_occ, int_array(0,2,4,5,6,9,10))
        self.assertEqual(dec[6][2].beta_occ, int_array(0,1,2,3,4,5,6))
    
    def test_cluster_decompose4(self):
        """t_{0 1 2}^{7 8 9}  (all in alpha)
        
        - t_0^7 t_1^8 t_2^9
        - t_0^8 t_1^9 t_2^7
        - t_0^7 t_{2 3}^{8 9}
        + t_0^9 t_{0 2}^{7 8}
        """
        dec = cluster_decompose(
            (int_array(0, 1, 2), int_array(7, 8, 9)),
            (int_array(), int_array()),
            self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 15)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][2].alpha_occ, int_array(0,2,3,4,5,6,8))
        self.assertEqual(dec[0][2].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][3].alpha_occ, int_array(0,1,3,4,5,6,9))
        self.assertEqual(dec[0][3].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[3][0], -1)
        self.assertEqual(dec[3][1].alpha_occ, int_array(1,2,3,4,5,6,8))
        self.assertEqual(dec[3][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[3][2].alpha_occ, int_array(0,2,3,4,5,6,9))
        self.assertEqual(dec[3][2].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[3][3].alpha_occ, int_array(0,1,3,4,5,6,7))
        self.assertEqual(dec[3][3].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[6][0], -1)
        self.assertEqual(dec[6][1].alpha_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[6][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[6][2].alpha_occ, int_array(0,3,4,5,6,8,9))
        self.assertEqual(dec[6][2].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[11][0], 1)
        self.assertEqual(dec[11][1].alpha_occ, int_array(0,2,3,4,5,6,9))
        self.assertEqual(dec[11][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[11][2].alpha_occ, int_array(1,3,4,5,6,7,8))
        self.assertEqual(dec[11][2].beta_occ, int_array(0,1,2,3,4,5,6))
    
    def test_cluster_decompose5(self):
        """t_{0a 0b}^{7a 7b}
        
        - t_0a^7a t_0b^7b
        """
        dec = cluster_decompose(
            (int_array(0), int_array(7)),
            (int_array(0), int_array(7)),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 1)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][2].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][2].beta_occ, int_array(1,2,3,4,5,6,7))
    
    def test_cluster_decompose6(self):
        """t_{0a 3b}^{9a 7b}
        
        - t_0a^9a t_3b^7b
        """
        dec = cluster_decompose(
            (int_array(0), int_array(9)),
            (int_array(3), int_array(7)),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 1)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(1,2,3,4,5,6,9))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][2].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][2].beta_occ, int_array(0,1,2,4,5,6,7))


    def test_cluster_decompose7(self):
        """t_{3 5}^{9 12}    (all in beta)
        
        - t_3^9 t_5^12
        + t_3^12 t_5^9
        """
        dec = cluster_decompose(
            (int_array(), int_array()),
            (int_array(3, 5), int_array(9, 12)),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 2)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,4,5,6,9))
        self.assertEqual(dec[0][2].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][2].beta_occ, int_array(0,1,2,3,4,6,12))
        self.assertEqual(dec[1][0], 1)
        self.assertEqual(dec[1][1].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][1].beta_occ, int_array(0,1,2,4,5,6,12))
        self.assertEqual(dec[1][2].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][2].beta_occ, int_array(0,1,2,3,4,6,9))

    def test_cluster_decompose3(self):
        """t_{0a 1a 2b 3b}^{7a 8a 9b 10b}  (only D)
        
        - t_{0a 1a}^{7a 8a} t_{2b 3b}^{9b 10b}
        - t_{0a 2b}^{7a 9b} t_{1a 3b}^{8a 10b}
        + t_{0a 2b}^{7a 10b} t_{1a 3b}^{8a 9b}
        + t_{0a 2b}^{8a 9b} t_{1a 3b}^{7a 10b}
        - t_{0a 2b}^{8a 10b} t_{1a 3b}^{7a 9b}
        + t_{0a 3b}^{7a 9b} t_{1a 2b}^{8a 10b}
        - t_{0a 3b}^{7a 10b} t_{1a 2b}^{8a 9b}
        - t_{0a 3b}^{8a 9b} t_{1a 2b}^{7a 10b}
        + t_{0a 3b}^{8a 10b} t_{1a 2b}^{7a 9b}
        
        """
        dec = cluster_decompose(
            (int_array(0, 1), int_array(7, 8)),
            (int_array(2, 3), int_array(9, 10)),
             self.ref_det)
        self.assertEqual(len(dec), 9)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(2,3,4,5,6,7,8))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][2].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][2].beta_occ, int_array(0,1,4,5,6,9,10))
        self.assertEqual(dec[1][0], -1)
        self.assertEqual(dec[1][1].alpha_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[1][1].beta_occ, int_array(0,1,3,4,5,6,9))
        self.assertEqual(dec[1][2].alpha_occ, int_array(0,2,3,4,5,6,8))
        self.assertEqual(dec[1][2].beta_occ, int_array(0,1,2,4,5,6,10))
        self.assertEqual(dec[2][0], 1)
        self.assertEqual(dec[2][1].alpha_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[2][1].beta_occ, int_array(0,1,3,4,5,6,10))
        self.assertEqual(dec[2][2].alpha_occ, int_array(0,2,3,4,5,6,8))
        self.assertEqual(dec[2][2].beta_occ, int_array(0,1,2,4,5,6,9))
        self.assertEqual(dec[3][0], 1)
        self.assertEqual(dec[3][1].alpha_occ, int_array(1,2,3,4,5,6,8))
        self.assertEqual(dec[3][1].beta_occ, int_array(0,1,3,4,5,6,9))
        self.assertEqual(dec[3][2].alpha_occ, int_array(0,2,3,4,5,6,7))
        self.assertEqual(dec[3][2].beta_occ, int_array(0,1,2,4,5,6,10))
        self.assertEqual(dec[4][0], -1)
        self.assertEqual(dec[4][1].alpha_occ, int_array(1,2,3,4,5,6,8))
        self.assertEqual(dec[4][1].beta_occ, int_array(0,1,3,4,5,6,10))
        self.assertEqual(dec[4][2].alpha_occ, int_array(0,2,3,4,5,6,7))
        self.assertEqual(dec[4][2].beta_occ, int_array(0,1,2,4,5,6,9))
        self.assertEqual(dec[5][0], 1)
        self.assertEqual(dec[5][1].alpha_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[5][1].beta_occ, int_array(0,1,2,4,5,6,9))
        self.assertEqual(dec[5][2].alpha_occ, int_array(0,2,3,4,5,6,8))
        self.assertEqual(dec[5][2].beta_occ, int_array(0,1,3,4,5,6,10))
        self.assertEqual(dec[6][0], -1)
        self.assertEqual(dec[6][1].alpha_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[6][1].beta_occ, int_array(0,1,2,4,5,6,10))
        self.assertEqual(dec[6][2].alpha_occ, int_array(0,2,3,4,5,6,8))
        self.assertEqual(dec[6][2].beta_occ, int_array(0,1,3,4,5,6,9))
        self.assertEqual(dec[7][0], -1)
        self.assertEqual(dec[7][1].alpha_occ, int_array(1,2,3,4,5,6,8))
        self.assertEqual(dec[7][1].beta_occ, int_array(0,1,2,4,5,6,9))
        self.assertEqual(dec[7][2].alpha_occ, int_array(0,2,3,4,5,6,7))
        self.assertEqual(dec[7][2].beta_occ, int_array(0,1,3,4,5,6,10))
        self.assertEqual(dec[8][0], 1)
        self.assertEqual(dec[8][1].alpha_occ, int_array(1,2,3,4,5,6,8))
        self.assertEqual(dec[8][1].beta_occ, int_array(0,1,2,4,5,6,10))
        self.assertEqual(dec[8][2].alpha_occ, int_array(0,2,3,4,5,6,7))
        self.assertEqual(dec[8][2].beta_occ, int_array(0,1,3,4,5,6,9))
