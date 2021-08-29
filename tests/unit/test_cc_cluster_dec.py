"""


"""
import unittest

import numpy as np

import tests
from util.other import int_array
from wave_functions.slater_det import SlaterDet
from coupled_cluster.cluster_decomposition import cluster_decompose
from coupled_cluster import coupled_cluster as cc
from wave_functions.interm_norm import IntermNormWaveFunction as WF
from orbitals.orbital_space import FullOrbitalSpace, OrbitalSpace
from coupled_cluster import coupled_cluster
from coupled_cluster import optimiser
from integrals import integrals



@tests.category('SHORT', 'ESSENTIAL')
class ClusterDecTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.ref_det = SlaterDet(
            c=0.0,
            alpha_occ=int_array(0,1,2,3,4,5,6),
            beta_occ=int_array(0,1,2,3,4,5,6))
    
    def test_1(self):
        """-c_{0 1}^{7 8}   (all in alpha)
        
        - t_{0 1}^{7 8}
        - t_0^7 t_1^8
        + t_0^8 t_1^7
        """
        dec = cluster_decompose(
            (int_array(0, 1), int_array(7, 8)),
            (int_array(), int_array()),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 3)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(2,3,4,5,6,7,8))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[1][1].alpha_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[1][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][2].alpha_occ, int_array(0,2,3,4,5,6,8))
        self.assertEqual(dec[1][2].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[2][0], 1)
        self.assertEqual(dec[2][1].alpha_occ, int_array(1,2,3,4,5,6,8))
        self.assertEqual(dec[2][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[2][2].alpha_occ, int_array(0,2,3,4,5,6,7))
        self.assertEqual(dec[2][2].beta_occ, int_array(0,1,2,3,4,5,6))
    
    def test_2(self):
        """-c_{3 5}^{9 12}    (all in alpha)
        
        - t_{3 5}^{9 12}
        - t_3^9 t_5^12
        + t_3^12 t_5^9
        """
        dec = cluster_decompose(
            (int_array(3, 5), int_array(9, 12)),
            (int_array(), int_array()),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 3)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(0,1,2,4,6,9,12))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[1][1].alpha_occ, int_array(0,1,2,4,5,6,9))
        self.assertEqual(dec[1][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][2].alpha_occ, int_array(0,1,2,3,4,6,12))
        self.assertEqual(dec[1][2].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[2][0], 1)
        self.assertEqual(dec[2][1].alpha_occ, int_array(0,1,2,4,5,6,12))
        self.assertEqual(dec[2][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[2][2].alpha_occ, int_array(0,1,2,3,4,6,9))
        self.assertEqual(dec[2][2].beta_occ, int_array(0,1,2,3,4,5,6))
    
    def test_3(self):
        """-c_{0 1 2 3}^{7 8 9 10}  (all in alpha, only D)
        
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
    
    def test_4(self):
        """-c_{0 1 2}^{7 8 9}  (all in alpha)
        
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
    
    def test_5(self):
        """-c_{0a 0b}^{7a 7b}
        
        - t_{0a 0b}^{7a 7b}
        - t_0a^7a t_0b^7b
        """
        dec = cluster_decompose(
            (int_array(0), int_array(7)),
            (int_array(0), int_array(7)),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 2)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[0][1].beta_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[1][0], -1)
        self.assertEqual(dec[1][1].alpha_occ, int_array(1,2,3,4,5,6,7))
        self.assertEqual(dec[1][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][2].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][2].beta_occ, int_array(1,2,3,4,5,6,7))
    
    def test_6(self):
        """-c_{0a 3b}^{9a 7b}
        
        - t_{0a 3b}^{9a 7b}
        - t_0a^9a t_3b^7b
        """
        dec = cluster_decompose(
            (int_array(0), int_array(9)),
            (int_array(3), int_array(7)),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 2)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(1,2,3,4,5,6,9))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,4,5,6,7))
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[1][1].alpha_occ, int_array(1,2,3,4,5,6,9))
        self.assertEqual(dec[1][1].beta_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][2].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][2].beta_occ, int_array(0,1,2,4,5,6,7))


    def test_7(self):
        """-c_{3 5}^{9 12}    (all in beta)
        
        - t_{3 5}^{9 12}
        - t_3^9 t_5^12
        + t_3^12 t_5^9
        """
        dec = cluster_decompose(
            (int_array(), int_array()),
            (int_array(3, 5), int_array(9, 12)),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 3)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[0][1].beta_occ, int_array(0,1,2,4,6,9,12))
        self.assertEqual(dec[1][0], -1)
        self.assertEqual(dec[1][1].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][1].beta_occ, int_array(0,1,2,4,5,6,9))
        self.assertEqual(dec[1][2].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[1][2].beta_occ, int_array(0,1,2,3,4,6,12))
        self.assertEqual(dec[2][0], 1)
        self.assertEqual(dec[2][1].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[2][1].beta_occ, int_array(0,1,2,4,5,6,12))
        self.assertEqual(dec[2][2].alpha_occ, int_array(0,1,2,3,4,5,6))
        self.assertEqual(dec[2][2].beta_occ, int_array(0,1,2,3,4,6,9))

    def test_3(self):
        """-c_{0a 1a 2b 3b}^{7a 8a 9b 10b}  (only D)
        
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


    def test_cc_t1_transf_1e_ij(self):
        n = 10
        no = 5
        nv = n - no
        point_group = 'C1'
        orbspace = FullOrbitalSpace(1)
        orbspace.set_full(OrbitalSpace(dim=[n], orb_type='R'))
        orbspace.set_ref(OrbitalSpace(dim=[no], orb_type='R'))
        orbspace.set_froz(OrbitalSpace(dim=[0], orb_type='R'))
        wfcc = WF.from_zero_amplitudes(point_group, orbspace, level='SD')
        wfcc.update_amplitudes(np.array([0.44, 0.52, 0.12, 0.50, 0.59, 0.60, 0.01, 0.48, 0.14, 0.46, 0.20, 0.50, 0.67, 0.04, 0.10, 0.44, 0.04, 0.72, 0.40, 0.02, 0.18, 0.98, 0.87, 0.77, 0.38]))
        h_sym = np.asarray([0.30,0.55,0.58,0.44,0.77,0.86,0.84,0.74,0.21,0.56,0.46,0.79,0.73,0.77,0.13,0.45,0.70,0.22,0.69,0.43,0.04,1.00,0.55,0.48,0.80,0.21,0.85,0.37,0.48,0.26,0.15,0.03,0.37,0.71,0.69,0.38,0.72,0.72,0.35,0.10,0.30,0.20,0.95,0.06,0.84,0.94,0.21,0.53,0.23,0.77,0.16,0.42,0.37,0.91,0.98])
        h = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    ij = i + j*(j+1)//2
                else:
                    ij = j + i*(i+1)//2
                h[i,j] = h_sym[ij]
        ht1 = cc.t1_1e_transf_oo(h,wfcc)
        ht1_corr = [[1.9902,1.5936,1.4744,1.7304,2.8502],[1.6591,1.3277,1.4090,1.5494,2.3154],[1.2921,1.2716,1.3115,0.5846,1.8414],[1.7489,1.2962,0.7951,0.9618,1.8687],[1.4071,1.6239,1.2579,1.3694,1.2587]]
        for i in range(no):
            for j in range(no):
                self.assertAlmostEqual(ht1[i,j],ht1_corr[i][j])

    def test_cc_t1_transf_1e_ai(self):
        n = 10
        no = 5
        nv = n - no
        point_group = 'C1'
        orbspace = FullOrbitalSpace(1)
        orbspace.set_full(OrbitalSpace(dim=[n], orb_type='R'))
        orbspace.set_ref(OrbitalSpace(dim=[no], orb_type='R'))
        orbspace.set_froz(OrbitalSpace(dim=[0], orb_type='R'))
        wfcc = WF.from_zero_amplitudes(point_group, orbspace, level='SD')
        wfcc.update_amplitudes(np.array([0.44, 0.52, 0.12, 0.50, 0.59, 0.60, 0.01, 0.48, 0.14, 0.46, 0.20, 0.50, 0.67, 0.04, 0.10, 0.44, 0.04, 0.72, 0.40, 0.02, 0.18, 0.98, 0.87, 0.77, 0.38]))
        h_sym = np.asarray([0.30,0.55,0.58,0.44,0.77,0.86,0.84,0.74,0.21,0.56,0.46,0.79,0.73,0.77,0.13,0.45,0.70,0.22,0.69,0.43,0.04,1.00,0.55,0.48,0.80,0.21,0.85,0.37,0.48,0.26,0.15,0.03,0.37,0.71,0.69,0.38,0.72,0.72,0.35,0.10,0.30,0.20,0.95,0.06,0.84,0.94,0.21,0.53,0.23,0.77,0.16,0.42,0.37,0.91,0.98])
        h = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    ij = i + j*(j+1)//2
                else:
                    ij = j + i*(i+1)//2
                h[i,j] = h_sym[ij]
        ht1 = cc.t1_1e_transf_vo(h,wfcc)
        ht1_corr = [[-1.963162, -1.439854, -1.180002, -1.14162, -1.957702],[-0.774459, -1.399919, -1.323774, -0.514086, -1.517232],[-2.939184, -2.972657, -2.467798, -2.551916, -3.436287],[-0.795985, -1.387925, -1.243743, -1.921258, -1.227791],[-0.32989, -1.255132, -0.98879, -1.364528, -0.841122]]
        for i in range(no):
            for j in range(no):
                self.assertAlmostEqual(ht1[i,j],ht1_corr[i][j])


    def test_cc_t1_transf_1e_ab(self):
        n = 10
        no = 5
        nv = n - no
        point_group = 'C1'
        orbspace = FullOrbitalSpace(1)
        orbspace.set_full(OrbitalSpace(dim=[n], orb_type='R'))
        orbspace.set_ref(OrbitalSpace(dim=[no], orb_type='R'))
        orbspace.set_froz(OrbitalSpace(dim=[0], orb_type='R'))
        wfcc = WF.from_zero_amplitudes(point_group, orbspace, level='SD')
        wfcc.update_amplitudes(np.array([0.44, 0.52, 0.12, 0.50, 0.59, 0.60, 0.01, 0.48, 0.14, 0.46, 0.20, 0.50, 0.67, 0.04, 0.10, 0.44, 0.04, 0.72, 0.40, 0.02, 0.18, 0.98, 0.87, 0.77, 0.38]))
        h_sym = np.asarray([0.30,0.55,0.58,0.44,0.77,0.86,0.84,0.74,0.21,0.56,0.46,0.79,0.73,0.77,0.13,0.45,0.70,0.22,0.69,0.43,0.04,1.00,0.55,0.48,0.80,0.21,0.85,0.37,0.48,0.26,0.15,0.03,0.37,0.71,0.69,0.38,0.72,0.72,0.35,0.10,0.30,0.20,0.95,0.06,0.84,0.94,0.21,0.53,0.23,0.77,0.16,0.42,0.37,0.91,0.98])
        h = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i < j:
                    ij = i + j*(j+1)//2
                else:
                    ij = j + i*(i+1)//2
                h[i,j] = h_sym[ij]
        
        ht1 = cc.t1_1e_transf_vv(h,wfcc)
        ht1_corr = [[-1.0030, -0.4058, 0.2330, -0.7168, -0.7254],[0.0500, -0.6333, -0.0010, 0.0954, -1.0997],[-0.6983, -0.7743, -0.2464, -0.9395, -1.0342],[-0.7389, -0.1279, -0.5193, 0.0942, -0.2955],[-0.6267, -0.5668, -0.1890, 0.0030, -0.0214]]
        for i in range(nv):
            for j in range(nv):
                self.assertAlmostEqual(ht1[i,j],ht1_corr[i][j])
    
    def test_cc_t1_transf_1e_id(self):
        """ Tests if there is errors in the matrices size"""
        n = 15
        no = 3
        nv = n - no
        point_group = 'C1'
        orbspace = FullOrbitalSpace(1)
        orbspace.set_full(OrbitalSpace(dim=[n], orb_type='R'))
        orbspace.set_ref(OrbitalSpace(dim=[no], orb_type='R'))
        orbspace.set_froz(OrbitalSpace(dim=[0], orb_type='R'))
        wfcc = WF.from_zero_amplitudes(point_group, orbspace, level='SD')
        wfcc.update_amplitudes(np.zeros(no*nv))
        h = np.random.rand(n,n)
        hoo = cc.t1_1e_transf_oo(h,wfcc)
        hvo = cc.t1_1e_transf_vo(h,wfcc)
        hvv = cc.t1_1e_transf_vv(h,wfcc)
        for i in range(n):
            for j in range(n):
                if i >= no:
                    if j >= no:
                        self.assertAlmostEqual(h[i,j],hvv[i-no,j-no])
                    else:
                        self.assertAlmostEqual(h[i,j],hvo[i-no,j])
                else:
                    if j < no:
                        self.assertAlmostEqual(h[i,j],hoo[i,j])

    def test_u_maker(self):
        #TODO: Generalize the symmetry
        n = 5
        no = 2
        nv = n - no
        point_group = 'C1'
        orbspace = FullOrbitalSpace(1)
        orbspace.set_full(OrbitalSpace(dim=[n], orb_type='R'))
        orbspace.set_ref(OrbitalSpace(dim=[no], orb_type='R'))
        orbspace.set_froz(OrbitalSpace(dim=[0], orb_type='R'))
        wfcc = WF.from_zero_amplitudes(point_group, orbspace, level='SD')
        wfcc.update_amplitudes(np.array([0.75, 0.79, 0.16, 0.87, 0.43, 0.14,              0.36, 0.40, 0.27, 0.90, 0.88, 0.44, 0.88, 0.09, 0.77, 0.01, 0.26, 0.77, 0.98, 0.25, 0.56, 0.91, 0.31, 0.23, 0.64, 0.68, 0.73, 0.62, 0.52, 0.91, 0.10, 0.65, 0.74]))
        u = coupled_cluster.make_u(wfcc,no,nv)
        u_corr = [0.36, -0.1, -0.34, 1.4, 0.88, 0.79, 1.49, -0.26, 0.77, 0.01, -0.46, 0.63, 1.7, 0.25, 0.81, 1.05, 0.06, 0.23, 0.64, 0.74, 1.36, 0.56, 0.52, 1.17, -0.53, 0.39, 0.74]
        for i in range(no*(no+1)*nv**2//2):
            self.assertAlmostEqual(u[i],u_corr[i])
    
    def test_F_maker(self):
        n = 2
        no = 1
        nv = n - no
        mol_int = integrals.Integrals(None,None,method=None,orth_method=None)
        mol_int.n_func = n
        mol_int.S = np.ndarray((2,2))
        mol_int.S[0,0] = 1
        mol_int.S[1,0] = 0
        mol_int.S[0,1] = 0
        mol_int.S[1,1] = 1
        mol_int.h = np.ndarray((2,2))
        mol_int.h[0,0] = -1.25281745
        mol_int.h[1,0] = 0
        mol_int.h[0,1] = 0
        mol_int.h[1,1] = -0.47549163
        mol_int.g = integrals.Two_Elec_Int()
        mol_int.g._format = 'ijkl'
        n_g = mol_int.n_func * (mol_int.n_func + 1) // 2
        n_g = n_g * (n_g + 1) // 2
        mol_int.g._integrals = np.zeros(n_g)
        mol_int.g._integrals[0] = 0.6745962516826632
        mol_int.g._integrals[1] = 0
        mol_int.g._integrals[2] = 0.18122366784286037
        mol_int.g._integrals[3] = 0.6636023327598741
        mol_int.g._integrals[4] = 0
        mol_int.g._integrals[5] = 0.6973849743389031
        F = coupled_cluster.make_F(mol_int.h,mol_int.g._integrals,no)
        F_corr = np.zeros((2,2))
        F_corr[0][0] = 0.5677597993595512
        F_corr[0][1] = 0
        F_corr[1][0] = 0
        F_corr[1][1] = 1.3679083699999999
        for i in range(no):
            for a in range(nv):
                self.assertAlmostEqual(F[i,a],F_corr[i][a])


    def test_cc_omega1(self):
        """ Tests the Omega1 calculation """
        n = 2
        no = 1
        nv = n - no
        point_group = 'C1'
        orbspace = FullOrbitalSpace(1)
        orbspace.set_full(OrbitalSpace(dim=[n], orb_type='R'))
        orbspace.set_ref(OrbitalSpace(dim=[no], orb_type='R'))
        orbspace.set_froz(OrbitalSpace(dim=[0], orb_type='R'))
        wfcc = WF.from_zero_amplitudes(point_group, orbspace, level='SD')
        wfcc.update_amplitudes(np.array([0.1, 0.01]))
        mol_int = integrals.Integrals(None,None,method=None,orth_method=None)
        mol_int.n_func = n
        mol_int.S = np.ndarray((2,2))
        mol_int.S[0,0] = 1
        mol_int.S[1,0] = 0
        mol_int.S[0,1] = 0
        mol_int.S[1,1] = 1
        mol_int.h = np.ndarray((2,2))
        mol_int.h[0,0] = -1.25281745
        mol_int.h[1,0] = 0
        mol_int.h[0,1] = 0
        mol_int.h[1,1] = -0.47549163
        mol_int.g = integrals.Two_Elec_Int()
        mol_int.g._format = 'ijkl'
        n_g = mol_int.n_func * (mol_int.n_func + 1) // 2
        n_g = n_g * (n_g + 1) // 2
        mol_int.g._integrals = np.zeros(n_g)
        mol_int.g._integrals[0] = 0.6745962516826632
        mol_int.g._integrals[1] = 0
        mol_int.g._integrals[2] = 0.18122366784286037
        mol_int.g._integrals[3] = 0.6636023327598741
        mol_int.g._integrals[4] = 0
        mol_int.g._integrals[5] = 0.6973849743389031
        u = coupled_cluster.make_u(wfcc,no,nv)
        F = coupled_cluster.make_F(mol_int.h,mol_int.g._integrals,no)
        omega1_corr = [0.124517652]
        self.assertAlmostEqual(coupled_cluster._res_t1_singles(wfcc,u,F,mol_int.g._integrals)[0],omega1_corr[0],4)

    def test_cc_omega2(self):
        """ Tests the Omega2 calculation """
        n = 2
        no = 1
        nv = n - no
        point_group = 'C1'
        orbspace = FullOrbitalSpace(1)
        orbspace.set_full(OrbitalSpace(dim=[n], orb_type='R'))
        orbspace.set_ref(OrbitalSpace(dim=[no], orb_type='R'))
        orbspace.set_froz(OrbitalSpace(dim=[0], orb_type='R'))
        wfcc = WF.from_zero_amplitudes(point_group, orbspace, level='SD')
        wfcc.update_amplitudes(np.array([0.1, 0.01]))
        mol_int = integrals.Integrals(None,None,method=None,orth_method=None)
        mol_int.n_func = n
        mol_int.S = np.ndarray((2,2))
        mol_int.S[0,0] = 1
        mol_int.S[1,0] = 0
        mol_int.S[0,1] = 0
        mol_int.S[1,1] = 1
        mol_int.h = np.ndarray((2,2))
        mol_int.h[0,0] = -1.25281745
        mol_int.h[1,0] = 0
        mol_int.h[0,1] = 0
        mol_int.h[1,1] = -0.47549163
        mol_int.g = integrals.Two_Elec_Int()
        mol_int.g._format = 'ijkl'
        n_g = mol_int.n_func * (mol_int.n_func + 1) // 2
        n_g = n_g * (n_g + 1) // 2
        mol_int.g._integrals = np.zeros(n_g)
        mol_int.g._integrals[0] = 0.6745962516826632
        mol_int.g._integrals[1] = 0
        mol_int.g._integrals[2] = 0.18122366784286037
        mol_int.g._integrals[3] = 0.6636023327598741
        mol_int.g._integrals[4] = 0
        mol_int.g._integrals[5] = 0.6973849743389031
        u = coupled_cluster.make_u(wfcc,no,nv)
        F = coupled_cluster.make_F(mol_int.h,mol_int.g._integrals,no)
        omega = [0]
        omega = coupled_cluster.t1_2e_transf_vovo(mol_int.g._integrals,wfcc)
        self.assertAlmostEqual(omega[0],0.17806508)
        omega = [0]
        coupled_cluster._res_doub_A2pp(omega,coupled_cluster.t1_2e_transf_vvvv(mol_int.g._integrals,wfcc),wfcc,no,nv)
        self.assertAlmostEqual(omega[0],0.0069919721)
        omega = [0]
        coupled_cluster._res_doub_B2(omega,coupled_cluster.t1_2e_transf_oooo(mol_int.g._integrals,wfcc),mol_int.g._integrals,wfcc,no,nv)
        self.assertAlmostEqual(omega[0],0.006782207,)
        omega = [0]
        coupled_cluster._res_doub_C2(omega,coupled_cluster.t1_2e_transf_oovv(mol_int.g._integrals,wfcc),mol_int.g._integrals,wfcc,no,nv)
        self.assertAlmostEqual(omega[0],-0.0198265)
        omega = [0]
        coupled_cluster._res_doub_D2(omega,coupled_cluster.make_L_t1_voov(coupled_cluster.t1_2e_transf_voov(mol_int.g._integrals,wfcc),mol_int.g._integrals,wfcc),coupled_cluster.make_L_ovov(mol_int.g._integrals,no,nv),coupled_cluster.make_u(wfcc,no,nv),no,nv)
        self.assertAlmostEqual(omega[0],-0.00302061)
        omega = [0]
        coupled_cluster._res_doub_E2(omega,coupled_cluster.t1_1e_transf_vv(F,wfcc),coupled_cluster.t1_1e_transf_oo(F,wfcc),mol_int.g._integrals,wfcc,coupled_cluster.make_u(wfcc,no,nv),no,nv)
        self.assertAlmostEqual(omega[0],0.024901,5)
        self.assertAlmostEqual(coupled_cluster.energy(wfcc,0,mol_int.g._integrals),0.003624474)


    def test_ccsd_H2(self):
        """ Test H2 calculation """
        n = 2
        no = 1
        nv = n - no
        point_group = 'C1'
        orbspace = FullOrbitalSpace(1)
        orbspace.set_full(OrbitalSpace(dim=[n], orb_type='R'))
        orbspace.set_ref(OrbitalSpace(dim=[no], orb_type='R'))
        orbspace.set_froz(OrbitalSpace(dim=[0], orb_type='R'))
        wfcc = WF.from_zero_amplitudes(point_group, orbspace, level='SD')
        mol_int = integrals.Integrals(None,None,method=None,orth_method=None)
        mol_int.n_func = n
        mol_int.S = np.ndarray((2,2))
        mol_int.S[0,0] = 1
        mol_int.S[1,0] = 0
        mol_int.S[0,1] = 0
        mol_int.S[1,1] = 1
        mol_int.h = np.ndarray((2,2))
        mol_int.h[0,0] = -1.25281745
        mol_int.h[1,0] = 0
        mol_int.h[0,1] = 0
        mol_int.h[1,1] = -0.47549163
        mol_int.g = integrals.Two_Elec_Int()
        mol_int.g._format = 'ijkl'
        n_g = mol_int.n_func * (mol_int.n_func + 1) // 2
        n_g = n_g * (n_g + 1) // 2
        mol_int.g._integrals = np.zeros(n_g)
        mol_int.g._integrals[0] = 0.6745962516826632
        mol_int.g._integrals[1] = 0
        mol_int.g._integrals[2] = 0.18122366784286037
        mol_int.g._integrals[3] = 0.6636023327598741
        mol_int.g._integrals[4] = 0
        mol_int.g._integrals[5] = 0.6973849743389031
#        wfcc.update_amplitudes(np.array([0, 8.8]))
        results = optimiser.cc_closed_shell(2*mol_int.h[0,0]+mol_int.g._integrals[0],mol_int,wf_ini=wfcc,max_inter=50)
        self.assertAlmostEqual(results.energy,-1.8516,4)
        print(results)

    def test_ccsd_size(self):
        """ """
        n = 10
        no = 4
        nv = n - no
        point_group = 'C1'
        orbspace = FullOrbitalSpace(1)
        orbspace.set_full(OrbitalSpace(dim=[n], orb_type='R'))
        orbspace.set_ref(OrbitalSpace(dim=[no], orb_type='R'))
        orbspace.set_froz(OrbitalSpace(dim=[0], orb_type='R'))
        wfcc = WF.from_zero_amplitudes(point_group, orbspace, level='SD')
        mol_int = integrals.Integrals(None,None,method=None,orth_method=None)
        mol_int.n_func = n
        mol_int.S = np.random.rand(n,n)*0.01
        mol_int.h = np.random.rand(n,n)*0.01
        mol_int.g = integrals.Two_Elec_Int()
        mol_int.g._format = 'ijkl'
        n_g = mol_int.n_func * (mol_int.n_func + 1) // 2
        n_g = n_g * (n_g + 1) // 2
        mol_int.g._integrals = np.random.rand(n_g)*0.01
        optimiser.cc_closed_shell(0,mol_int,wf_ini=wfcc,max_inter=3)

