"""Tests for fci

"""
import unittest

import numpy as np

from wave_functions import fci, general
from wave_functions.fci import make_occ
import test
from util import int_dtype

class ClusterDecTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)
        self.ref_det = fci.SlaterDet(
            c=0.0,
            alpha_occ=make_occ([0,1,2,3,4,5,6]),
            beta_occ=make_occ([0,1,2,3,4,5,6]))
    
    def test_cluster_decompose1(self):
        """t_{0 1}^{7 8}   (all in alpha)
        
        - t_0^7 t_1^8
        + t_0^8 t_1^7
        """
        dec = fci.cluster_decompose(
            (make_occ([0, 1]), make_occ([7, 8])),
            (make_occ([]), make_occ([])),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 2)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, make_occ([1,2,3,4,5,6,7]))
        self.assertEqual(dec[0][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][2].alpha_occ, make_occ([0,2,3,4,5,6,8]))
        self.assertEqual(dec[0][2].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[1][0], 1)
        self.assertEqual(dec[1][1].alpha_occ, make_occ([1,2,3,4,5,6,8]))
        self.assertEqual(dec[1][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[1][2].alpha_occ, make_occ([0,2,3,4,5,6,7]))
        self.assertEqual(dec[1][2].beta_occ, make_occ([0,1,2,3,4,5,6]))
    
    def test_cluster_decompose2(self):
        """t_{3 5}^{9 12}    (all in alpha)
        
        - t_3^9 t_5^12
        + t_3^12 t_5^9
        """
        dec = fci.cluster_decompose(
            (make_occ([3, 5]), make_occ([9, 12])),
            (make_occ([]), make_occ([])),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 2)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, make_occ([0,1,2,4,5,6,9]))
        self.assertEqual(dec[0][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][2].alpha_occ, make_occ([0,1,2,3,4,6,12]))
        self.assertEqual(dec[0][2].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[1][0], 1)
        self.assertEqual(dec[1][1].alpha_occ, make_occ([0,1,2,4,5,6,12]))
        self.assertEqual(dec[1][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[1][2].alpha_occ, make_occ([0,1,2,3,4,6,9]))
        self.assertEqual(dec[1][2].beta_occ, make_occ([0,1,2,3,4,5,6]))
    
    def test_cluster_decompose3(self):
        """t_{0 1 2 3}^{7 8 9 10}  (all in alpha, only D)
        
        - t_{0 1}^{7 8} t_{2 3}^{9 10}
        + t_{0 1}^{7 9} t_{2 3}^{8 10}
        + t_{0 2}^{7 8} t_{1 3}^{9 10}
        
        """
        dec = fci.cluster_decompose(
            (make_occ([0, 1, 2, 3]), make_occ([7, 8, 9, 10])),
            (make_occ([]), make_occ([])),
             self.ref_det)
        self.assertEqual(len(dec), 18)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, make_occ([2,3,4,5,6,7,8]))
        self.assertEqual(dec[0][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][2].alpha_occ, make_occ([0,1,4,5,6,9,10]))
        self.assertEqual(dec[0][2].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[1][0], 1)
        self.assertEqual(dec[1][1].alpha_occ, make_occ([2,3,4,5,6,7,9]))
        self.assertEqual(dec[1][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[1][2].alpha_occ, make_occ([0,1,4,5,6,8,10]))
        self.assertEqual(dec[1][2].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[6][0], 1)
        self.assertEqual(dec[6][1].alpha_occ, make_occ([1,3,4,5,6,7,8]))
        self.assertEqual(dec[6][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[6][2].alpha_occ, make_occ([0,2,4,5,6,9,10]))
        self.assertEqual(dec[6][2].beta_occ, make_occ([0,1,2,3,4,5,6]))
    
    def test_cluster_decompose4(self):
        """t_{0 1 2}^{7 8 9}  (all in alpha)
        
        - t_0^7 t_1^8 t_2^9
        - t_0^8 t_1^9 t_2^7
        - t_0^7 t_{2 3}^{8 9}
        + t_0^9 t_{0 2}^{7 8}
        """
        dec = fci.cluster_decompose(
            (make_occ([0, 1, 2]), make_occ([7, 8, 9])),
            (make_occ([]), make_occ([])),
            self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 15)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, make_occ([1,2,3,4,5,6,7]))
        self.assertEqual(dec[0][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][2].alpha_occ, make_occ([0,2,3,4,5,6,8]))
        self.assertEqual(dec[0][2].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][3].alpha_occ, make_occ([0,1,3,4,5,6,9]))
        self.assertEqual(dec[0][3].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[3][0], -1)
        self.assertEqual(dec[3][1].alpha_occ, make_occ([1,2,3,4,5,6,8]))
        self.assertEqual(dec[3][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[3][2].alpha_occ, make_occ([0,2,3,4,5,6,9]))
        self.assertEqual(dec[3][2].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[3][3].alpha_occ, make_occ([0,1,3,4,5,6,7]))
        self.assertEqual(dec[3][3].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[6][0], -1)
        self.assertEqual(dec[6][1].alpha_occ, make_occ([1,2,3,4,5,6,7]))
        self.assertEqual(dec[6][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[6][2].alpha_occ, make_occ([0,3,4,5,6,8,9]))
        self.assertEqual(dec[6][2].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[11][0], 1)
        self.assertEqual(dec[11][1].alpha_occ, make_occ([0,2,3,4,5,6,9]))
        self.assertEqual(dec[11][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[11][2].alpha_occ, make_occ([1,3,4,5,6,7,8]))
        self.assertEqual(dec[11][2].beta_occ, make_occ([0,1,2,3,4,5,6]))
    
    def test_cluster_decompose5(self):
        """t_{0a 0b}^{7a 7b}
        
        - t_0a^7a t_0b^7b
        """
        dec = fci.cluster_decompose(
            (make_occ([0]), make_occ([7])),
            (make_occ([0]), make_occ([7])),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 1)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, make_occ([1,2,3,4,5,6,7]))
        self.assertEqual(dec[0][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][2].alpha_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][2].beta_occ, make_occ([1,2,3,4,5,6,7]))
    
    def test_cluster_decompose6(self):
        """t_{0a 3b}^{9a 7b}
        
        - t_0a^9a t_3b^7b
        """
        dec = fci.cluster_decompose(
            (make_occ([0]), make_occ([9])),
            (make_occ([3]), make_occ([7])),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 1)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, make_occ([1,2,3,4,5,6,9]))
        self.assertEqual(dec[0][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][2].alpha_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][2].beta_occ, make_occ([0,1,2,4,5,6,7]))


    def test_cluster_decompose7(self):
        """t_{3 5}^{9 12}    (all in beta)
        
        - t_3^9 t_5^12
        + t_3^12 t_5^9
        """
        dec = fci.cluster_decompose(
            (make_occ([]), make_occ([])),
            (make_occ([3, 5]), make_occ([9, 12])),
             self.ref_det,
            mode='SD')
        self.assertEqual(len(dec), 2)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][1].beta_occ, make_occ([0,1,2,4,5,6,9]))
        self.assertEqual(dec[0][2].alpha_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][2].beta_occ, make_occ([0,1,2,3,4,6,12]))
        self.assertEqual(dec[1][0], 1)
        self.assertEqual(dec[1][1].alpha_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[1][1].beta_occ, make_occ([0,1,2,4,5,6,12]))
        self.assertEqual(dec[1][2].alpha_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[1][2].beta_occ, make_occ([0,1,2,3,4,6,9]))

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
        dec = fci.cluster_decompose(
            (make_occ([0, 1]), make_occ([7, 8])),
            (make_occ([2, 3]), make_occ([9, 10])),
             self.ref_det)
        self.assertEqual(len(dec), 9)
        self.assertEqual(dec[0][0], -1)
        self.assertEqual(dec[0][1].alpha_occ, make_occ([2,3,4,5,6,7,8]))
        self.assertEqual(dec[0][1].beta_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][2].alpha_occ, make_occ([0,1,2,3,4,5,6]))
        self.assertEqual(dec[0][2].beta_occ, make_occ([0,1,4,5,6,9,10]))
        self.assertEqual(dec[1][0], -1)
        self.assertEqual(dec[1][1].alpha_occ, make_occ([1,2,3,4,5,6,7]))
        self.assertEqual(dec[1][1].beta_occ, make_occ([0,1,3,4,5,6,9]))
        self.assertEqual(dec[1][2].alpha_occ, make_occ([0,2,3,4,5,6,8]))
        self.assertEqual(dec[1][2].beta_occ, make_occ([0,1,2,4,5,6,10]))
        self.assertEqual(dec[2][0], 1)
        self.assertEqual(dec[2][1].alpha_occ, make_occ([1,2,3,4,5,6,7]))
        self.assertEqual(dec[2][1].beta_occ, make_occ([0,1,3,4,5,6,10]))
        self.assertEqual(dec[2][2].alpha_occ, make_occ([0,2,3,4,5,6,8]))
        self.assertEqual(dec[2][2].beta_occ, make_occ([0,1,2,4,5,6,9]))
        self.assertEqual(dec[3][0], 1)
        self.assertEqual(dec[3][1].alpha_occ, make_occ([1,2,3,4,5,6,8]))
        self.assertEqual(dec[3][1].beta_occ, make_occ([0,1,3,4,5,6,9]))
        self.assertEqual(dec[3][2].alpha_occ, make_occ([0,2,3,4,5,6,7]))
        self.assertEqual(dec[3][2].beta_occ, make_occ([0,1,2,4,5,6,10]))
        self.assertEqual(dec[4][0], -1)
        self.assertEqual(dec[4][1].alpha_occ, make_occ([1,2,3,4,5,6,8]))
        self.assertEqual(dec[4][1].beta_occ, make_occ([0,1,3,4,5,6,10]))
        self.assertEqual(dec[4][2].alpha_occ, make_occ([0,2,3,4,5,6,7]))
        self.assertEqual(dec[4][2].beta_occ, make_occ([0,1,2,4,5,6,9]))
        self.assertEqual(dec[5][0], 1)
        self.assertEqual(dec[5][1].alpha_occ, make_occ([1,2,3,4,5,6,7]))
        self.assertEqual(dec[5][1].beta_occ, make_occ([0,1,2,4,5,6,9]))
        self.assertEqual(dec[5][2].alpha_occ, make_occ([0,2,3,4,5,6,8]))
        self.assertEqual(dec[5][2].beta_occ, make_occ([0,1,3,4,5,6,10]))
        self.assertEqual(dec[6][0], -1)
        self.assertEqual(dec[6][1].alpha_occ, make_occ([1,2,3,4,5,6,7]))
        self.assertEqual(dec[6][1].beta_occ, make_occ([0,1,2,4,5,6,10]))
        self.assertEqual(dec[6][2].alpha_occ, make_occ([0,2,3,4,5,6,8]))
        self.assertEqual(dec[6][2].beta_occ, make_occ([0,1,3,4,5,6,9]))
        self.assertEqual(dec[7][0], -1)
        self.assertEqual(dec[7][1].alpha_occ, make_occ([1,2,3,4,5,6,8]))
        self.assertEqual(dec[7][1].beta_occ, make_occ([0,1,2,4,5,6,9]))
        self.assertEqual(dec[7][2].alpha_occ, make_occ([0,2,3,4,5,6,7]))
        self.assertEqual(dec[7][2].beta_occ, make_occ([0,1,3,4,5,6,10]))
        self.assertEqual(dec[8][0], 1)
        self.assertEqual(dec[8][1].alpha_occ, make_occ([1,2,3,4,5,6,8]))
        self.assertEqual(dec[8][1].beta_occ, make_occ([0,1,2,4,5,6,10]))
        self.assertEqual(dec[8][2].alpha_occ, make_occ([0,2,3,4,5,6,7]))
        self.assertEqual(dec[8][2].beta_occ, make_occ([0,1,3,4,5,6,9]))



class SlaterDetTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)
        
    def test_simple_construction_and_namedtuple(self):
        det = fci.SlaterDet(c=0.123,
                            alpha_occ=np.array([0, 1, 2], dtype=int_dtype),
                            beta_occ=np.array([1], dtype=int_dtype))
        self.assertEqual(det.c, 0.123)
        self.assertEqual(det[0], 0.123)
        self.assertEqual(det.alpha_occ[1], 1)
        self.assertEqual(det[1][1], 1)
        self.assertEqual(det.beta_occ[0], 1)
        self.assertEqual(det[2][0], 1)
        with self.assertRaises(IndexError):
            det.beta_occ[1]

    def test_get_from_FCI_line_1(self):
        line = '    -0.162676901257  1  2  7  1  2  7'
        n_core = general.OrbitalsSets([0, 0, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, -0.162676901257, places=10)
        self.assertEqual(det.alpha_occ, make_occ([0, 1, 6]))
        self.assertEqual(det.beta_occ, make_occ([0, 1, 6]))

    def test_get_from_FCI_line_2(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = general.OrbitalsSets([0, 0, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, -0.049624632911, places=10)
        self.assertEqual(det.alpha_occ, make_occ([0, 1, 3]))
        self.assertEqual(det.beta_occ, make_occ([0, 1, 5]))

    def test_get_from_FCI_line_3(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = general.OrbitalsSets([0, 0, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core, zero_coefficient=True)
        self.assertAlmostEqual(det.c, 0.0, places=10)

    def test_get_from_FCI_line_4(self):
        line = '0.000000000000  1  2  9  1  2 10'
        n_core = general.OrbitalsSets([0, 0, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, 0.0, places=10)
        self.assertEqual(det.alpha_occ, make_occ([0, 1, 8]))
        self.assertEqual(det.beta_occ, make_occ([0, 1, 9]))

    def test_get_from_FCI_line_5(self):
        line = '    -0.162676901257  1  2  7  1  2  7'
        n_core = general.OrbitalsSets([1, 1, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, -0.162676901257, places=10)
        self.assertEqual(det.alpha_occ, make_occ([4]))
        self.assertEqual(det.beta_occ, make_occ([4]))

    def test_get_from_FCI_line_6(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = general.OrbitalsSets([1, 1, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, -0.049624632911, places=10)
        self.assertEqual(det.alpha_occ, make_occ([1]))
        self.assertEqual(det.beta_occ, make_occ([3]))

    def test_get_from_FCI_line_7(self):
        line = '    -0.049624632911  1  2  4  1  2  6'
        n_core = general.OrbitalsSets([1, 1, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core, zero_coefficient=True)
        self.assertAlmostEqual(det.c, 0.0, places=10)

    def test_get_from_FCI_line_8(self):
        line = '0.000000000000  1  2  9  1  2 10'
        n_core = general.OrbitalsSets([1, 1, 0, 0], occ_type='R')
        Ms = 0.0
        det = fci._get_slater_det_from_fci_line(
            line, Ms, n_core)
        self.assertAlmostEqual(det.c, 0.0, places=10)
        self.assertEqual(det.alpha_occ, make_occ([6]))
        self.assertEqual(det.beta_occ, make_occ([7]))


class ExcInfoTestCase(unittest.TestCase):
        
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, test.assert_arrays)
        self.wf = fci.WaveFunctionFCI.from_Molpro_FCI(
            'test/inputs_outputs/h2o__Req__sto3g__C2v/FCI_allE.out')

    def test_get_exc_1(self):
        rank, alpha_hp, beta_hp = self.wf.get_exc_info(
            fci.SlaterDet(c=0.0,
                            alpha_occ=make_occ([0,1,2,4,8]),
                            beta_occ=make_occ([0,1,2,4,5])))
        self.assertEqual(rank, 1)
        self.assertEqual(alpha_hp[0], make_occ([5]))
        self.assertEqual(alpha_hp[1], make_occ([8]))
        self.assertEqual(beta_hp[0], make_occ([]))
        self.assertEqual(beta_hp[1], make_occ([]))

    def test_get_exc_2(self):
        rank, alpha_hp, beta_hp = self.wf.get_exc_info(
            fci.SlaterDet(c=0.0,
                            alpha_occ=make_occ([0,1,2,4,8]),
                            beta_occ=make_occ([0,1,4,5,9])))
        self.assertEqual(rank, 2)
        self.assertEqual(alpha_hp[0], make_occ([5]))
        self.assertEqual(alpha_hp[1], make_occ([8]))
        self.assertEqual(beta_hp[0], make_occ([2]))
        self.assertEqual(beta_hp[1], make_occ([9]))

    def test_get_exc_3(self):
        rank, alpha_hp, beta_hp = self.wf.get_exc_info(
            fci.SlaterDet(c=0.0,
                            alpha_occ=make_occ([0,4,7,8,11]),
                            beta_occ=make_occ([0,1,4,5,9])))
        self.assertEqual(rank, 4)
        self.assertEqual(alpha_hp[0], make_occ([1,2,5]))
        self.assertEqual(alpha_hp[1], make_occ([7,8,11]))
        self.assertEqual(beta_hp[0], make_occ([2]))
        self.assertEqual(beta_hp[1], make_occ([9]))

        
