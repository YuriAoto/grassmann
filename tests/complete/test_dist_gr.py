"""Tests min dist to the Grassmannian

"""
import unittest

import numpy as np

import tests


@tests.category('COMPLETE')
class MinDistCISDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.base_cmd = ['--method', 'dist_Grassmann']
        self.H2 = 'H2__5__ccpVDZ__D2h'
        self.Li2 = 'Li2__5__ccpVDZ__D2h'
        self.Li2_sto3g = 'Li2__5__sto3g__D2h'
        self.N2_sto3g = 'N2__3__sto3g__D2h'

    def test_H2(self):
        arguments = self.base_cmd + ['--molpro_output',
                                     tests.CISD_file(self.H2)]
        to_check = [
            tests.CheckFloat(substr='|<minD|extWF>|',
                             position=2,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|<minE|minD>|',
                             position=2,
                             tol=1.0E-10)
        ]
        with tests.run_grassmann(arguments, to_check) as run_gr:
            self.assertEqual(run_gr, run_gr.reference())

    def test_Li2_sto3g(self):
        arguments = self.base_cmd + ['--save_final_orb',
                                     '--molpro_output',
                                     tests.CISD_file(self.Li2_sto3g)]
        to_check = [
            tests.CheckFloat(substr='|<minD|extWF>|',
                             position=2,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|<minE|minD>|',
                             position=2,
                             tol=1.0E-10)
        ]
        with tests.run_grassmann(arguments, to_check) as run_gr:
            self.assertEqual(run_gr, run_gr.reference(1))
            # U = np.load(run_gr.outdir + '/U_minD.npz')
            # U_ref = np.load(run_gr.reference(1) + '_U.npz')
            # for iU in U:
            #     print()
            #     print(U[iU])
            #     print(U_ref[iU])
            #     self.assertEqual(U[iU], U_ref[iU])
        
        arguments = self.base_cmd + ['--at_ref', '--molpro_output',
                                     tests.CISD_file(self.Li2_sto3g)]
        to_check = [
            tests.CheckFloat(substr='|J|',
                             position=4,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|ΔK|',
                             position=6,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|ΔK|',
                             position=2,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|<refWF|1st it>|',
                             position=3,
                             tol=1.0E-10),
            # tests.CheckFloat(substr='|<refWF|1st it>|',
            #                  position=3,
            #                  tol=1.0E-10)
        ]
        with tests.run_grassmann(arguments, to_check) as run_gr:
            self.assertEqual(run_gr, run_gr.reference(2))
        
        arguments = self.base_cmd + ['--molpro_output',
                                     tests.CCSD_file(self.Li2_sto3g)]
        to_check = [
            tests.CheckFloat(substr='|<minD|extWF>|',
                             position=2,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|<minE|minD>|',
                             position=2,
                             tol=1.0E-10)
        ]
        with tests.run_grassmann(arguments, to_check) as run_gr:
            self.assertEqual(run_gr, run_gr.reference(1))
        
        arguments = self.base_cmd + ['--at_ref', '--molpro_output',
                                     tests.CCSD_file(self.Li2_sto3g)]
        to_check = [
            tests.CheckFloat(substr='|J|',
                             position=4,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|ΔK|',
                             position=6,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|ΔK|',
                             position=2,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|<refWF|1st it>|',
                             position=3,
                             tol=1.0E-10),
            # tests.CheckFloat(substr='|<refWF|1st it>|',
            #                  position=3,
            #                  tol=1.0E-10)
        ]
        with tests.run_grassmann(arguments, to_check) as run_gr:
            self.assertEqual(run_gr, run_gr.reference(2))

    def test_Li2_ccpVDZ(self):
        arguments = self.base_cmd + ['--molpro_output',
                                     tests.CISD_file(self.Li2)]
        to_check = [
            tests.CheckFloat(substr='|<minD|extWF>|',
                             position=2,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|<minE|minD>|',
                             position=2,
                             tol=1.0E-10)
        ]
        with tests.run_grassmann(arguments, to_check) as run_gr:
            self.assertEqual(run_gr, run_gr.reference())
        
    def test_N2_sto3g(self):
        arguments = self.base_cmd + ['--molpro_output',
                                     tests.CCSD_file(self.N2_sto3g)]
        to_check = [
            tests.CheckFloat(substr='|<minD|extWF>|',
                             position=2,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|<minE|minD>|',
                             position=2,
                             tol=1.0E-10)
        ]
        with tests.run_grassmann(arguments, to_check) as run_gr:
            self.assertEqual(run_gr, run_gr.reference(1))
        
        arguments = self.base_cmd + ['--at_ref', '--molpro_output',
                                     tests.CCSD_file(self.N2_sto3g)]
        to_check = [
            tests.CheckFloat(substr='|J|',
                             position=4,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|ΔK|',
                             position=6,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|ΔK|',
                             position=2,
                             tol=1.0E-10),
            tests.CheckFloat(substr='|<refWF|1st it>|',
                             position=3,
                             tol=1.0E-10),
            # tests.CheckFloat(substr='|<refWF|1st it>|',
            #                  position=3,
            #                  tol=1.0E-10)
        ]
        with tests.run_grassmann(arguments, to_check) as run_gr:
            self.assertEqual(run_gr, run_gr.reference(2))
