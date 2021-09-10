"""Tests min dist to the CC manifold

"""
import unittest

import numpy as np

import tests


def _set_args_test(level, base_cmd, system, allE=False, memory=None):
    arguments = base_cmd + ['--molpro_output',
                            tests.FCI_file(system, allE=allE)]
    if memory is not None:
        arguments.extend(['--memory', memory])
    to_check = [
        tests.CheckFloat(substr=f'D(FCI, CC{level} manifold)',
                         position=4,
                         tol=1.0E-10)
    ]
    return arguments, to_check



@tests.category('COMPLETE', 'SHORT')
class MinDistCCDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.base_cmd = ['--method', 'CCD_mani_minD']
        self.Be_sto3g_d2h = 'Be__at__sto3g__D2h'
        self.Li2_sto3g_c2v = 'Li2__5__sto3g__C2v'
        self.Li2_sto3g_d2h = 'Li2__5__sto3g__D2h'
        self.Li2_631g_d2h = 'Li2__5__631g__D2h'
        self.Li2_ccpvdz_d2h = 'Li2__5__ccpVDZ__D2h'

    def test_Be_sto3g_d2h(self):
        with tests.run_grassmann(
                *_set_args_test(level='D',
                                base_cmd=self.base_cmd,
                                system=self.Be_sto3g_d2h)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref1')

    def test_Be_sto3g_d2h_allE(self):
        with tests.run_grassmann(
                *_set_args_test(level='D',
                                base_cmd=self.base_cmd,
                                system=self.Be_sto3g_d2h,
                                allE=True)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref1')

    def test_Li2_sto3g_d2h(self):
        with tests.run_grassmann(
                *_set_args_test(level='D',
                                base_cmd=self.base_cmd,
                                system=self.Li2_sto3g_d2h)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref1')

    @tests.category('COMPLETE', 'LONG')
    def test_Li2_sto3g_d2h_allE(self):
        with tests.run_grassmann(
                *_set_args_test(level='D',
                                base_cmd=self.base_cmd,
                                system=self.Li2_sto3g_d2h,
                                allE=True,
                                memory='20.0MB')) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref1')

    def test_Li2_sto3g_c2v(self):
        with tests.run_grassmann(
                *_set_args_test(level='D',
                                base_cmd=self.base_cmd,
                                system=self.Li2_sto3g_c2v)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref1')

    def test_Li2_631g_d2h(self):
        with tests.run_grassmann(
                *_set_args_test(level='D',
                                base_cmd=self.base_cmd,
                                system=self.Li2_631g_d2h)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref1')

    @tests.category('COMPLETE', 'VERY LONG')
    def test_Li2_631g_d2h_allE(self):
        with tests.run_grassmann(
                *_set_args_test(level='D',
                                base_cmd=self.base_cmd,
                                system=self.Li2_631g_d2h,
                                allE=True,
                                memory='20.0MB')
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref1')

    def test_Li2_ccpvdz_d2h(self):
        with tests.run_grassmann(
                *_set_args_test(level='D',
                                base_cmd=self.base_cmd,
                                system=self.Li2_ccpvdz_d2h)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref1')


@tests.category('COMPLETE', 'SHORT')
class MinDistCCSDTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.base_cmd = ['--method', 'CCSD_mani_minD']
        self.Be_sto3g_d2h = 'Be__at__sto3g__D2h'
        self.Li2_sto3g_c2v = 'Li2__5__sto3g__C2v'
        self.Li2_sto3g_d2h = 'Li2__5__sto3g__D2h'
        self.Li2_631g_d2h = 'Li2__5__631g__D2h'
        self.Li2_ccpvdz_d2h = 'Li2__5__ccpVDZ__D2h'

    def test_Be_sto3g_d2h(self):
        with tests.run_grassmann(
                *_set_args_test(level='SD',
                                base_cmd=self.base_cmd,
                                system=self.Be_sto3g_d2h)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref2')

    def test_Be_sto3g_d2h_allE(self):
        with tests.run_grassmann(
                *_set_args_test(level='SD',
                                base_cmd=self.base_cmd,
                                system=self.Be_sto3g_d2h,
                                allE=True)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref2')

    def test_Li2_sto3g_d2h(self):
        with tests.run_grassmann(
                *_set_args_test(level='SD',
                                base_cmd=self.base_cmd,
                                system=self.Li2_sto3g_d2h)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref2')

    @tests.category('COMPLETE', 'LONG')
    def test_Li2_sto3g_d2h_allE(self):
        with tests.run_grassmann(
                *_set_args_test(level='SD',
                                base_cmd=self.base_cmd,
                                system=self.Li2_sto3g_d2h,
                                allE=True,
                                memory='20.0MB')) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref2')

    def test_Li2_sto3g_c2v(self):
        with tests.run_grassmann(
                *_set_args_test(level='SD',
                                base_cmd=self.base_cmd,
                                system=self.Li2_sto3g_c2v)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref2')

    def test_Li2_631g_d2h(self):
        with tests.run_grassmann(
                *_set_args_test(level='SD',
                                base_cmd=self.base_cmd,
                                system=self.Li2_631g_d2h)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref2')

    @tests.category('COMPLETE', 'VERY LONG')
    def test_Li2_631g_d2h_allE(self):
        with tests.run_grassmann(
                *_set_args_test(level='SD',
                                base_cmd=self.base_cmd,
                                system=self.Li2_631g_d2h,
                                allE=True,
                                memory='20.0MB')
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref2')

    def test_Li2_ccpvdz_d2h(self):
        with tests.run_grassmann(
                *_set_args_test(level='SD',
                                base_cmd=self.base_cmd,
                                system=self.Li2_ccpvdz_d2h)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref2')


@tests.category('COMPLETE', 'SHORT')
class MinDistCCSDDiagHessTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.base_cmd = ['--method', 'CCSD_mani_minD', '--cc_diag_hess']
        self.Be_sto3g_d2h = 'Be__at__sto3g__D2h'
        self.Li2_sto3g_c2v = 'Li2__5__sto3g__C2v'
        self.Li2_sto3g_d2h = 'Li2__5__sto3g__D2h'
        self.Li2_631g_d2h = 'Li2__5__631g__D2h'
        self.Li2_ccpvdz_d2h = 'Li2__5__ccpVDZ__D2h'

    def test_Li2_sto3g_d2h(self):
        with tests.run_grassmann(
                *_set_args_test(level='SD',
                                base_cmd=self.base_cmd,
                                system=self.Li2_sto3g_d2h)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref3')

    def test_Li2_631g_d2h(self):
        with tests.run_grassmann(
                *_set_args_test(level='SD',
                                base_cmd=self.base_cmd,
                                system=self.Li2_631g_d2h)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref3')

    def test_Li2_ccpvdz_d2h(self):
        with tests.run_grassmann(
                *_set_args_test(level='SD',
                                base_cmd=self.base_cmd,
                                system=self.Li2_ccpvdz_d2h)
        ) as run_gr:
            self.assertEqual(run_gr, run_gr.output + '_ref3')
