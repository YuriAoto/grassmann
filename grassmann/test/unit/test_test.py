"""Tests for the testing environment
"""
import unittest

import test


class TestsTestCase(unittest.TestCase):

    def test_get_inpout_file(self):
        self.assertTrue(test.sys_util._get_inpout_file(
            'Li2__5__to2s__C2v/CISD.out', only_check=True))
        self.assertFalse(test.sys_util._get_inpout_file(
            'Li2__5__to2s__C2v/CISDaaa.out', only_check=True))
        with self.assertRaises(FileNotFoundError):
            test.sys_util._get_inpout_file('Li2__5__to2s__C2v/CISDaaa.out')
    
    def test_get_file(self):
        self.assertEqual(test.CISD_file('Li2__5__to2s__C2v'),
                         test.main_files_dir + 'Li2__5__to2s__C2v/CISD.out')
        self.assertEqual(test.CCSD_file('Li2__5__to2s__C2v'),
                         test.main_files_dir + 'Li2__5__to2s__C2v/CCSD.out')
        self.assertEqual(test.FCI_file('Li2__5__to2s__C2v'),
                         test.main_files_dir + 'Li2__5__to2s__C2v/FCI.out')
        self.assertEqual(test.CISD_file('Li2__5__to2s__C2v', allE=True),
                         test.main_files_dir + 'Li2__5__to2s__C2v/CISD_allE.out')
        self.assertEqual(test.CCSD_file('Li2__5__to2s__C2v', allE=True),
                         test.main_files_dir + 'Li2__5__to2s__C2v/CCSD_allE.out')
        self.assertEqual(test.FCI_file('Li2__5__to2s__C2v', allE=True),
                         test.main_files_dir + 'Li2__5__to2s__C2v/FCI_allE.out')
    
    def test_get_file_noallE(self):
        self.assertEqual(test.RHF_file('Li2__5__to2s__C2v'),
                         test.main_files_dir + 'Li2__5__to2s__C2v/RHF.out')
        with self.assertRaises(TypeError):
            self.assertEqual(test.RHF_file('Li2__5__to2s__C2v', allE=True),
                             test.main_files_dir + 'Li2__5__to2s__C2v/RHF.out')

    
    def test_get_file_only_check(self):
        self.assertTrue(test.CISD_file('Li2__5__to2s__C2v', only_check=True))
        self.assertFalse(test.CISD_file('Li2__5__to2s__C2vaa', only_check=True))
        self.assertTrue(test.CCSD_file('Li2__5__to2s__C2v', only_check=True))
        self.assertFalse(test.CCSD_file('Li2__5__to2s__C2vaa', only_check=True))
        self.assertTrue(test.FCI_file('Li2__5__to2s__C2v', only_check=True))
        self.assertFalse(test.FCI_file('Li2__5__to2s__C2vaa', only_check=True))


class GeneratorTestCase(unittest.TestCase):

    def test_generator_1(self):
        for i, s in enumerate(test.test_systems()):
            with self.subTest(s=s):
                self.assertEqual(test.sys_util._all_test_systems[i], s)

    def test_generator_2(self):
        for i, s in enumerate(test.test_systems(basis=('sto3g', 'ccpVDZ'))):
            with self.subTest(s=s):
                self.assertTrue('sto3g' in  s or 'ccpVDZ' in s)

    def test_generator_3(self):
        for i, s in enumerate(test.test_systems(symmetry='C2v')):
            with self.subTest(s=s):
                self.assertTrue('C2v' in  s)

    def test_generator_4(self):
        for i, s in enumerate(test.test_systems(molecule='Li2')):
            with self.subTest(s=s):
                self.assertEqual(s[:3], 'Li2')

    def test_generator_5(self):
        for i, s in enumerate(test.test_systems(has_method='FCI')):
            with self.subTest(s=s):
                self.assertTrue(test.FCI_file(s, only_check=True))

    def test_generator_6(self):
        for i, s in enumerate(test.test_systems(has_method=('CISD',
                                                            'CISD_allE'))):
            with self.subTest(s=s):
                self.assertTrue(test.CISD_file(s, only_check=True))
                self.assertTrue(test.CISD_file(s, allE=True, only_check=True))

    def test_generator_7(self):
        with self.assertRaises(ValueError):
            for i, s in enumerate(test.test_systems(has_method='mmmmm')):
                pass
