"""Tests for input parser

"""
import os
import sys
import unittest
from unittest.mock import patch

import tests
from hartree_fock.main import _define_hfstep_func


@tests.category('SHORT')
class InternalsTestCase(unittest.TestCase):

    def test_SCF(self):
        f = _define_hfstep_func('SCF')
        self.assertEqual(f(0, 0.0), 'RH-SCF')
        self.assertEqual(f(3, 4.0), 'RH-SCF')
        self.assertEqual(f(6, 2.0), 'RH-SCF')

    def test_Absil(self):
        f = _define_hfstep_func('Absil')
        self.assertEqual(f(0, 0.0), 'Absil')
        self.assertEqual(f(3, 4.0), 'Absil')
        self.assertEqual(f(6, 2.0), 'Absil')

    def test_SCF_Absil_n5(self):
        f = _define_hfstep_func('SCF-Absil_n5')
        self.assertEqual(f(0, 0.0), 'RH-SCF')
        self.assertEqual(f(3, 4.0), 'RH-SCF')
        self.assertEqual(f(6, 2.0), 'Absil')

    def test_SCF_Absil_n2(self):
        f = _define_hfstep_func('SCF-Absil_n2')
        self.assertEqual(f(0, 0.0), 'RH-SCF')
        self.assertEqual(f(3, 4.0), 'Absil')
        self.assertEqual(f(6, 2.0), 'Absil')

    def test_SCF_Absil_g1_0(self):
        f = _define_hfstep_func('SCF-Absil_grad1.0')
        self.assertEqual(f(0, 0.0), 'Absil')
        self.assertEqual(f(3, 4.0), 'RH-SCF')
        self.assertEqual(f(6, 2.0), 'RH-SCF')

    def test_SCF_Absil_g3_0(self):
        f = _define_hfstep_func('SCF-Absil_grad3.0')
        self.assertEqual(f(0, 0.0), 'Absil')
        self.assertEqual(f(3, 4.0), 'RH-SCF')
        self.assertEqual(f(6, 2.0), 'Absil')

    def test_raise(self):
        with self.assertRaises(ValueError):
            f = _define_hfstep_func('SCF-Absil_3.0')
        
