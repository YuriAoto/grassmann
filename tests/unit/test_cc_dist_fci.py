"""Integration tests for wave_functions.fci

"""
import unittest

import numpy as np

import coupled_cluster.dist_to_fci as dfci
import tests


@tests.category('SHORT')
class CalcAllDistTestCase(unittest.TestCase):

    def setUp(self):
        self.res = dfci.AllDistResults("Test of distances")
        self.res.fci__min_d = 1.0
        self.res.fci__vert = 1.0
        self.res.fci__vert_ci = 1.0
        self.res.cc__vert = 1.0
        self.res.cc__vert_ampl = 1.0
        self.res.cc__min_d = 1.0
        self.res.cc__min_d_ampl = 1.0
        self.res.vert__min_d = 1.0
        self.res.vert__min_d_ampl = 1.0

    def test_str(self):
        tests.logger.info("%s", self.res)
        x = str(self.res)
        self.assertFalse('regular CI' in x)
        self.assertFalse('regular CC' in x)
        
        self.res.fci__cc = 1.0
        tests.logger.info("%s", self.res)
        x = str(self.res)
        self.assertFalse('regular CI' in x)
        self.assertTrue('regular CC' in x)
        
        self.res.fci__ci = 1.0
        tests.logger.info("%s", self.res)
        x = str(self.res)
        self.assertTrue('regular CI' in x)
        self.assertTrue('regular CC' in x)
        
        del self.res.fci__cc
        tests.logger.info("%s", self.res)
        x = str(self.res)
        self.assertTrue('regular CI' in x)
        self.assertFalse('regular CC' in x)
        
        self.assertTrue('CC manifold' in x)
        self.assertTrue('D(FCI, minD CC) = 1.00000' in x)
