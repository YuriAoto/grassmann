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
        self.res.FCI__minD = 1.0
        self.res.FCI__vertCC = 1.0
        self.res.FCI__vertCI = 1.0
        self.res.CC__vertCC = 1.0
        self.res.CC__vertCC_ampl = 1.0
        self.res.CC__minD = 1.0
        self.res.CC__minD_ampl = 1.0
        self.res.minD__vertCC = 1.0
        self.res.minD__vertCC_ampl = 1.0

    def test_str(self):
        tests.logger.info("%s", self.res)
        x = str(self.res)
        self.assertFalse('regular CI' in x)
        self.assertFalse('regular CC' in x)
        
        self.res.has_cc = True
        self.res.fci__cc = 1.0
        tests.logger.info("%s", self.res)
        x = str(self.res)
        self.assertFalse('regular CI' in x)
        self.assertTrue('regular CC' in x)
        
        self.res.has_ci = True
        self.res.fci__ci = 1.0
        self.res.ci__cc = 1.0        
        tests.logger.info("%s", self.res)
        x = str(self.res)
        self.assertTrue('regular CI' in x)
        self.assertTrue('regular CC' in x)
        
        self.res.has_cc = False
        del self.res.fci__cc
        tests.logger.info("%s", self.res)
        x = str(self.res)
        self.assertTrue('regular CI' in x)
        self.assertFalse('regular CC' in x)
        self.assertTrue('CC manifold' in x)
        self.assertTrue('D(FCI, minD)' in x)
