"""Tests for orbitals

"""
import unittest

import numpy as np

from dist_grassmann.metric import dist_from_ovlp


class OverlapTestCase(unittest.TestCase):

    def test_raises(self):
        with self.assertRaises(ValueError):
            dist_from_ovlp(0.0,
                           metric='Fubini-Studi')
        with self.assertRaises(ValueError):
            dist_from_ovlp(2.0,
                           metric='Fubini-Study')

    def test_tol(self):
        self.assertAlmostEqual(dist_from_ovlp(1.0 + 1E-9,
                                              metric='Fubini-Study'),
                               0.0)
        self.assertAlmostEqual(dist_from_ovlp(1.00001,
                                              metric='Fubini-Study',
                                              tol=0.0001),
                               0.0)
        with self.assertRaises(ValueError):
            self.assertAlmostEqual(dist_from_ovlp(1.00001,
                                                  metric='Fubini-Study'),
                                   0.0)

    def test_dist_from_ovlp_Fubini_Study(self):
        self.assertAlmostEqual(dist_from_ovlp(0.0,
                                              metric='Fubini-Study'),
                               np.pi/2)
        self.assertAlmostEqual(dist_from_ovlp(-0.5,
                                              metric='Fubini-Study'),
                               np.pi/3)
        self.assertAlmostEqual(dist_from_ovlp(1.0,
                                              metric='Fubini-Study'),
                               0.0)
        self.assertAlmostEqual(dist_from_ovlp(0.0,
                                              metric='Fubini-Study',
                                              norms=(3.0, 2.0)),
                               np.pi/2)
        self.assertAlmostEqual(dist_from_ovlp(1.5,
                                              metric='Fubini-Study',
                                              norms=(2.0, 1.5)),
                               np.pi/3)
        self.assertAlmostEqual(dist_from_ovlp(-3.0,
                                              metric='Fubini-Study',
                                              norms=(6.0, 0.5)),
                               0.0)

    def test_dist_from_ovlp_DAmico(self):
        self.assertAlmostEqual(dist_from_ovlp(0.0,
                                              metric='DAmico'),
                               np.sqrt(2.0))
        self.assertAlmostEqual(dist_from_ovlp(-0.5,
                                              metric='DAmico'),
                               1.0)
        self.assertAlmostEqual(dist_from_ovlp(1.0,
                                              metric='DAmico'),
                               0.0)
        self.assertAlmostEqual(dist_from_ovlp(0.0,
                                              metric='DAmico',
                                              norms=(3.0, 2.0)),
                               np.sqrt(2.0))
        self.assertAlmostEqual(dist_from_ovlp(1.5,
                                              metric='DAmico',
                                              norms=(2.0, 1.5)),
                               1.0)
        self.assertAlmostEqual(dist_from_ovlp(-3.0,
                                              metric='DAmico',
                                              norms=(6.0, 0.5)),
                               0.0)

    def test_dist_from_ovlp_BenavidesRiveros(self):
        self.assertAlmostEqual(dist_from_ovlp(0.0,
                                              metric='Benavides-Riveros'),
                               1.0)
        self.assertAlmostEqual(dist_from_ovlp(-0.5,
                                              metric='Benavides-Riveros'),
                               0.75)
        self.assertAlmostEqual(dist_from_ovlp(1.0,
                                              metric='Benavides-Riveros'),
                               0.0)
        self.assertAlmostEqual(dist_from_ovlp(0.0,
                                              metric='Benavides-Riveros',
                                              norms=(3.0, 2.0)),
                               1.0)
        self.assertAlmostEqual(dist_from_ovlp(1.5,
                                              metric='Benavides-Riveros',
                                              norms=(2.0, 1.5)),
                               0.75)
        self.assertAlmostEqual(dist_from_ovlp(-3.0,
                                              metric='Benavides-Riveros',
                                              norms=(6.0, 0.5)),
                               0.0)
