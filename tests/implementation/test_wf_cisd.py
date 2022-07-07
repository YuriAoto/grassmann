"""Checks for wave_functions cisd

"""
import unittest

import numpy as np

import tests
from input_output.log import logtime


@tests.category('SHORT')
class CheckCsdInCCSDwf(unittest.TestCase):
    """Check for array manipulation in from_intNorm
    
    Checks if the array manipulation to get Csd is as it should be
    """
    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.n_corr = 50
        self.n_ext = 50
        self.doubles = np.random.RandomState(
            tests.init_random_state).random_sample(
            size=(self.n_ext, self.n_ext))
        self.singles = np.random.RandomState(
            tests.init_random_state).random_sample(
            size=(self.n_corr, self.n_ext))
        self.Csd1 = np.zeros((self.n_corr, self.n_ext,
                              self.n_corr, self.n_ext))
        self.Csd2 = np.zeros((self.n_corr, self.n_ext,
                              self.n_corr, self.n_ext))

    def test_check_D(self):
        with logtime('Check_D, loops'):
            for i in range(self.n_corr):
                for j in range(self.n_corr):
                    for a in range(self.n_ext):
                        for b in range(self.n_ext):
                            if a == b:
                                self.Csd1[i, a, j, a] += self.doubles[a, a]
                                if i != j:
                                    self.Csd1[j, a, i, a] = self.Csd1[i, a, j, a]
                            elif i == j:  # a != b
                                self.Csd1[i, a, i, b] += (
                                    self.doubles[a, b] + self.doubles[b, a]) / 2
                            else:  # a != b and i != j
                                self.Csd1[i, a, j, b] += self.doubles[a, b]
                                self.Csd1[j, a, i, b] += self.doubles[b, a]
        with logtime('Check_D, array'):
            for i in range(self.n_corr):
                for j in range(self.n_corr):
                    self.Csd2[i, :, j, :] += self.doubles[:, :]
                    self.Csd2[j, :, i, :] += self.doubles[:, :].T
                    if i == j:
                        self.Csd2[i, :, i, :] /= 2
        self.assertEqual(self.Csd1, self.Csd2)

    def test_check_S(self):
        with logtime('Check_S, loops'):
            for i in range(self.n_corr):
                for j in range(self.n_corr):
                    for a in range(self.n_ext):
                        for b in range(self.n_ext):
                            if a == b:
                                self.Csd1[i, a, j, a] += (
                                    self.singles[i, a] * self.singles[j, a])
                                if i != j:
                                    self.Csd1[j, a, i, a] = self.Csd1[i, a, j, a]
                            elif i == j:  # a != b
                                self.Csd1[i, a, i, b] += (
                                    self.singles[i, a] * self.singles[i, b])
                            else:  # a != b and i != j
                                self.Csd1[i, a, j, b] += (
                                    self.singles[i, a] * self.singles[j, b])
                                self.Csd1[j, a, i, b] += (
                                    self.singles[j, a] * self.singles[i, b])
        with logtime('Check_S, array (outer)'):
            for i in range(self.n_corr):
                for j in range(self.n_corr):
                    self.Csd2[i, :, j, :] += np.outer(self.singles[i, :],
                                                      self.singles[j, :])
                    self.Csd2[j, :, i, :] += np.outer(self.singles[j, :],
                                                      self.singles[i, :])
                    if i == j:
                        self.Csd2[i, :, i, :] /= 2
        self.assertEqual(self.Csd1, self.Csd2)
