"""Tests for Grassmann


# Running tests:

At the top directory, run:

GR_TESTS_CATEG="<categories>" python3 -m unittest <test specification>

GR_TESTS_CATEG="<categories>" python3 setup.py test


Where <test specification> is a unittest specification, for exemple:

tests.integration.test_wf_cc_dist_to_fci
tests.integration.test_wf_cc_dist_to_fci.VertDistTwoElecCCDTestCase
tests.integration.test_wf_cc_dist_to_fci.VertDistTwoElecCCDTestCase.test_h2_sto3g_d2h

<categories> are the categories to be tested, see bellow


# Tests categories:

Each test should be in one or more categories, that can be selected by
the user with the environment variable GR_TESTS_CATEG. Possible categories,
and their meaninig are:

ALL
    Run all tests. It is equivalent to not setting GR_TESTS_CATEG,
    and it is intended for the user only. Do not add this category to
    any test.

NONE
    Do not run any test. Yt is intended for the user only.
    Do not add this category to any test.

VERY SHORT
SHORT
LONG
VERY LONG
    These categories identify how big how fast the test runs:
    VERY SHORT for tests that run in a fraction of a second,
    SHORT for tests that run in few seconds
    LONG for tests that run un some minutes
    VERY LONG for tests that take several minutes to run


ESSENTIAL
    Tests that are essential for Grassmann. Run these tests always
    before commiting. When making the tests for a new function,
    make sure to add at least few SHORT, or VERY SHORT, ESSENTIAL tests.

COMPLETE
    Complete runs of Grassmann. These tests tipically compare outputs


# Creating tests:

Tests are created based on the unittest module:

https://docs.python.org/3/library/unittest.html


For each test, add the decorator tests.category() with one or more of the
above categories. This decorator can be added at the class or method level.
Try to make ESSENTIAL tests also SHORT or VERY SHORT. Only if necessary,
use LONG, but never a VERY LONG

If you have to compare numpy arrays, add the function tests.assert_arrays to be
used for the comparison, in the setUp method:

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)


# Creating complete tests:

tests that run a complete call for Grassmann should use the class run_grassmann,
defined at run_util. See module documentation

# Standard test systems

In the directory tests/inputs_outputs there are several molpro outputs, that
can be used to test the functionalities of Grassmann. The module sys_util
contain several functions to handle finding these outputs. See this module
documentation.
"""
import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/'))

sys.path.insert(0, src_dir)

from .func_util import *
from .var_util import *
from .sys_util import *
from .run_util import *
