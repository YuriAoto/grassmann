"""Tests for input parser

"""
import os
import sys
import unittest
from unittest.mock import patch

import test
from input_output import parser


class InternalsTestCase(unittest.TestCase):

    def setUp(self):
        self.filename = 'input_test'
        with open(self.filename, 'w') as f:
            f.write('key1 = value1\n')
            f.write('key2 = value2\n')
            f.write('key3 = value3\n')
            f.write('key4 = value4\n')
        self.filename2 = 'input_test2'
        with open(self.filename2, 'w') as f:
            f.write('key a = value a\n')
            f.write('key2 = value2\n')
            f.write('\n')
            f.write(self.filename + ' # a file name\n')
            f.write('key4 = new value 4\n')

    def tearDown(self):
        os.remove(self.filename)
        os.remove(self.filename2)

    def test_argvise_file(self):
        file_content = []
        args = parser._argvise_file(self.filename, file_content, '')
        self.assertEqual(args[0], '--key1')
        self.assertEqual(args[1], 'value1')
        self.assertEqual(args[2], '--key2')
        self.assertEqual(args[3], 'value2')
        self.assertEqual(args[4], '--key3')
        self.assertEqual(args[5], 'value3')
        self.assertEqual(args[6], '--key4')
        self.assertEqual(args[7], 'value4')

    def test_argvise_file2(self):
        file_content = []
        args = parser._argvise_file(self.filename2, file_content, '')
        self.assertEqual(args[0], '--key_a')
        self.assertEqual(args[1], 'value_a')
        self.assertEqual(args[2], '--key2')
        self.assertEqual(args[3], 'value2')
        self.assertEqual(args[4], '--key1')
        self.assertEqual(args[5], 'value1')
        self.assertEqual(args[6], '--key2')
        self.assertEqual(args[7], 'value2')
        self.assertEqual(args[8], '--key3')
        self.assertEqual(args[9], 'value3')
        self.assertEqual(args[10], '--key4')
        self.assertEqual(args[11], 'value4')
        self.assertEqual(args[12], '--key4')
        self.assertEqual(args[13], 'new_value_4')

    def test_argvise_sysargv(self):
        testargs = [__name__, '--key1', 'value1', self.filename, '--key7', 'value7']
        with patch.object(sys, 'argv', testargs):
            parser._parse_files_in_sys_argv()
            self.assertEqual(sys.argv[1], '--key1')
            self.assertEqual(sys.argv[2], 'value1')
            self.assertEqual(sys.argv[3], '--key1')
            self.assertEqual(sys.argv[4], 'value1')
            self.assertEqual(sys.argv[5], '--key2')
            self.assertEqual(sys.argv[6], 'value2')
            self.assertEqual(sys.argv[7], '--key3')
            self.assertEqual(sys.argv[8], 'value3')
            self.assertEqual(sys.argv[9], '--key4')
            self.assertEqual(sys.argv[10], 'value4')
            self.assertEqual(sys.argv[11], '--key7')
            self.assertEqual(sys.argv[12], 'value7')

    def test_argvise_sysargv2(self):
        testargs = [__name__, '--key1', 'value1', self.filename2, '--key7', 'value7']
        with patch.object(sys, 'argv', testargs):
            parser._parse_files_in_sys_argv()
            self.assertEqual(sys.argv[1], '--key1')
            self.assertEqual(sys.argv[2], 'value1')
            self.assertEqual(sys.argv[3], '--key_a')
            self.assertEqual(sys.argv[4], 'value_a')
            self.assertEqual(sys.argv[5], '--key2')
            self.assertEqual(sys.argv[6], 'value2')
            self.assertEqual(sys.argv[7], '--key1')
            self.assertEqual(sys.argv[8], 'value1')
            self.assertEqual(sys.argv[9], '--key2')
            self.assertEqual(sys.argv[10], 'value2')
            self.assertEqual(sys.argv[11], '--key3')
            self.assertEqual(sys.argv[12], 'value3')
            self.assertEqual(sys.argv[13], '--key4')
            self.assertEqual(sys.argv[14], 'value4')
            self.assertEqual(sys.argv[15], '--key4')
            self.assertEqual(sys.argv[16], 'new_value_4')
            self.assertEqual(sys.argv[17], '--key7')
            self.assertEqual(sys.argv[18], 'value7')

class ParserTestCase(unittest.TestCase):
    
    def setUp(self):
        self.filename = 'input_test'
        with open(self.filename, 'w') as f:
            f.write('memory = 20MB\n')
            f.write('basis = cc-pVTZ\n')
            f.write('geometry = filegeom\n')
        self.filename2 = 'input_test2'
        with open(self.filename2, 'w') as f:
            f.write('algorithm = general_Absil\n')
            f.write('\n')
            f.write(self.filename + ' # a file name\n')
            f.write('memory = 10GB\n')
    
    def test_parse_cmd1(self):
        testargs = [__name__, self.filename]
        with patch.object(sys, 'argv', testargs):
            args = parser.parse()
            self.assertEqual(args.memory, (float(20), 'MB'))
            self.assertEqual(args.basis, 'cc-pVTZ')
            self.assertEqual(args.geometry, 'filegeom')
    
    def test_parse_cmd2(self):
        testargs = [__name__, '--basis', '10kB', self.filename, '--basis', '6-31g']
        with patch.object(sys, 'argv', testargs):
            args = parser.parse()
            self.assertEqual(args.memory, (float(20), 'MB'))
            self.assertEqual(args.basis, '6-31g')
            self.assertEqual(args.geometry, 'filegeom')
    
    def test_parse_cmd3(self):
        testargs = [__name__, self.filename2]
        with patch.object(sys, 'argv', testargs):
            args = parser.parse()
            self.assertEqual(args.memory, (float(10), 'GB'))
            self.assertEqual(args.basis, 'cc-pVTZ')
            self.assertEqual(args.algorithm, 'general_Absil')
            self.assertEqual(args.geometry, 'filegeom')
