"""Classes to help handling tests to compare full runs of Grassmann


Classes with checks:

Check(substr=SUBSTR,
      convert=FUNC_CONV,
      compare=FUNC_COMP]

SUBSTR is a string
FUNC_CONV is a function that receives one argument: a line
FUNC_COMP is a function that receives two arguments: values to compare

Runs the files until substring SUBSTR is found.
At these lines, applies FUNC_CONV at both lines and
compare them with FUNC_COMP: the test will fail if this returns
False:

FUNC_COMP(FUNC_CONV(line), FUNC_CONV(line_ref))


Some wrappers:

-----

CheckFloat(substr=SUBSTR,
           position=POS,
           tol=TOL)

same as:

Check(substr=SUBSTR,
      convert=lambda line: float(line.split()[POS]),
      compare=lambda x, xref: abs(x - xref) < TOL)

-----

CheckInt(substr=SUBSTR,
         position=POS)

same as:

Check(substr=SUBSTR,
      convert=lambda line: int(line.split()[POS]),
      compare=lambda x, xref: x == xref)

-----

CheckLine(substr=SUBSTR)

same as:

Check(substr=SUBSTR,
      convert=lambda line: line,
      compare=lambda x, xref: x == xref)



Examples:
--------
Run dist_Grassmann for H2.out, and compare the output, with the file
H2.gr_ref. Checks if the floats, at the thirs entry, of the lines with
'|<minD|extWF>|' and '|<minE|minD>|' are the same within 1.0E-10

args = ['--method', 'dist_Grassmann', '--molpro_output', 'H2.out']
to_check = [Check(substr='|<minD|extWF>|',
                  convert=lambda line : float(line.split()[2]),
                  compare=lambda x, xref : abs(x - xref) < 1.0E-10),
            CheckFloat(substr='|<minE|minD>|',
                       position=2,
                       tol=1.0E-10)]
with tests.run_grassmann(args, tocheck) as out:
    self.assertEqual(out, out.output + '_ref')

-----

Run Grassmann and compares if an int at the second position and a float at
the fourth position (with tolerance of 1.0E-5) of the *second* line that
has the substring 'Energy' match (this is just a hypothetical example)

args = ....
def f_conv(line):
    lspl = line.split()
    return int(lspl[1]), float(lspl[3])
def f_comp(x, xref):
    return x[0] == xref[0] and abs(x[1] - x[1]) < 1.0E-5
to_check = [Check(substr='Energy', # This first is just to skip first 'Energy'
                  convert=lambda line: line,
                  compare=lambda x, xref: True),
            Check(substr='Energy',
                  convert=f_conv,
                  compare=f_comp)]
with tests.run_grassmann(args, tocheck) as out:
    self.assertEqual(out, out.output + '_ref')


"""
import os
import shutil
import unittest
from collections import namedtuple
from input_output.parser import parse

grassmann_exe = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../src/Grassmann'))

Check = namedtuple('Check',
                   ['substr', 'convert', 'compare'])

CheckFloat = namedtuple('CheckFloat',
                        ['substr', 'position', 'tol'])

CheckInt = namedtuple('CheckInt',
                      ['substr', 'position'])

CheckStr = namedtuple('CheckStr',
                      ['substr', 'position'])

CheckLine = namedtuple('CheckLine',
                       ['substr'])


class run_grassmann:
    """Run Grassmann with the args as in the list args
    
    This class works as context manager to compare two files,
    one that has been generated by Grassmann and another that is the reference.
    """
    
    def __init__(self, args, to_check, removefiles=True):
        """Initialises
        
        Parameters:
        -----------
        args (list of str)
            Arguments to run Grassmann
        
        to_check (list of Check, CheckFloat, etc)
            A list of things to be checked. These should be instances
            of of the namedtuples Check, CheckFloat, CheckInt, or CheckStr,
            or CheckLine. The entries in this list should be in the order
            that these information occur at the output.
        
        removefiles (bool, optional, default=True)
            If True, remove all files generated by Grassmann at the end of
            the context management
        
        """
        self.args = args
        parsed = parse(args)
        self.output = parsed.output
        self.out_extension = parsed.out_extension
        self.logfile = parsed.logfile
        self.outdir = parsed.outdir
        self.f = None
        self.to_check = to_check
        self.removefiles = removefiles
    
    def __enter__(self):
        os.system(grassmann_exe + ' ' + ' '.join(self.args))
        self.f = open(self.output, 'r')
        return self
    
    def __eq__(self, ref_filename):
        with open(ref_filename) as ref_f:
            for check in self.to_check:
                found_substr = False
                found_substr_ref = False
                for line in self.f:
                    if check.substr in line:
                        found_substr = True
                        break
                if not found_substr:
                    raise unittest.TestCase.failureException(
                        f'Substring not found at generated file:\n'
                        + check.substr
                        + f'\n Arguments were {self.args}')
                for line_ref in ref_f:
                    if check.substr in line_ref:
                        found_substr_ref = True
                        break
                if not found_substr_ref:
                    raise unittest.TestCase.failureException(
                        f'Substring not found at reference file {ref_filename}:\n'
                        + check.substr) 
                if isinstance(check, Check):
                    failed = not check.compare(check.convert(line),
                                               check.convert(line_ref))
                else:
                    if isinstance(check, CheckLine):
                        failed = line != line_ref
                    else:
                        x = line.split()[check.position]
                        x_ref = line_ref.split()[check.position]
                        if isinstance(check, CheckFloat):
                            failed = abs(float(x) - float(x_ref)) > check.tol
                        elif isinstance(check, CheckInt):
                            failed = int(x) != int(x_ref)
                        elif isinstance(check, CheckStr):
                            failed = x != x_ref
                if failed:
                    raise unittest.TestCase.failureException(
                        'Comparison failed:\n'
                        + f'{line.strip()} -> obtained\n'
                        + f'{line_ref.strip()} -> at reference')
        return True
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.f.close()
        if self.removefiles:
            try:
                os.remove(self.output)
            except OSError:
                pass
            try:
                os.remove(self.logfile)
            except OSError:
                pass
            try:
                shutil.rmtree(self.outdir)
            except OSError:
                pass

    def reference(self, n=''):
        """The reference file to compare the output
        
        The file is obtained from the output file, replacing its extension
        by f"ref{n}"
        
        Parameters:
        -----------
        n (str-able, usually str or int. Optional, default = '')
            The identifier of the reference file, in case of multiple files
            for the same output.
        """
        return self.output.replace(self.out_extension, f'.ref{n}')