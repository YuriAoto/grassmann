"""Tests for integrals (the interface to Knizia's ir-wmme)

"""
import os
import unittest

import tests
from integrals.integrals import (_check_basis, _fetch_from_basis_set_exchange,
                                 _write_basis, _from_molpro_to_wmme,
                                 basis_file, BasisSetError, BasisInfo)
from molecular_geometry.molecular_geometry import MolecularGeometry, Atom


@tests.category('SHORT')
class FetchTestCase(unittest.TestCase):

    def setUp(self):
        self.wmme_dir = os.environ['GR_IR_WMME_DIR']

    def test_is_basis_there_1(self):
        atoms = [1, 2]
        filename = os.path.join(self.wmme_dir, 'bases/emsl_cc-pVDZ.libmol')
        _check_basis(filename, atoms)
        self.assertEqual(atoms, [])
        atoms = [1, 3]
        _check_basis(filename, atoms)
        self.assertEqual(atoms, [])

    def test_is_basis_there_2(self):
        atoms = [1, 2]
        filename = os.path.join(self.wmme_dir, 'bases/emsl_cc-pVTZ.libmol')
        _check_basis(filename, atoms)
        self.assertEqual(atoms, [])

    def test_is_basis_there_3(self):
        atoms = [1, 2]
        filename = os.path.join(self.wmme_dir, 'bases/emsl_cc-pVTZ.libmolaaaa')
        _check_basis(filename, atoms)
        self.assertEqual(atoms, [1, 2])

    def test_is_basis_there_4(self):
        atoms = [1, 2, 6, 10, 34, 100]
        filename = os.path.join(self.wmme_dir, 'bases/emsl_cc-pVTZ.libmol')
        _check_basis(filename, atoms)
        self.assertEqual(atoms, [100])

    def test_fetch_1(self):
        atoms = [1, 2]
        info = _fetch_from_basis_set_exchange('UGBS', atoms)
        self.assertEqual(len(info), 2)
        self.assertEqual(info[0][0], 'UGBS')
        self.assertEqual(info[0][1], 1)
        self.assertIn('13739.0854166734', info[0][2])
        self.assertEqual(info[1][0], 'UGBS')
        self.assertEqual(info[1][1], 2)
        self.assertIn('52680.4659115022', info[1][2])

    def test_fetch_2(self):
        atoms = [110]
        with self.assertRaises(BasisSetError) as bserror:
            _fetch_from_basis_set_exchange('UGBS', atoms)
        self.assertIn('Perhaps they do not have', str(bserror.exception))


@tests.category('SHORT')
class WriteBasisFileTestCase(unittest.TestCase):

    def setUp(self):
        self.wmme_dir = os.environ['GR_IR_WMME_DIR']
    
    def test_write_1(self):
        info = [BasisInfo(name='asdfg', atom=2,
                          basis=('! helium (a) -> [c]\n'
                                 +'The basis asdfg for 2\nThe basis asdfg for 2\n')),
                BasisInfo(name='asdfg', atom=1,
                          basis=('! hydrogen (a) -> [c]\n'
                                 +'The basis asdfg for 1\nThe basis asdfg for 1\n')),
                BasisInfo(name='asdfg', atom=6,
                          basis=('! carbon (a) -> [c]\n'
                                 +'The basis asdfg for 6\nThe basis asdfg for 6\n'))
        ]
        _write_basis(info, self.wmme_dir)
        info = [BasisInfo(name='asdfg', atom=3,
                          basis=('! lithium (a) -> [c]\n'
                                 +'The basis asdfg for 3\nThe basis asdfg for 3\n'))
        ]
        _write_basis(info, self.wmme_dir)
        self.assertTrue(os.path.isfile(os.path.join(self.wmme_dir, f'bases/emsl_asdfg.libmol')))
        with open(os.path.join(self.wmme_dir, f'bases/emsl_asdfg.libmol'), 'r') as f:
            all_lines = f.readlines()
        self.assertEqual(''.join(all_lines),
                        '! hydrogen (a) -> [c]\n'
                        + 'The basis asdfg for 1\nThe basis asdfg for 1\n'
                        + '! helium (a) -> [c]\n'
                        + 'The basis asdfg for 2\nThe basis asdfg for 2\n'
                        + '! lithium (a) -> [c]\n'
                        + 'The basis asdfg for 3\nThe basis asdfg for 3\n'
                        + '! carbon (a) -> [c]\n'
                        + 'The basis asdfg for 6\nThe basis asdfg for 6\n')
        os.remove(os.path.join(self.wmme_dir, f'bases/emsl_asdfg.libmol'))

@tests.category('SHORT')
class BasisFileTestCase(unittest.TestCase):

    def setUp(self):
        self.wmme_dir = os.environ['GR_IR_WMME_DIR']
        self.mol_geom = MolecularGeometry()
        self.mol_geom.add_atom(Atom('H', [0.0, 0.0, 0.0]))
        self.mol_geom.add_atom(Atom('H', [1.0, 0.0, 0.0]))
        self.mol_geom.add_atom(Atom('O', [2.0, 0.0, 0.0]))
        self.mol_geom.add_atom(Atom('F', [3.0, 0.0, 0.0]))
        self.mol_geom.add_atom(Atom('N', [4.0, 0.0, 0.0]))

    def test_is_basis_there_1(self):
        basis = 'cc-pVDZ'
        f = basis_file(basis, self.mol_geom, self.wmme_dir, try_getting_it=False)
        self.assertEqual(f, os.path.join(self.wmme_dir, 'bases/emsl_' + basis + '.libmol'))

    def test_is_basis_there_2(self):
        basis = 'cc-pVTZ'
        f = basis_file(basis, self.mol_geom, self.wmme_dir, try_getting_it=False)
        self.assertEqual(f, os.path.join(self.wmme_dir, 'bases/emsl_' + basis + '.libmol'))

    def test_is_basis_there_3(self):
        basis = 'cc-pVTZ'
        self.mol_geom.add_atom(Atom('Og', [100.0, 0.0, 0.0]))
        with self.assertRaises(BasisSetError) as bserror:
            basis_file(basis, self.mol_geom, self.wmme_dir, try_getting_it=False)
        self.assertIn('We do not have the basis', str(bserror.exception))

    def test_is_basis_there_4(self):
        basis = 'cc-pVTZaaa'
        with self.assertRaises(BasisSetError) as bserror:
            basis_file(basis, self.mol_geom, self.wmme_dir, try_getting_it=False)
        self.assertIn('We do not have the basis', str(bserror.exception))



@tests.category('SHORT')
class FromMolprotoWMME(unittest.TestCase):

    def test_molpro_1(self):
        with open(os.path.join(tests.main_files_dir, 'ccpVDZ_Na_molpro'), 'r') as f:
            basis_molpro = ''.join(f.readlines())
        with open(os.path.join(tests.main_files_dir, 'ccpVDZ_Na_wmme'), 'r') as f:
            basis_wmme = ''.join(f.readlines())
        self.assertEqual(_from_molpro_to_wmme(basis_molpro).replace(' ', '').replace('\n', ''),
                         basis_wmme.replace(' ', '').replace('\n', ''))

    @unittest.skip('Conversion from JSON not implemented')
    def test_json_1(self):
        with open(os.path.join(tests.main_files_dir, 'ccpVDZ_Na_json'), 'r') as f:
            basis_json = ''.join(f.readlines())
        with open(os.path.join(tests.main_files_dir, 'ccpVDZ_Na_wmme'), 'r') as f:
            basis_wmme = ''.join(f.readlines())
        self.assertEqual(_from_json_to_wmme(basis_josn).replace(' ', '').replace('\n', ''),
                         basis_wmme.replace(' ', '').replace('\n', ''))

    def test_molpro_2(self):
        with open(os.path.join(tests.main_files_dir, '6311ppGst_Fe_molpro'), 'r') as f:
            basis_molpro = ''.join(f.readlines())
        with open(os.path.join(tests.main_files_dir, '6311ppGst_Fe_wmme'), 'r') as f:
            basis_wmme = ''.join(f.readlines())
        self.assertEqual(_from_molpro_to_wmme(basis_molpro).replace(' ', '').replace('\n', ''),
                         basis_wmme.replace(' ', '').replace('\n', ''))

    @unittest.skip('Conversion from JSON not implemented')
    def test_json_1(self):
        with open(os.path.join(tests.main_files_dir, '6311ppGst_Fe_json'), 'r') as f:
            basis_json = ''.join(f.readlines())
        with open(os.path.join(tests.main_files_dir, '6311ppGst_Fe_wmme'), 'r') as f:
            basis_wmme = ''.join(f.readlines())
        self.assertEqual(_from_json_to_wmme(basis_josn).replace(' ', '').replace('\n', ''),
                         basis_wmme.replace(' ', '').replace('\n', ''))

