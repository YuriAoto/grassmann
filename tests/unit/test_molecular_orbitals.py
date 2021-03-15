"""Tests for molecular orbitals

"""

import unittest

import numpy as np

import tests
from integrals.integrals import Integrals
from integrals.integrals import Two_Elec_Int
from orbitals.orbitals import MolecularOrbitals


class AtomicToMolecularIntegralsTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)
        self.prng = np.random.RandomState(tests.init_random_state)

    def test_1e_int(self):
        """H_2 one-electron integrals tranformation from atomic to molecular
           g values: Szabo p.162
        """
        atomic_int = Integrals(None,None,method=None,orth_method=None)
        atomic_int.n_func = 2
        atomic_int.S = np.ndarray((2,2))
        atomic_int.S[0,0] = 1
        atomic_int.S[1,0] = 0.6593
        atomic_int.S[0,1] = 0.6593
        atomic_int.S[1,1] = 1
        atomic_int.h = np.ndarray((2,2))
        atomic_int.h[0,0] = -1.1204
        atomic_int.h[1,0] = -0.9584
        atomic_int.h[0,1] = -0.9584
        atomic_int.h[1,1] = -1.1204
        atomic_int.g = Two_Elec_Int()
        atomic_int.g._format = 'ijkl'
        n_g = atomic_int.n_func * (atomic_int.n_func + 1) // 2
        n_g = n_g * (n_g + 1) // 2
        atomic_int.g._integrals = np.zeros(n_g)
        mol_orb = MolecularOrbitals()
        mol_orb._basis_len = atomic_int.n_func
        mol_orb.n_irrep  = 1
        mol_orb._coefficients = [np.ndarray((2,2))]
        mol_orb._coefficients[0][0,0] = 1/np.sqrt(2)
        mol_orb._coefficients[0][0,1] = 1/np.sqrt(2)
        mol_orb._coefficients[0][1,0] = 1/np.sqrt(2)
        mol_orb._coefficients[0][1,1] = -1/np.sqrt(2)
### S matrix
#        mol_orb._coefficients[0][0,0] =  1/np.sqrt(2*(1+0.6593))
#        mol_orb._coefficients[0][0,1] =  1/np.sqrt(2*(1-0.6593))
#        mol_orb._coefficients[0][1,0] =  1/np.sqrt(2*(1+0.6593))
#        mol_orb._coefficients[0][1,1] = -1/np.sqrt(2*(1-0.6593))
###
        new_int = Integrals.from_atomic_to_molecular(atomic_int, mol_orb)
#        print(new_int.h) 
#        print(new_int.S) 
        h_corr = [[-2.0788, 0],[0, -0.1620]]
        S_corr = [[1.6593, 0],[0, 0.3407]]
        for i in range(atomic_int.n_func):
            for j in range(atomic_int.n_func):
                self.assertAlmostEqual(new_int.h[i,j],h_corr[i][j])
                self.assertAlmostEqual(new_int.S[i,j],S_corr[i][j])
        


    def test_2e_int(self):
        """H_2 two-electron integrals tranformation from atomic to molecular
           g values: Szabo p.162
           C: matrix simplified
        """
        atomic_int = Integrals(None,None,method=None,orth_method=None)
        atomic_int.n_func = 2
        atomic_int.g = Two_Elec_Int()
        atomic_int.g._format = 'ijkl'
        n_g = atomic_int.n_func * (atomic_int.n_func + 1) // 2
        n_g = n_g * (n_g + 1) // 2
        atomic_int.g._integrals = np.zeros(n_g)
        atomic_int.g._integrals[0] = 0.7746
        atomic_int.g._integrals[1] = 0.4441
        atomic_int.g._integrals[2] = 0.2970
        atomic_int.g._integrals[3] = 0.5697
        atomic_int.g._integrals[4] = 0.4441
        atomic_int.g._integrals[5] = 0.7746
#        print(atomic_int.g)
        mol_orb = MolecularOrbitals()
        mol_orb._integrals = atomic_int
        mol_orb.n_irrep  = 1
        mol_orb._coefficients = [np.ndarray((2,2))]
        mol_orb._coefficients[0][0,0] = 1/np.sqrt(2)
        mol_orb._coefficients[0][0,1] = 1/np.sqrt(2)
        mol_orb._coefficients[0][1,0] = 1/np.sqrt(2)
        mol_orb._coefficients[0][1,1] = -1/np.sqrt(2)
#        mol_orb._coefficients[0][0,0] =  1/np.sqrt(2*(1+0.6593))
#        mol_orb._coefficients[0][0,1] =  1/np.sqrt(2*(1-0.6593))
#        mol_orb._coefficients[0][1,0] =  1/np.sqrt(2*(1+0.6593))
#        mol_orb._coefficients[0][1,1] = -1/np.sqrt(2*(1-0.6593))
        mol_int = Two_Elec_Int.from_2e_atomic_to_molecular(atomic_int,mol_orb)
        g_corr = [1.85735,0,0.10245,0.37515,0,0.08095]
        for i in range(len(g_corr)):
#            print(mol_int._integrals[i])
            self.assertAlmostEqual(mol_int._integrals[i],g_corr[i])

    def test_2e_int_2(self):
        """Random g matrix, two rotation operations leading to a C1 rotation"""
        atomic_int = Integrals(None,None,method=None,orth_method=None)
        atomic_int.n_func = 8
        atomic_int.g = Two_Elec_Int()
        atomic_int.g._format = 'ijkl'
        n_g = atomic_int.n_func * (atomic_int.n_func + 1) // 2
        n_g = n_g * (n_g + 1) // 2
        atomic_int.g._integrals = np.random.rand(n_g)
#        print(atomic_int.g._integrals)
        mol_orb = MolecularOrbitals()
        mol_orb._integrals = atomic_int
        mol_orb.n_irrep  = 1
        mol_orb._coefficients = [np.rot90(np.identity(atomic_int.n_func))]
#        print(mol_orb._coefficients)
        mol_int = Two_Elec_Int.from_2e_atomic_to_molecular(atomic_int,mol_orb) #This func must return an Integrals object
        #######Change This Later
        atomic_int_2 = Integrals(None,None,method=None,orth_method=None)
        atomic_int_2.n_func = atomic_int.n_func
        atomic_int_2.g = mol_int
        atomic_int_2.g._format = 'ijkl'
#        print(mol_int._integrals)
        #######
        mol_int_2 = Two_Elec_Int.from_2e_atomic_to_molecular(atomic_int_2,mol_orb)
#        print(mol_int_2._integrals)
        for i in range(n_g):
            self.assertAlmostEqual(mol_int_2._integrals[i],atomic_int.g._integrals[i])
        


    def test_mol_int(self):
        """H_2 two-electron integrals tranformation from atomic to molecular
           g values: Szabo p.162
           C: matrix simplified
        """
        atomic_int = Integrals(None,None,method=None,orth_method=None)
        atomic_int.n_func = 2
        atomic_int.S = np.ndarray((2,2))
        atomic_int.S[0,0] = 1
        atomic_int.S[1,0] = 0.6593
        atomic_int.S[0,1] = 0.6593
        atomic_int.S[1,1] = 1
        atomic_int.h = np.ndarray((2,2))
        atomic_int.h[0,0] = -1.1204
        atomic_int.h[1,0] = -0.9584
        atomic_int.h[0,1] = -0.9584
        atomic_int.h[1,1] = -1.1204
        atomic_int.g = Two_Elec_Int()
        atomic_int.g._format = 'ijkl'
        n_g = atomic_int.n_func * (atomic_int.n_func + 1) // 2
        n_g = n_g * (n_g + 1) // 2
        atomic_int.g._integrals = np.zeros(n_g)
        atomic_int.g._integrals[0] = 0.7746
        atomic_int.g._integrals[1] = 0.4441
        atomic_int.g._integrals[2] = 0.2970
        atomic_int.g._integrals[3] = 0.5697
        atomic_int.g._integrals[4] = 0.4441
        atomic_int.g._integrals[5] = 0.7746
        mol_orb = MolecularOrbitals()
        mol_orb._integrals = atomic_int
        mol_orb.n_irrep  = 1
        mol_orb._coefficients = [np.ndarray((2,2))]
        mol_orb._coefficients[0][0,0] = 1/np.sqrt(2)
        mol_orb._coefficients[0][0,1] = 1/np.sqrt(2)
        mol_orb._coefficients[0][1,0] = 1/np.sqrt(2)
        mol_orb._coefficients[0][1,1] = -1/np.sqrt(2)
        #mol_orb._coefficients[0][0,0] =  1/np.sqrt(2*(1+0.6593))
        #mol_orb._coefficients[0][0,1] =  1/np.sqrt(2*(1-0.6593))
        #mol_orb._coefficients[0][1,0] =  1/np.sqrt(2*(1+0.6593))
        #mol_orb._coefficients[0][1,1] = -1/np.sqrt(2*(1-0.6593))
        h_corr = [[-2.0788, 0],[0, -0.1620]]
        S_corr = [[1.6593, 0],[0, 0.3407]]
        g_corr = [1.85735,0,0.10245,0.37515,0,0.08095]
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(mol_orb.molecular_integrals.h[i,j],h_corr[i][j])
                self.assertAlmostEqual(mol_orb.molecular_integrals.S[i,j],S_corr[i][j])
        for i in range(len(g_corr)):
            self.assertAlmostEqual(mol_orb.molecular_integrals.g._integrals[i],g_corr[i])
