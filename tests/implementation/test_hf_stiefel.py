
"""Tests for input parser

"""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

import tests
from hartree_fock import stiefel
from hartree_fock.starting_orbitals import initial_orbitals
from hartree_fock import absil



def get_stiefel_element_from_molecule(molecule, geometry, basis):
    molsys = tests.get_test_system(molecule=molecule,
                                   geometry=geometry,
                                   basis=basis)
    S = molsys.integrals.S
    N = molsys.n_elec // 2
    C = initial_orbitals('Hcore', molsys, restricted=True,
                         conjugacy=None, step_size=None)[0][:, :N]
    st = stiefel.StiefelManifold(C.shape, S=S)
    return C, st, molsys.integrals


mymolecule = 'Li2'
mygeometry = '5'
mybasis = 'cc-pVTZ'


def _calc_hessian(C, st, tgsp_basis, integrals):
    P = 2 * C @ C.T
    blocks = absil.common_blocks(C, P, integrals.g._integrals)
    Fock = absil.restricted_fock(blocks[0],
                                 blocks[1],
                                 integrals.h,
                                 integrals.g._integrals)
    Fock = np.asarray(Fock)
    Euc_grad = 4 * Fock @ C
    Euc_Hess = absil.restricted_hessian(C,
                                        Fock,
                                        blocks[1],
                                        blocks[2],
                                        integrals.g._integrals)
    Hess_in_basis = st.true_Hessian_directly(C,
                                             Euc_grad,
                                             Euc_Hess,
                                             basis=tgsp_basis)
    return Hess_in_basis


def energy(C, integrals):
    P = 2 * C @ C.T
    blocks = absil.common_blocks(C, P, integrals.g._integrals)
    Fock = absil.restricted_fock(blocks[0],
                                 blocks[1],
                                 integrals.h,
                                 integrals.g._integrals)
    P_fock = np.tensordot(P, Fock)
    one_el_energy = np.tensordot(P, integrals.h)
    return (one_el_energy + P_fock) / 2

    
def num_sec_der(C, integrals, v, st, t=0.0001):
    C_p = st.geodesic(C, v, t)
    C_m = st.geodesic(C, -v, t)
    E_p = energy(C_p, integrals)
    E_m = energy(C_m, integrals)
    E = energy(C, integrals)
    return (E_p + E_m - 2*E) / (t**2)


@tests.category('SHORT')
class HessTestCase(unittest.TestCase):

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, tests.assert_arrays)

    def test_prj_tg_space(self):
        C, st, integrals = get_stiefel_element_from_molecule(molecule=mymolecule,
                                                             geometry=mygeometry,
                                                             basis=mybasis)

        tgsp_basis = st.basis_for_tg_space(C)

        H = _calc_hessian(C, st, tgsp_basis, integrals)
        for i, v in enumerate(tgsp_basis):
            vHv = H[i, i]
            d2dt2_E = num_sec_der(C, integrals, v, st)
            print(f'vHv =     {vHv:>10.6f}\n'
                  f'd2dt2_E = {d2dt2_E:>10.6f}\n'
                  '-------------------------')
            self.assertAlmostEqual(vHv, d2dt2_E, places=4)

        
