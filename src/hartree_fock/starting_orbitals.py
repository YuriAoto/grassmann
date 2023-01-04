"""Create Initial orbitals for Hartree-Fock


"""
import sys
import logging

import numpy as np

from molecular_geometry.periodic_table import ATOMS
from molecular_geometry.molecular_geometry import MolecularGeometry
from orbitals.orbitals import MolecularOrbitals
from hartree_fock import main


logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()


def initial_orbitals(ini_orb, molecular_system, restricted, conjugacy, step_size):
    """Initial orbitals
    
    
    Parameters:
    -----------
    ini_orb (str)
        Method to obtain initial orbitals. Possible values are:
        - 'Hcore'      eigenvectors of one electron operator
        - 'SAD'        superposition of atomic densities
        - <filename>   get orbitals from file (see MolecularOrbitals.from_file)
    
    molecular_system (MolecularGeometry)
        The system
    
    restricted (bool)
        True to obtain restricted orbitals

    conjugacy and step_size are a hack to get the SAD working. find a better way.
    
    Return:
    -------
    MolecularOrbitals, with the orbitals
    
    """
    logmsg = 'Starting guess for orbitals'
    if ini_orb == 'Hcore':
        logger.info(f'{logmsg}: eigenvectors of orthogonalised h')
        return MolecularOrbitals.from_eig_h(
            molecular_system.integrals,
            molecular_system.integrals.basis_set + '(AO)',
            restricted=restricted)
    if ini_orb == 'SAD':
        logger.info(f'{logmsg}: superposition of atomic densities')
        return MolecularOrbitals.from_dens(superpos_atdens(molecular_system,
                                                           conjugacy,
                                                           step_size),
                                           restricted,
                                           molecular_system.integrals)
    logger.info(f'{logmsg}: from file {ini_orb}')
    orb = MolecularOrbitals.from_file(ini_orb)
    if not args.restricted:
        orb = MolecularOrbitals.unrestrict(orb)
    if orb.restricted and not restricted:
        raise ValueError('Initial orbitals should be of unrestricted type.')
    orb.orthogonalise(X=molecular_system.integrals.X)


def superpos_atdens(molecular_system, conjugacy, step_size):
    """Generate superposition of atomic densities
    
    Parameters:
    -----------
    molecular_system (MolecularGeometry)
        The system

    conjugacy and step_size are a hack to get the SAD working. find a better way.
    
    Return:
    -------
    2-tuple of np.arrays.
    Generated molecular alpha and beta "densities",
    These are not true densities, as they are probably not idempotent.
    """
    atomic_dens = {}
    for at in molecular_system:
        atbas = at.element + at.basis
        if atbas in atomic_dens:
            continue
        atomic_dens[atbas] = calc_at_dens(at.element, at.basis, conjugacy, step_size)
    n = molecular_system.integrals.n_func
    mol_dens_a = np.zeros((n, n))
    mol_dens_b = np.zeros((n, n))
    offset = 0
    for at in molecular_system:
        atbas = at.element + at.basis
        len_atbas = atomic_dens[atbas][0].shape[0]
        mol_dens_a[offset:offset + len_atbas,
                   offset:offset + len_atbas] = atomic_dens[atbas][0]
        mol_dens_b[offset:offset + len_atbas,
                   offset:offset + len_atbas] = atomic_dens[atbas][1]
        offset += len_atbas
    return (mol_dens_a + mol_dens_b) / 2, (mol_dens_a + mol_dens_b) / 2


class _SADargs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def calc_at_dens(element, basis, conjugacy, step_size):
    """Calculate atomic densisty
    
    Parameters:
    -----------
    element (str)
        Element symbol
    
    basis (str)
        Basis set name

    conjugacy and step_size are a hack to get the SAD working. find a better way.
    
    """
    fname = 'atomicdenstmpFindnewname.xyz'
    with open(fname, 'w') as f:
        f.write('1\n')
        f.write('Atom for calc_at_dens\n')
        f.write(f'{element} 0.0 0.0 0.0\n')
    args = _SADargs(geometry=fname,
                    basis=basis,
                    ms2=ATOMS.index(element) % 2,
                    charge=0,
#                    ms2=0,
#                    charge=ATOMS.index(element) % 2,
                    restricted=False,
                    max_iter=30,
                    diis=5,
                    diis_at_F=True,
                    diis_at_P=False,
                    grad_type='F_asym',
                    step_type='SCF',
                    ini_orb='Hcore',
                    conjugacy=conjugacy,
                    step_size=step_size
    )
    atHF = main.main(args, None)#sys.stdout)
    return atHF.density
