"""Put molecular integrals in the atomic basis into a file

Copy this file to a local folder, change the variable grassmann_dir,
and run it as below.

Usage:

python generate_integrals <xyzfile> <basis_set> [print]


Example:

python Hartree_Fock.py H2O.xyz

"""
import os
import sys

import numpy as np

grassmann_dir = '/PATH_TO_/grassmann/'

sys.path.append(grassmann_dir)

import tests
from molecular_geometry.molecular_geometry import MolecularGeometry

xyzfile = sys.argv[1]
basis = sys.argv[2]
print_ = False
if len(sys.argv) > 3 and sys.argv[3] == 'print':
    print_ = True

molecular_system = MolecularGeometry.from_xyz_file(xyzfile)
molecular_system.calculate_integrals(basis, int_meth='ir-wmme')
molecular_system.integrals.g.transform_to_4D()

h = molecular_system.integrals.h
g = molecular_system.integrals.g._integrals
S = molecular_system.integrals.S
Vnuc = molecular_system.nucl_rep

outname = xyzfile.replace('.xyz', '_'+basis)

if print_:
    print(f'h:\n{h}\n\n')
    print(f'g:\n{g}\n\n')
    print(f'S:\n{S}\n\n')

np.savez(f'{outname}', h=h, g=g, S=S, Vnuc=Vnuc)
