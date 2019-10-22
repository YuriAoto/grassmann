import os
import sys
import traceback
import logging

import numpy as np

np.set_printoptions(linewidth=100,
                    formatter={'float':lambda x: '{0:8.5f}'.format(x)})

logger = logging.getLogger()
handler = logging.StreamHandler()
myfmt = logging.Formatter('{levelname}: {funcName} - {filename}:\n'
                          + '{message}\n',
                          style='{')
logger.setLevel(logging.DEBUG)
handler.setFormatter(myfmt)
logger.addHandler(handler)


try:
    os.chdir('/home/yuriaoto/Documents/Codes/min_dist_Gr/tests/')
    sys.path.insert(0,os.getcwd() + '/../')
    import dGr_orbitals as orbitals
    MO1 = orbitals.Molecular_Orbitals.from_file('H2__R_5__ccpVDZ__C2v/RHF_orbitals.xml')
    MO2 = orbitals.Molecular_Orbitals.from_file('H2__R_5__ccpVDZ__C2v/UHF_orbitals.xml')
    U = MO1.in_the_basis_of(MO2)
    print('Transformation matrices:')
    for Ui in U:
        print()
        print(Ui)
    U2 = MO2.in_the_basis_of(MO1)
    print('\nTransformation matrices (U2):')
    for Ui in U2:
        print()
        print(Ui)

#    print(U[0]@U2[0])
    
except:
    traceback.print_exc()
    raise
finally:
    exit()
