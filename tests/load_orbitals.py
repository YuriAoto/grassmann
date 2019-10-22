import os
import sys
import traceback

try:
    os.chdir('/home/yuriaoto/Documents/Codes/min_dist_Gr/tests/')
    sys.path.insert(0,os.getcwd() + '/../')
    import dGr_orbitals as orbitals
    orb = orbitals.Molecular_Orbitals.from_file('H2__R_5__ccpVDZ__nosym/UHF_orbitals.xml')
    print(orb)
    
except:
    traceback.print_exc()
    raise
finally:
    exit()
