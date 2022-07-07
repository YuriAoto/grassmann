"""Functions to calculated and deal with memory requirements

"""
import numpy as np

def n_G_of_singles_CS_CISD(orb_dim, ref_occ, n_core):
    """Return the number of floats stored for all G of singles"""
    if isinstance(orb_dim, (int, np.integer)):
        orb_dim = (orb_dim,)
        ref_occ = (ref_occ,)
        n_core = (n_core,)
    S = 0
    for i in range(len(orb_dim)):
        S += ((ref_occ[i] - n_core[i]) * (orb_dim[i] - ref_occ[i])
              * orb_dim[i] * ref_occ[i])
    return S

def n_H_of_singles_CS_CISD(orb_dim, ref_occ, n_core):
    """Return the number of floats stored for all H of singles"""
    if isinstance(orb_dim, (int, np.integer)):
        orb_dim = (orb_dim,)
        ref_occ = (ref_occ,)
        n_core = (n_core,)
    S = 0
    for i in range(len(orb_dim)):
        S += ((ref_occ[i] - n_core[i]) * (orb_dim[i] - ref_occ[i])
              * (orb_dim[i] * ref_occ[i] * orb_dim[i] * ref_occ[i]))
    return S
