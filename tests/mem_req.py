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



bytes_unit_factor = {
    'B': 0,
    'kB': 1,
    'MB': 2,
    'GB': 3,
    'TB': 4}

def toGB(n, float_size=64, unit='GB'):
    """The amount of memory used by n floats"""
    try:
        return n * float_size / (1024**bytes_unit_factor[unit])
    except KeyError:
        raise Exception('unknown memory unit: ' + unit)


