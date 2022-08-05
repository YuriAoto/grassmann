"""Functions to deal with the dataset


"""
import re
from collections import namedtuple

from ase_dataset import g2_1, g2_2

class HF_args:
    def __init__(self, fname):
        self.geometry = fname


def iterate_over(dataset_name):
    if dataset_name == 'g2_1':
        this_dataset = g2_1.data
    elif dataset_name == 'g2_2':
        this_dataset = g2_2.data
        
    for mol, data in this_dataset.items():
        yield mol, _make_args(data)


def _make_args(data):
    """Create the variable args for the function hartree_fock.main.main()
    
    data is an entry of the dict from g2_1 or g2_2
    """
    fname = _make_geom_file(data['positions'], data['symbols'])
    return HF_args(fname)


def _make_geom_file(positions, symbols):
    """Create the xyz file
    
    positions: list of lists, with the cartesian coordinates, for example:
    [[0.0, 0.0, 0.119262],
     [0.0, 0.763239, -0.477047],
     [0.0, -0.763239, -0.477047]]
    
    symbols: atomic symbols, in the order of the positions, for example:
    'OHH'
    
    """
    fname = f'{symbols}_fromdataset.xyz'
    with open(fname, 'w') as f:
        f.write(f'{len(positions)}\nFrom the data set: symbols\n')
        for at, symb in zip(positions, re.findall('[A-Z][a-z]*', symbols)):
            f.write(f'{symb} {at[0]} {at[1]} {at[2]}\n')
    return fname
