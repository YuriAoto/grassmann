"""Symmetry


"""

import numpy as np

from util.variables import int_dtype

irrep_product = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7],
                            [1, 0, 3, 2, 5, 4, 7, 6],
                            [2, 3, 0, 1, 6, 7, 4, 5],
                            [3, 2, 1, 0, 7, 6, 5, 4],
                            [4, 5, 6, 7, 0, 1, 2, 3],
                            [5, 4, 7, 6, 1, 0, 3, 2],
                            [6, 7, 4, 5, 2, 3, 0, 1],
                            [7, 6, 5, 4, 3, 2, 1, 0]],
                           dtype=int_dtype)

number_of_irreducible_repr = {
    'C1': 1,
    'Cs': 2,
    'C2': 2,
    'Ci': 2,
    'C2v': 4,
    'C2h': 4,
    'D2': 4,
    'D2h': 8}
