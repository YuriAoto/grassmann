""" Some useful functions and definitions

"""

import math

sqrt2 = math.sqrt(2.0)

def dist_from_ovlp(x):
    return sqrt2 * math.sqrt(1 - abs(x))
