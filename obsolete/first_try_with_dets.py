Single_Amplitude = namedtuple('Single_Amplitude', ['t', 'i', 'a'])
Double_Amplitude = namedtuple('Double_Amplitude', ['t', 'i', 'j', 'a', 'b'])

Slater_Det = namedtuple('Slater_Det', ['c',
                                       'occupied_orb'])
Slater_Det.__doc__ = """A Slater Determinant

occupied_orb = [a1, a2, ..., an_a, b1, b2, ..., bn_a]
TODO: use bits for separated alpha and beta strings
"""

Ref_Det = namedtuple('Ref_Det', ['c'])

Singly_Exc_Det = namedtuple('Singly_Exc_Det', ['c',
                                               'i', 'a', 'spin'])

Doubly_Exc_Det = namedtuple('Doubly_Exc_Det', ['c',
                                               'i', 'a', 'spin_ia',
                                               'j', 'b', 'spin_jb'])
Doubly_Exc_Det.__doc__ = """
Important convention:
alpha spin __always__ first!
So, using the canonical order that the alpha orbitals come
first, then "i" < "a" < "j" < "b", where "x" indicates
x + orb_dim for beta orbitals.
Thus, if spin_ia != spin_jb, then spin_ia = +1 and spin_jb = -1
"""

class Slater_Det():
    """A Slater determinant, along with a coefficient
    
    """
    __slots__ = ('c',
                 'alpha_occ',
                 'beta_occ')

    def __init__(self):
        c = 0.0
        alpha_occ = 0
        beta_occ = 0
