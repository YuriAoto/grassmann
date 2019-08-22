"""General electronic wave function

Base class for electronic wave functions.

"""
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)

Single_Amplitude = namedtuple('Single_Amplitude', ['t', 'i', 'a'])
Double_Amplitude = namedtuple('Double_Amplitude', ['t', 'i', 'j', 'a', 'b'])

Ref_Det = namedtuple('Ref_Det', ['c'])
Singly_Exc_Det = namedtuple('Singly_Exc_Det', ['c',
                                               'i', 'a', 'spin'])
# Important convention:
# alpha spin __always__ first!
# So, using the canonical order that the alpha orbitals come
# first, then "i" < "a" < "j" < "b", where "x" indicates
# x + orb_dim for beta orbitals.
# Thus, if spin_ia != spin_jb, then spin_ia = +1 and spin_jb = -1
#
Doubly_Exc_Det = namedtuple('Doubly_Exc_Det', ['c',
                                               'i', 'a', 'spin_ia',
                                               'j', 'b', 'spin_jb'])

class Wave_Function():
    """A base class for electronic wave functions
    
    Atributes:
    orb_dim (int)    dimension of the (spatial) orbital space
    n_frozen (int)   number of frozen orbitals (assumed to be equal for alpha and beta)
    n_elec (int)     number of electrons
    n_alpha (int)    number of alpha electrons
    n_beta (int)     number of beta electrons (always equals n_alpha in the current
                     implementation)
    WF_type (str)    type of wave function
    
    """
    
    def __init__(self):
        self.orb_dim = None
        self.n_frozen = None
        self.n_elec = None
        self.n_alpha = None
        self.n_beta = None
        self.WF_type = None
    
    def __repr__(self):
        """Return string with parameters of the wave function."""
        x = []
        x.append('-'*50)
        x.append('n frozen: {}'.format(self.n_frozen))
        x.append('n electrons: {}'.format(self.n_elec))
        x.append('n alpha: {}'.format(self.n_alpha))
        x.append('n beta: {}'.format(self.n_beta))
        x.append('orb dim: {}'.format(self.orb_dim))
        x.append('WF type: {}'.format(self.WF_type))
        x.append('-'*50)
        return '\n'.join(x)

