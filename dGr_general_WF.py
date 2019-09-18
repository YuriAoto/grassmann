"""General electronic wave function

Abstract base class for electronic wave functions.

"""
import logging
from collections import namedtuple
from collections.abc import Sequence
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Wave_Function(ABC, Sequence):
    """An abstract base class for electronic wave functions
    
    Atributes:
    ----------
    
    n_irrep (int)                     the number of irreducible representations
    orb_dim (n_irrep-tuple of int)    dimension of the orbital space of each irrep
    n_elec (int)                      number of electrons
    n_alpha (int)                     number of alpha electrons
    n_beta (int)                      number of beta electrons
    n_frozen (int)                    number of frozen orbitals
                                       (assumed to be equal for alpha and beta)
    ref_occ (n_irrep-tuple of int)    occupation of the reference determinant
    WF_type (str)                     type of wave function
    
    Some rules about Sequence's abstract methods:
    
    __getitem__ should accept values such as returned by string_indices and return
    the corresponding CI coefficient
    
    __len__  should return a tuple of length n_irrep, with the total number of
    strings that string_indices(spirrep=spirrep) will give
    """
    
    def __init__(self):
        self.n_irrep = 0
        self.orb_dim = []
        self.n_elec = 0
        self.n_alpha = 0
        self.n_beta = 0
        self.n_frozen = 0
        self.ref_occ = []
        self.WF_type = ''
    
    def __repr__(self):
        """Return string with parameters of the wave function."""
        x = []
        x.append('-'*50)
        x.append('n irrep: {}'.format(self.n_irrep))
        x.append('orb dim: {}'.format(self.orb_dim))
        x.append('n electrons: {}'.format(self.n_elec))
        x.append('n alpha: {}'.format(self.n_alpha))
        x.append('n beta: {}'.format(self.n_beta))
        x.append('n frozen: {}'.format(self.n_frozen))
        x.append('WF type: {}'.format(self.WF_type))
        x.append('-'*50)
        return '\n'.join(x)
    
    @abstractmethod
    def spirrep_blocks(self):
        """Yield the possible spin and irreps, as a single integer."""
        pass
    
    @abstractmethod
    def string_indices(self, spirrep=None, coupled_to=None):
        """A generator that yields all string indices
        
        Behaviour:
        ----------
        
        The indices I that this generator yield should be iterable,
        giving the string index of each spirrep (in the sequence yield
        by spirrep_blocks) of such determinant.
        The wave function should be indexable by the values
        that this function yield, returning the corresponding coefficient.
        That is, the following construction should print all
        CI coefficients:
        for I in wf.string_indices():
             print(wf[I])
        
        Examples:
        ---------
        
        for a FCI wave function this yields all
        possible such subindices.
        
        for a CISD wave function this yields all subindices
        that differ from range(ref_occ(irrep)) for at most
        two inices (that is, a double excitation)
        
        Parameters:
        -----------
        
        spirrep (int, as returned from spirrep_blocks, default=None)
            If passed, only subindices of this spirrep are yield
        
        coupled_to (tuple, default=None)
            If passed, it should be a even-length tuple
            with each pair being (spirrep_i, I_i), and the function should
            yield all subindices that have the subindice I for spirrep.
        
        Yield:
        ------
        
        String subindices (of only one spirrep, if spirrep is passed) as a
        list of integers
        """
        pass
    
    @abstractmethod
    def make_Jac_Hess(self, analytic = True):
        """Construct the Jacobian and the Hessian of the function overlap.
    
        Behaviour:
        ----------
        
        he function is f(x) = <wf(x), det1(x)>
        where x parametrises the orbital rotations and
        det1 is the first determinant in wf.
        
        Parameters:
        -----------
        
        analytic (bool, default = True)
            If True, calculate the Jacobian and the Hessian by the
            analytic expression, if False calculate numerically
        
        Returns:
        --------
        
        The tuple (Jac, Hess), with the Jacobian and the Hessian
        """
        pass

    @abstractmethod
    def calc_wf_from_z(self, z, just_C0 = False):
        """Calculate the wave function in a new orbital basis
        
        Behaviour:
        ----------
        
        Given the wave function in the current orbital basis,
        a new (representation of the) wave function is constructed
        in a orbital basis that has been modified by a step z.
        
        Paramters:
        ----------
        
        z        the update in the orbital basis (given in the space of the
             K_i^a parameters) from the position z=0 (that is, the orbital
             basis used to construct the current representation of the
             wave function
        cur_wf   current representation of the wave function
             (as Molpro_FCI_Wave_Function)
        just_C0  Calculates only the first coefficient (see transform_wf)
        
        Return:
        -------
        
        a tuple (new_wf, Ua, Ub) where new_wf is a Molpro_FCI_Wave_Function
        with the wave function in the new representation, and Ua and Ub
        are the transformations from the previous to the new orbital
        basis (alpha and beta, respectively).
        """
        pass
    
    @abstractmethod
    def change_orb_basis(self, Ua, Ub, just_C0 = False):
        """Transform the wave function as induced by a transformation in the orbital basis
        
        Behaviour:
        ----------
        
        If the coefficients of wf are given in the basis |u_I>:
        
        |wf> = \sum_I c_I |u_I>
        
        it calculates the wave function in the basis |v_I>:
        
        |wf> = \sum_I d_I |v_I>
        
        and Ua and Ub are the matrix transformations of the MO from the basis |v_I>
        to the basis |u_I>:
    
        |MO of (u)> = |MO of (v)> U

        Parameters:
        -----------
        
        wf   the initial wave function as Molpro_FCI_Wave_Function
        Ua   the orbital transformation for alpha orbitals
        Ub   the orbital transformation for beta orbitals
        just_C0   If True, calculates the coefficients of the initial determinant only
              (default False)
        
        Return:
        -------
        
        The transformed wave function
        """
        pass


