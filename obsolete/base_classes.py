"""Some base classes for wave functions


"""

from wave_functions.general import WaveFunction


class FCIforMinDGrassmannWaveFunction(WaveFunction):
    """Wave function for optimisation of Phi_minD at the Grassmannian
    """
        
    @abstractmethod
    def string_indices(self,
                       spirrep=None,
                       coupled_to=None,
                       no_occ_orb=False,
                       only_ref_occ=False,
                       only_this_occ=None):
        """A generator that yields all string indices
        
        Behaviour:
        ----------
        
        The indices that this generator yield should be an instance
        of StringIndex or of SpirrepStringIndex.
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
        that differ from range(ref_orb(irrep)) for at most
        two inices (that is, a double excitation)
        
        Parameters:
        -----------
        
        spirrep (int, default=None)
            If passed, SpirrepStringIndex of this spirrep are yield
        
        coupled_to (tuple, default=None)
            If passed, it should be a tuple of SpirrepIndex,
            and the function should yield all StringIndex that have the
            .Index for .spirrep, or all SpirrepStringIndex of the
            given spirrep that are coupled to spirrep for that wave function
        
        no_occ_orb (bool, default=False)
            If True, do not waste time filling the attribute occ_orb.
        
        only_ref_occ (bool, default=False)
            If True, yield only the string indices that have the same
            occupation of the reference wave function per irrep
        
        only_this_occ (int or tuple of int, default=None)
            If passed, should be an int if parameter spirrep was
            also given, otherwise a tuple where the entries are the
            occupation per irrep. Thus, only indices with such occupation
            are yield. If not given, the occupation of reference is used.
        
        Yield:
        ------
        
        Instances of StringIndex or of SpirrepStringIndex (if spirrep
        was given)
        """
        pass
    
    @abstractmethod
    def make_Jac_Hess_overlap(self, analytic=True):
        """Construct the Jacobian and the Hessian of the function overlap.
        
        Behaviour:
        ----------
        
        The function is f(x) = <wf(x), det1(x)>
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
    def calc_wf_from_z(self, z, just_C0=False):
        """Calculate the wave function in a new orbital basis
        
        Behaviour:
        ----------
        
        Given the wave function in the current orbital basis,
        a new (representation of the) wave function is constructed
        in a orbital basis that has been modified by a step z.
        
        Paramters:
        ----------
        
        z   the update in the orbital basis (given in the space of the
            K_i^a parameters) from the position z=0 (that is, the orbital
            basis used to construct the current representation of the
            wave function
        just_C0  Calculates only the first coefficient (see transform_wf)
        
        Return:
        -------
        
        a tuple (new_wf, Ua, Ub) where new_wf is a WaveFunction
        with the wave function in the new representation, and Ua and Ub
        are the transformations from the previous to the new orbital
        basis (alpha and beta, respectively).
        """
        pass
    
    @abstractmethod
    def change_orb_basis(self, U, just_C0=False):
        r"""Transform the wave function after a change in the orbital basis
        
        Behaviour:
        ----------
        
        If the coefficients of wf are given in the basis |u_I>:
        
        |wf> = \sum_I c_I |u_I>
        
        it calculates the wave function in the basis |v_I>:
        
        |wf> = \sum_I d_I |v_I>
        
        and Ua and Ub are the matrix transformations of the
        MO from the basis |v_I> to the basis |u_I>:
    
        |MO of (u)> = |MO of (v)> U

        Parameters:
        -----------
        
        wf   the initial wave function as WaveFunction
        U    the orbital transformation
        just_C0   If True, calculates the coefficients of
                  the initial determinant only (default False)
        
        Return:
        -------
        
        The transformed wave function
        """
        pass

