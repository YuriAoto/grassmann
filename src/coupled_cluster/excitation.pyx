

cdef class SDExcitation:
    """Single and double excitation
    
    A class to store single and double excitations
    
    The excitation from orbitals i,j to a,b are stored in 
    the holes ("_h") and particle ("_p") arrays, respectively.
    There are separated arrays for alpha and beta orbitals.
    
    This is a low-level extension type, and the functions do not check
    if there is space in the arrays for a new pair, such that the ranks
    remain lower or equal than 2. The caller is responsible to avoid this
    
    Attributes:
    -----------
    alpha_rank, beta_rank (int)
        The ranks of alpha and beta excitations. Should be lower or equal 2
    
    alpha_h, alpha_p, beta_h, beta_p (int[2])
        The arrays with alpha and beta holes and particles.
        They each should be in strict ascending order
    
    Examples:
    ---------
    X means an undetermined integer.
    
    # Single excitation from the alpha orbital 1 to the alpha orbital 8:
    
    exc.alpha_rank = 1
    exc.beta_rank = 0
    exc.alpha_h = [1, X]
    exc.alpha_p = [8, X]
    exc.beta_h = [X, X]
    exc.beta_p = [X, X]
    
    # Double excitation from the alpha orbital 1 to the alpha orbital 8,
    and from beta orbital 4 to beta orbital 11:
    
    exc.alpha_rank = 1
    exc.beta_rank = 1
    exc.alpha_h = [1, X]
    exc.alpha_p = [8, X]
    exc.beta_h = [4, X]
    exc.beta_p = [11, X]
    
    # Double excitation from the alpha orbitals 1 and 4 to the alpha orbitals
    8 and 11:
    
    exc.alpha_rank = 2
    exc.beta_rank = 0
    exc.alpha_h = [1, 4]
    exc.alpha_p = [8, 11]
    exc.beta_h = [X, X]
    exc.beta_p = [X, X]
    
    """
    
    def __cinit__(self):
        self.alpha_rank = 0
        self.beta_rank = 0

    cdef inline int rank(self):
        return self.alpha_rank + self.beta_rank

    cdef void add_alpha_hp(self, int i, int a):
        """Add a new pair hole/particle to the alpha excitation
        
        The rank of alpha is increased.
        
        Parameters:
        -----------
        i, a
            The hole and the particle indices
        
        """
        self.alpha_h[self.alpha_rank] = i
        self.alpha_p[self.alpha_rank] = a
        self.alpha_rank += 1
    
    cdef void add_beta_hp(self, int i, int a):
        """Add a new pair hole/particle to the beta excitation
        
        The rank of beta is increased
        
        Parameters:
        -----------
        i, a
            The hole and the particle indices
        
        """
        self.beta_h[self.beta_rank] = i
        self.beta_p[self.beta_rank] = a
        self.beta_rank += 1
