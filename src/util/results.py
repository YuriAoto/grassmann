"""Results of calculations

"""


_res_box_string = '\n============================================================\n'

def inside_box(func):
    def wrapper_res_box(*args, **kargs):
        x = func(*args, **kargs)
        return ''.join([_res_box_string,
                        x.replace(_res_box_string, ''),
                        _res_box_string])
    return wrapper_res_box


class Results:
    """Class to store results
    
    Basic class for results of calculations.
    
    Attributes:
    -----------
    kind (str)
        Kind of result
    
    success (bool)
        Indicate success of calculation.
        If a optimisation procedure, should be True if converged
    
    error (str)
        Eventual error messages. Should be None if success == True.
        If success == False, this attribute should indicate what happened
        for the lack of success
    
    warning (str)
        Eventual warning
    
    """
    
    def __init__(self, kind):
        self.kind = kind
        self.success = None
        self.error = None
        self.warning = None
    
    @inside_box
    def __str__(self):
        x = ['Results for ' + self.kind + ':']
        x.append('Success = ' + str(self.success))
        if self.error is not None:
            x.append('error = ' + str(self.error))
        if self.warning is not None:
            x.append('warning = ' + str(self.warning))
        return '\n'.join(x)


class OptResults(Results):
    """ Class to store results of a optimisations
    
    Every attribute that is not relevant for this kind
    of calculation should be None.
    
    These are main attributes.
    Add extra attributes to the instances if needed.
    
    Attributes:
    -----------
    energy (float)
        Final energy. Should be None if it is not a energy calculation
    
    distance (float)
        Final distance. Should be None if it is not a energy calculation
    
    wave_function (WaveFunction)
        The final wave function
    
    orbitals (MolecularOrbitals)
        Final molecular orbitals
    
    n_iter (int)
        Number of iterations to reach convergence
    
    conv_norm (float or tuple of floats)
        The norm of vector or vectors that should vanish at convergence
    
    """
    
    def __init__(self, kind):
        super().__init__(kind)
        self.energy = None
        self.distance = None
        self.wave_function = None
        self.orbitals = None
        self.n_iter = None
        self.conv_norm = None

    @inside_box
    def __str__(self):
        x = [super().__str__()]
        if self.energy is not None:
            x.append('Energy = ' + str(self.energy))
        if self.distance is not None:
            x.append('Distance = ' + str(self.distance))
        return '\n'.join(x)
