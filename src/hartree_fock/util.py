"""General functions for Hartree-Fock

"""

class HFResult():
    """ Class to store results of a Hartree-Fock calculation
    """
    
    def __init__(self, E, orbitals, success, n_iter):
        self.energy = E
        self.orbitals = orbitals
        self.success = success
        self.n_iter = n_iter
        self.kind = None
        self.error = None
        self.warning = None

    def __repr__(self):
        return '<HFResult:E=' + str(self.Energy), ';success=',str(self.success) + '>'

    def __str__(self):
        x = 'Results for ' + self.kind + ':\n'
        x += 'Energy = ' + str(self.energy) + '\n'
        x += 'success = ' + str(self.success) + '\n'
        if self.error is not None:
            x += 'error = ' + str(self.error) + '\n'
        if self.warning is not None:
            x += 'warning = ' + str(self.error) + '\n'
        return x
