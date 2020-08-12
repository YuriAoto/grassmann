"""General functions for Hartree-Fock

"""
import logging

import numpy as np
from scipy.linalg import solve

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()

fmt_HF_header = '{0:<5s}  {1:<16s}  {2:<16s}  {3:s}\n'
fmt_HF_iter = '{0:<5d}  {1:<16.12f}  {2:<16.12f}  {3:s}\n'

fmt_HF_header_general = '{0:<5s}  {1:<16s}  {2:<16s}  {3:6s}  {4:s}\n'
fmt_HF_iter_general = '{0:<5d}  {1:<16.12f}  {2:<16.12f}  {3:6s}  {4:s}\n'


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
        return ('<HFResult:E=' + str(self.Energy)
                + ';success=' + str(self.success) + '>')

    def __str__(self):
        x = 'Results for ' + self.kind + ':\n'
        x += 'Energy = ' + str(self.energy) + '\n'
        x += 'success = ' + str(self.success) + '\n'
        if self.error is not None:
            x += 'error = ' + str(self.error) + '\n'
        if self.warning is not None:
            x += 'warning = ' + str(self.error) + '\n'
        return x


def calculate_DIIS(Dmat, grad, cur_n_DIIS, i_DIIS):
    """Calculate a DIIS step
        
    Parameters:
    -----------
    
    Dmat (np.ndarray)
        The density matrix
    
    grad (np.ndarray)
        Hartree-Fock Gradient
    
    cur_n_DIIS (int)
        The current dimension of the iterative subspace
    
    i_DIIS (int)
        index of Dmat of current iteration
    
    Returns:
    --------
    Does not return anything, but the density matrix
    is updated.
    
    TODO:
    ----
    This does not give the same result for equivalent
    restricted and unrestricted calculations.
    Investigate why.
    
    This implementation is rather simple, and is not sufficient
    to achieve convergence for benzene. Improve this
    
    """
    B = np.zeros((cur_n_DIIS + 1, cur_n_DIIS + 1))
    B[:cur_n_DIIS + 1, :cur_n_DIIS + 1] = np.einsum('iap,iaq->pq',
                                                    grad[:, :, :cur_n_DIIS+1],
                                                    grad[:, :, :cur_n_DIIS+1])
    for i in range(cur_n_DIIS):
        B[i, cur_n_DIIS] = B[cur_n_DIIS, i] = -1.0
    B[cur_n_DIIS, cur_n_DIIS] = 0.0
    logger.debug('DIIS B matrix:\n%r', B)
    a = np.zeros((cur_n_DIIS + 1, 1))
    a[cur_n_DIIS] = -1.0
    w = solve(B, a)
    logger.debug('DIIS w vector:\n%r', w)
    logger.debug('w entry associated to the current step: %f',
                 w[i_DIIS])
    Dmat[:, :, i_DIIS] *= w[i_DIIS]
    for k in range(cur_n_DIIS):
        if k == i_DIIS:
            continue
        Dmat[:, :, i_DIIS] += w[k] * Dmat[:, :, k]
    logger.debug('Density matrix (after DIIS):\n%r',
                 Dmat[:, :, i_DIIS])