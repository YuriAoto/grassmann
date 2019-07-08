"""Optimizer of the distance to the FCI wave function

History
    Aug 2018 - Start
    Mar 2019 - Organise and comment the code
               Add to git

Yuri
"""
import copy
import math
import sys
import logging

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)

def optimize_distance_to_FCI(fci_wf,
                             construct_Jac_Hess,
                             str_Jac_Hess,
                             calc_wf_from_z,
                             transform_wf,
                             max_iter = 20,
                             f_out = sys.stdout,
                             ini_U = None):
    """Find a single Slater determinant that minimise the distance to fci_wf

    It uses a Newton-Raphson method for the optimisation.
    See Molpro_FCI_Wave_Function for an example of the classes and
    functions used.

    The argument fci_wf is not altered.

    Returns the following tuple:
    ((Ua, Ub),
     (normZ, normJ),
     i_iteration,
     converged)

    where Ua and Ub are the transformation matrices for the optimised orbitals,
    normZ and normJ are the norm of the vectors z and J,
    i_iteration is the number of iterations,
    converged is True or False, indicating if the procedure converged


    Parameters:
    fci_wf   an instance of a class that represents the fci wave function 
             (or a wave function with the structure of a FCI wave function).
             

    construct_Jac_Hess   a method that receives fci_wf and contructs the
                         Jacobian and the Hessian of the function
                         f(x) = <x,fci_wf>, where x is a Slater determinant

    str_Jac_Hess       a method that returns str of the Jacobian and the Hessian

    calc_wf_from_z     a method that calculates the wave function from 
                       a parametrization z

    transform_wf      a method to transforma the wave function to a new basis

    max_iter   the maximum number of iterations in the optimisation
               (default = 20)

    f_out   the output stream (default = sys.stdout)

    ini_U   if not None, it should be a two elements tuple with
            initial transformation for alpha and beta orbitals.
            if None, Identity is used as initial transformation.
            (default = None)
    """
    converged = False
    try_uphill = False
    i_pos_eigVal = -1
    i_iteration = 0
    if ini_U is None:
        Ua = np.identity(fci_wf.orb_dim)
        Ub = np.identity(fci_wf.orb_dim)
        cur_wf = copy.copy(fci_wf)
    else:
        Ua = ini_U[0]
        Ub = ini_U[1]
        cur_wf = transform_wf(fci_wf, Ua, Ub)

    f_out.write('{0:<5s}  {1:<11s}  {2:<11s}\n'.\
                format('it.', '|z|', '|Jac|'))

    for i_iteration in range(max_iter):
        logger.info('Starting iteration %d',
                    i_iteration)
        if  logger.level <= logging.DEBUG:
            logger.debug('Wave function:\n' + str(cur_wf))

        Jac, Hess = construct_Jac_Hess(cur_wf)
        logger.info('Hessian and Jacobian are built\n')
        if logger.level <= 1:
            logger.log(1, str_Jac_Hess(Jac, Hess, cur_wf))

        if False: # Check Jac and Hessian
            num_Jac, num_Hess = construct_Jac_Hess(cur_wf,
                                                   analytic=False)
            diff = 0.0
            for i,x in enumerate(num_Jac):
                diff += abs(x-Jac[i])
            logger.debug('Sum abs(diff(Jac, numJac)) = {0:10.6e}'.\
                         format(diff))
            diff = 0.0
            for i1, x1 in enumerate(num_Hess):
                for i2, x2 in enumerate(x1):
                    diff += abs(x2-Hess[i1][i2])
            logger.debug('Sum abs(diff(Hess, numHess)) = {0:10.6e}'.\
                         format(diff))
            logger.debug('Analytic:\n%s',
                         str_Jac_Hess(Jac, Hess, cur_wf))
            logger.debug('Numeric:\n%s',
                         str_Jac_Hess(num_Jac, num_Hess, cur_wf))

        eig_val, eig_vec = linalg.eigh(Hess)
        Hess_has_pos_eig = False
        Hess_dir_with_posEvec = None
        for i,eigVal in enumerate(eig_val):
            if eigVal > 0.0:
                Hess_has_pos_eig = True
                Hess_dir_with_posEvec = np.array(eig_vec[:,i])
                logger.warning('Hessian has positive eigenvalue.')
                break

        normJ = 0.0
        for i in Jac:
            normJ += i**2
        normJ = math.sqrt(normJ)

        if not try_uphill:
            # Newton step
            Hess_inv = linalg.inv(Hess)
            logger.info('Hessian is inverted.')
            z = -np.matmul(Hess_inv, Jac)
            logger.info('z vector has been calculated by Newton step.')

        else:
            #den = np.matmul(np.matmul(Jac.transpose(), Hess), Jac)
            #gamma = 0.0
            #for j in Jac:
            #    gamma += j**2
            #gamma = -gamma/den

            # Look maximum in the direction of eigenvector of Hess with pos 
            gamma = 0.5
            logger.info('Calculating z vector by Gradient descent;\n'
                        + 'gamma = {0:.5f}\n'.format(gamma))
            max_c0 = cur_wf.determinants[0][0]
            max_i0 = 0
            if logger.level <= 15:
                logger.log(15, 'Current C0: {0:.4f}'.format(max_c0))
            for i in range(6):
                tmp_wf, tmp_Ua, tmp_Ub = calc_wf_from_z(i*gamma*Hess_dir_with_posEvec, cur_wf,
                                                        just_C0=True)
                this_c0 = tmp_wf.determinants[0][0]
                logger.debug('Attempting for i=%d: C0 = %f', i, this_c0)
                if abs(max_c0) < abs(this_c0):
                    max_c0 = this_c0
                    max_i0 = i
            z = max_i0*gamma*Hess_dir_with_posEvec
            logger.info('z vector obtained: %d*gamma*Hess_dir_with_posEvec',
                        max_i0)
            try_uphill = False

        normZ = 0.0
        for i in z:
            normZ += i**2
        normZ = math.sqrt(normZ)

        f_out.write('{0:<5d}  {1:<11.8f}  {2:<11.8f}\n'.\
                    format(i_iteration,
                           math.sqrt(normZ),
                           math.sqrt(normJ)))
        f_out.flush()
        logger.info('Norm of z vector: {0:8.5e}'.\
                    format(normZ))
        logger.info('Norm of J vector: {0:8.5e}'.\
                    format(normJ))
    
        if normJ < 1.0E-8 and normZ < 1.0E-8 and not try_uphill:
            if not Hess_has_pos_eig:
                converged = True
                break
            else:
                try_uphill = True
            # elif try_uphill:
            #     try_uphill = False

        logger.info('Calculating new basis and transforming wave function.')
        cur_wf, cur_Ua, cur_Ub = calc_wf_from_z(z, cur_wf)
        logger.info('New basis and wave function have been calculated!')

        Ua = np.matmul(Ua, cur_Ua)
        Ub = np.matmul(Ub, cur_Ub)

    return ((Ua, Ub),
            (normZ, normJ),
            i_iteration,
            converged)