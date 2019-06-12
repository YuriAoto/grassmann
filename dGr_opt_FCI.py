"""Optimizer of the distance to the FCI wave function

History
    Aug 2018 - Start
    Mar 2019 - Organise and comment the code
               Add to git

Yuri
"""
import copy
import math
import numpy as np
from scipy import linalg
import sys

def optimize_distance_to_FCI(fci_wf,
                             construct_Jac_Hess,
                             print_Jac_Hess,
                             calc_wf_from_z,
                             transform_wf,
                             max_iter = 20,
                             verbose = 1,
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

    print_Jac_Hess      a method to print the Jacobian and the Hessian

    calc_wf_from_z     a method that calculates the wave function from 
                       a parametrization z

    transform_wf      a method to transforma the wave function to a new basis

    max_iter   the maximum number of iterations in the optimisation
               (default = 20)

    verbose    the level of printing to the output (default = 1):
            <    0   No output
            ==   0   only warnings
            >=   1   iterations
            >=  10   info about steps inside the iterations
                20   C0
                50   the wave function in each iteration
               100   the Jacobian and Hessian in each iteration
               
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

    if verbose == 1:
        f_out.write('{0:<5s}  {1:<11s}  {2:<11s}'.\
                    format('it.', '|z|', '|Jac|'))

    for i_iteration in range(max_iter):
        if verbose >= 10:
            f_out.write('='*50 + '\n')
            f_out.write('Starting iteration {0:4d}\n'.\
                        format(i_iteration))
            if verbose >= 50:
                f_out.write('Wave function:\n' + str(cur_wf))
            f_out.write('='*25 + '\n')

        Jac, Hess = construct_Jac_Hess(cur_wf, f_out, verb = 0)
        if verbose >= 10:
            f_out.write('## Hessian and Jacobian are built\n')
        if verbose >= 100:
            print_Jac_Hess(Jac, Hess, f_out)

        if False: # Check Jac and Hessian
            num_Jac, num_Hess = construct_Jac_Hess(cur_wf, f_out,
                                                   verb = 0, analytic=False)
            diff = 0.0
            for i,x in enumerate(num_Jac):
                diff += abs(x-Jac[i])
            f_out.write('Sum abs(diff(Jac, numJac)) = {0:10.6e}\n'.\
                        format(diff))
            diff = 0.0
            for i1, x1 in enumerate(num_Hess):
                for i2, x2 in enumerate(x1):
                    diff += abs(x2-Hess[i1][i2])
            f_out.write('Sum abs(diff(Hess, numHess)) = {0:10.6e}\n'.\
                        format(diff))
            f_out.write('=-'*30 + '\n#Analytic\n')
            print_Jac_Hess(Jac, Hess, f_out)
            f_out.write('=-'*30 + '\n#Numeric\n')
            print_Jac_Hess(num_Jac, num_Hess, f_out)

        eig_val, eig_vec = linalg.eigh(Hess)
        Hess_has_pos_eig = False
        Hess_dir_with_posEvec = None
        for i,eigVal in enumerate(eig_val):
            if eigVal > 0.0:
                Hess_has_pos_eig = True
                Hess_dir_with_posEvec = np.array(eig_vec[:,i])
                if verbose >= 0:
                    f_out.write('WARNING: Hessian has positive eigenvalue.\n')
                break

        normJ = 0.0
        for i in Jac:
            normJ += i**2
        normJ = math.sqrt(normJ)

        if not try_uphill:
            # Newton step
            
            Hess_inv = linalg.inv(Hess)
            if verbose >= 10:
                f_out.write('## Hessian is inverted\n')

            z = -np.matmul(Hess_inv, Jac)
            if verbose >= 10:
                f_out.write('## z vector have been calculated by Newton step\n')

        else:
            #den = np.matmul(np.matmul(Jac.transpose(), Hess), Jac)
            #gamma = 0.0
            #for j in Jac:
            #    gamma += j**2
            #gamma = -gamma/den

            # Look maximum in the direction of eigenvector of Hess with pos 
            gamma = 0.5
            if verbose >= 10:
                f_out.write('## Calculating z vector by Gradient descent;\n' +\
                            'gamma = {0:.5f}\n'.format(gamma))
            max_c0 = cur_wf.determinants[0][0]
            max_i0 = 0
            if verbose >= 20:
                f_out.write(' Current C0: {0:.4f}\n'.format(max_c0))
                f_out.flush()
            for i in range(6):
                tmp_wf, tmp_Ua, tmp_Ub = calc_wf_from_z(i*gamma*Hess_dir_with_posEvec, cur_wf, f_out,
                                                        verb = 0, just_C0 = True)
                this_c0 = tmp_wf.determinants[0][0]
                if verbose >= 20:
                    f_out.write(' Attempting for i={0:d}: C0={1:.4f}\n'.format(i, this_c0))
                if abs(max_c0) < abs(this_c0):
                    max_c0 = this_c0
                    max_i0 = i
            z = max_i0*gamma*Hess_dir_with_posEvec
            if verbose >= 10:
                f_out.write('## z vector obtained: {0:d}*gamma*Hess_dir_with_posEvec\n'.\
                            format(max_i0))
            try_uphill = False


        normZ = 0.0
        for i in z:
            normZ += i**2
        normZ = math.sqrt(normZ)

        if verbose == 1:
            f_out.write('{0:<5d}  {1:<11.8f}  {2:<11.8f}\n'.\
                        format(i_iteration,
                               math.sqrt(normZ),
                               math.sqrt(normJ)))
        elif verbose >= 1:
            f_out.write('Norm of z vector: {0:8.5e}\n'.\
                        format(normZ))
            f_out.write('Norm of J vector: {0:8.5e}\n'.\
                        format(normJ))
    
        if normJ < 1.0E-8 and normZ < 1.0E-8 and not try_uphill:
            if not Hess_has_pos_eig:
                converged = True
                break
            else:
                try_uphill = True
#        elif try_uphill:
#            try_uphill = False

        if verbose >= 10:
            f_out.write('## Calculating new basis and transforming wave function...\n')
        cur_wf, cur_Ua, cur_Ub = calc_wf_from_z(z, cur_wf, f_out, verb = verbose)
        if verbose >= 10:
            f_out.write('## New basis and wave function have been calculated!\n')

        Ua = np.matmul(Ua, cur_Ua)
        Ub = np.matmul(Ub, cur_Ub)

    return ((Ua, Ub),
            (normZ, normJ),
            i_iteration,
            converged)
