def print_Jac_Hess(J, H, wf, f_str):
    """Print the Jacobian (J) and the Hessian (H) of wave function wf to f_str."""
    f_str.write('='*50 + '\n')
    f_str.write('Jacobian:\n')
    for i, x in enumerate(J):
        exc_from, exc_to = get_i_a_from_pos_in_jac(i, wf.n_alpha, wf.beta_shift)
        f_str.write('[{0:3d}] = {1:.6f}; spin = {2:s}; K_{3:d}^{4:d}\n'.\
                    format(i, x,
                           'a' if i<wf.beta_shift else 'b',
                           exc_from, exc_to))
    f_str.write('-'*50 + '\n')
    f_str.write('Hessian:\n')
    for i,II in enumerate(H):
        exc_from_i, exc_to_i = get_i_a_from_pos_in_jac(i, wf.n_alpha, wf.beta_shift)
        for j,x in enumerate(II): 
            exc_from_j, exc_to_j = get_i_a_from_pos_in_jac(j, wf.n_alpha, wf.beta_shift)
            f_str_write(' [{0:3d},{1:3d}] = {2:.6f}; K_{3:d}^{4:d} [{5:s}]; K_{6:d}^{7:d} [{8:s}]\n'.\
                        format(i, j,  x,
                               exc_from_i, exc_to_i, 'a' if i<wf.beta_shift else 'b',
                               exc_from_j, exc_to_j, 'a' if j<wf.beta_shift else 'b'))
    f_str.write('='*50 + '\n')

    


def print_matrix(X, f_str):
    """Print the matrix X to f_str."""
    for i in X:
        for j in i:
            f_str.write(' {0:10.6f} '.format(j)\
                        if abs(j) > 1.0E-7 else
                        (' ' + '-'*10 + ' '))
        f_str.write('\n')
    f_str.write('\n')

