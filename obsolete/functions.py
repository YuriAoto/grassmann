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



###
#  From Wave_Function_Int_Norm. Initial version
    def distance_to_det_OLDER_version(self, U, assume_orth = False):
        """Calculates the distance to the determinant U
        
        See dGr_FCI_Molpro.Molpro_FCI_Wave_Function.distance_to_det
        """
        if isinstance(U, tuple):
            Ua, Ub = U
        else:
            Ua = Ub = U
        f0_a = Absil.calc_fI(Ua, range(self.n_alpha))
        f0_b = Absil.calc_fI(Ub, range(self.n_beta))
        f = f0_a * f0_b
        for S in self.singles:
            fI_a = Absil.calc_fI(Ua, get_I(self.n_alpha, S.i, S.a)) * f0_b
            fI_b = Absil.calc_fI(Ub, get_I(self.n_beta, S.i, S.a)) * f0_a
            if (S.i + self.n_alpha-1) % 2 == 1:
                fI_a *= -1
            if (S.i + self.n_beta-1) % 2 == 1:
                fI_b *= -1
            f += S.t * (fI_a + fI_b)
        for D in self.doubles:
            if D.a < D.b:
                continue
            fI = (Absil.calc_fI(Ua, get_I(self.n_alpha, D.i, D.a))
                  * Absil.calc_fI(Ub, get_I(self.n_beta, D.j, D.b)))
            if D.a != D.b:
                fI += (Absil.calc_fI(Ua, get_I(self.n_alpha, D.i, D.b))
                       * Absil.calc_fI(Ub, get_I(self.n_beta, D.j, D.a)))
            if D.i != D.j:
                fI += (Absil.calc_fI(Ua, get_I(self.n_alpha, D.j, D.a))
                       * Absil.calc_fI(Ub, get_I(self.n_beta, D.i, D.b)))
                if D.a != D.b:
                    fI += (Absil.calc_fI(Ua, get_I(self.n_alpha, D.j, D.b))
                           * Absil.calc_fI(Ub, get_I(self.n_beta, D.i, D.a)))
            t_compl = 0.0
            if D.a != D.b:
                fI /= 2    
                for Dba in self.doubles:
                    if (D.i == Dba.i
                        and D.j == Dba.j
                        and D.a == Dba.b
                        and D.b == Dba.a):
                        t_compl = Dba.t
                        break
            fI *= D.t + t_compl
            if D.a != D.b and D.i != D.j:
                fI2 = Absil.calc_fI(Ua, get_I(self.n_alpha, [D.i, D.j], [D.b, D.a])) * f0_b
                fI2 += Absil.calc_fI(Ua, get_I(self.n_beta, [D.i, D.j], [D.b, D.a])) * f0_a
                fI += fI2 * (t_compl - D.t)
            if (D.i + D.j) % 2 == 1:
                fI = -fI
            f += fI
        f /= self.norm
        if not assume_orth:
            Da = linalg.det(np.matmul(Ua.T, Ua))
            Db = linalg.det(np.matmul(Ub.T, Ub))
            f /= math.sqrt(Da * Db)
        return f
