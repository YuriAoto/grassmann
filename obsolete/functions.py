            self.norm = 1.0
            for S in self.singles:
                self.norm += 2 * S.t**2
            for D in self.doubles:
                if D.a == D.b:
                    self.norm += (1 if D.i == D.j else 2) * D.t**2
                elif D.a > D.b:
                    t_compl = 0.0
                    for Dba in self.doubles:
                        if (D.i == Dba.i
                            and D.j == Dba.j
                            and D.a == Dba.b
                            and D.b == Dba.a):
                            t_compl = Dba.t
                            break
                    if D.i == D.j:
                        self.norm += 0.5 * (D.t + t_compl)**2
                    else:
                        self.norm += 4 * (D.t**2 + t_compl**2 - D.t*t_compl)
                else:
                    continue


elif self.WF_type == 'MRCI':
    if MRCI_ref_str == l:
        ref_found = True
    elif MRCI_sgl_str == l:
        sgl_found = True
    elif MRCI_dbl_str == l:
        dbl_found = True
    elif dbl_found:
        lspl = l.split()
        if len(lspl) == 9:
            doubles.append([lspl[-1],
                            int(lspl[1].split('.')[0]),
                            int(lspl[2].split('.')[0]),
                            int(lspl[3].split('.')[0]),
                            int(lspl[4].split('.')[0]),
                            int(lspl[5])])
    elif sgl_found:
        lspl = l.split()
        if len(lspl) == 3:
            singles.append([lspl[-1],
                            lspl[0],
                            int(lspl[1].split('.')[0])])
    elif ref_found:
        lspl = l.split()
        if len(lspl) == 2:
            ref = lspl[1]
    if 'RESULTS' in l:
        if not ref_found:
            raise Exception('Reference coefficients not found!')
        if not dbl_found:
            raise Exception('Double excitations not found!')
        break

elif self.WF_type == 'CCSD' or self.WF_type == 'CISD':
    if CC_sgl_str == l:
        sgl_found = True
    elif CC_dbl_str == l:
        dbl_found = True
    elif dbl_found:
        lspl = l.split()
        if len(lspl) == 7:
            doubles.append([lspl[-1],
                            int(lspl[0]),
                            int(lspl[1]),
                            int(lspl[4])+self.n_alpha,
                            int(lspl[5])+self.n_alpha])
    elif sgl_found:
        lspl = l.split()
        if len(lspl) == 4:
            singles.append([lspl[-1],
                            int(lspl[0]),
                            int(lspl[2])+self.n_alpha])
    if 'RESULTS' in l:
        if not dbl_found:
            raise Exception('Double excitations not found!')
        break

    if self.WF_type is None:
        raise Exception('We found no wave function in ' + file_name)

    elif self.WF_type == 'MRCI':
        self.load_MRCI_WF(ref, singles, doubles)

    def load_MRCI_WF(self, C0, Csgl, Cdbl):
        """Load the MRCISD wave function."""
        sqrt_2 = math.sqrt(2.0)
        logger.debug('ref coef: %f', C0)
        if logger.level <= logging.DEBUG:
            logmsg = []
            logmsg.append('Singly exc:')
            for i in Csgl:
                logmsg.append(i)
            logmsg.append('-------------------\n')
            logmsg.append('Doubly exc:')
            for i in Cdbl:
                logmsg.append(i)
            logmsg.append('-------------------\n')
            logger.debug('\n'.join(logmsg))
        raise Exception ('load_MRCI_WF is not implemented correctly!')

        for det in self.determinants:
            if rank_of_exc(det) == 0:
                det[0] = float(C0)
            if rank_of_exc(det) == 1:
                occ_of_occ = [0] * self.n_alpha
                ext_occ = None
                for i in range(self.n_alpha):
                    if (i+1) in det[1:self.n_alpha+1]:
                        occ_of_occ[i] += 1
                    if (i+1) in det[self.n_alpha+1:]:
                        occ_of_occ[i] += 1
                orb_ini = occ_of_occ.index(1)+1
                occ_of_occ = ''.join(map(str, occ_of_occ)).replace('1','\\')
                if det[self.n_alpha] > self.n_alpha:
                    ext_occ = det[self.n_alpha]
                if det[-1] > self.n_alpha:
                    if ext_occ is not None:
                        raise Exception(
                            'Found more than one externally occ orb in single excitation!')
                    else:
                        ext_occ = det[-1]

                for s in Csgl:
                    if s[1] == occ_of_occ and s[2] == ext_occ:
                        det[0] = (float(s[0])/sqrt_2 if (self.n_alpha+orb_ini)%2==0 else
                                  -float(s[0])/sqrt_2)
                        logger.debug('Replacing coef of singles: %s %s', str(det), str(s))
                        break

            if rank_of_exc(det) == 2:
                exc_descr = [] # occ1, occ2, virt1, virt2, NP (-1 => excitation from same spin)
                exc_type = ''
                for i in range(self.n_alpha):
                    if (i+1) not in det[1:self.n_alpha+1]:
                        exc_descr.append(i+1)
                        exc_type += 'a'
                for i in range(self.n_alpha):
                    if (i+1) not in det[self.n_alpha+1:]:
                        exc_descr.append(i+1)
                        exc_type += 'b'
                for i in det[1:]:
                    if i > self.n_alpha:
                        exc_descr.append(i)
                if len(exc_descr) != 4:
                    raise Exception('Len of exc_descr is not 4 for double excitation.')
                exc_descr.append(exc_type)

                logger.debug('This det: %s; exc_descr = %s',
                             str(det), str(exc_descr))
                for d in Cdbl:
                    if (set(d[1:3]) == set(exc_descr[0:2]) and
                        set(d[3:5]) == set(exc_descr[2:4])):
                        logger.debug('Replacing coef of doubles: %s: %s = %f',
                                     str(det), exc_descr, d)
                        if d[1] == d[2] and d[3] == d[4]:
                            det[0] += float(d[0])
                        elif d[1] == d[2]:
                            det[0] += float(d[0])
                        elif d[3] == d[4]:
                            det[0] += (float(d[0])/sqrt_2 if (d[1] + d[2])%2 == 0
                                       else -float(d[0])/sqrt_2)
                        else:
                            if exc_descr[4] in ['aa', 'bb']:
                                if d[5] == -1:
                                    det[0] += (float(d[0])/2 if (d[1] + d[2])%2 == 1
                                               else -float(d[0])/2)
                            elif exc_descr[4] == 'ab':
                                if d[5] == 1:
                                    det[0] += (float(d[0])/2 if (d[1] + d[2])%2 == 0
                                               else -float(d[0])/2)
                                else:
                                    det[0] += (-float(d[0])/2 if (d[1] + d[2])%2 == 0
                                               else float(d[0])/2)






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



##
## W don't need to load these cases in the FCI class:
##
def load_CISD_WF(self, Csgl, Cdbl):
    """Load the CISD wave function, written as CCSD ansatz"""
    if logger.level <= logging.DEBUG:
        logmsg = ['Amplitudes of single excitations:']
        for i in Csgl:
            logmsg.append(str(i))
        logmsg.append('-------------------\n')
        logmsg.append('Amplitudes of double excitations:')
        for i in Cdbl:
            logmsg.append(str(i))
            logmsg.append('-------------------\n')
        logger.debug('\n'.join(logmsg))
    for det in self.determinants:
        rk = rank_of_exc(det)
        if rk == 0:
            det[0] = 1.0
        if rk == 1 or rk == 2:
            exc_descr = [] # occ1[, occ2], virt1[, virt2]
            exc_type = ''
            for i in range(self.n_alpha):
                if (i+1) not in det[1:self.n_alpha+1]:
                    exc_descr.append(i+1)
                    exc_type += 'a'
            for i in range(self.n_alpha):
                if (i+1) not in det[self.n_alpha+1:]:
                    exc_descr.append(i+1)
                    exc_type += 'b'
            for i in det[1:]:
                if i > self.n_alpha:
                    exc_descr.append(i)
            logger.debug('This det: %s; exc_descr = %s ; exc_type = %s',
                         str(det), str(exc_descr), exc_type)
            if rk == 1:
                if len(exc_descr) != 2:
                    raise Exception('Length of exc_descr is not 2 for single excitation.')
                for s in Csgl:
                    if s[1:] == exc_descr:
                        det[0] = float(s[0]) if (self.n_alpha+s[1])%2 == 0 else -float(s[0])
                        break
            if rk == 2:
                if exc_type == 'ab':
                    ai_are_both_larger = ((exc_descr[0] > exc_descr[1])
                                           == (exc_descr[2] > exc_descr[3]))
                if len(exc_descr) != 4:
                    raise Exception('Length of exc_descr is not 4 for double excitation.')
                for d in Cdbl:
                    if (set(d[1:3]) == set(exc_descr[0:2]) and
                        set(d[3:5]) == set(exc_descr[2:4])):
                        if d[1] == d[2] and d[3] == d[4]:
                            det[0] += float(d[0])
                        elif d[1] == d[2]:
                            det[0] += float(d[0])/2
                        elif d[3] == d[4]:
                            det[0] += float(d[0]) if (d[1] + d[2])%2 == 0 else -float(d[0])
                        else:
                            if exc_type in ['aa', 'bb']:
                                if d[3] < d[4]:
                                    det[0] += float(d[0]) if (d[1] + d[2])%2 == 0 else -float(d[0])
                                else:
                                    det[0] += float(d[0]) if (d[1] + d[2])%2 == 1 else -float(d[0])
                            elif exc_type == 'ab':
                                if ai_are_both_larger == (d[3] > d[4]):
                                    det[0] += float(d[0]) if (d[1] + d[2])%2 == 0 else -float(d[0])
    self.normalise()

def load_CCSD_WF(self, Tsgl, Tdbl):
    """Load the CCSD wave function."""
    if logger.level <= logging.DEBUG:
        logmsg = []
        logmsg.append('Amplitudes of single excitations:')
        for i in Tsgl:
            logmsg.append(str(i))
        logmsg.append('-------------------\n')
        logmsg.append('Amplitudes of double excitations:')
        for i in Tdbl:
            logmsg.append(str(i))
        logmsg.append('-------------------\n')
        logger.debug('\n'.join(logmsg))
    raise Exception('load_CCSD_WF: Not implemented yet!')



def _get_norm_of_matrix(M):
    """Return norm of M
    
    If M = array,              returns sqrt(sum Mij)
    If M = (array_a, array_b), returns sqrt((sum M[0]ij + M[1]ij)/2)
    """
    norm = 0.0
    if isinstance(M, tuple):
        norm += _get_norm_of_matrix(M[0])
        norm += _get_norm_of_matrix(M[1])
        norm = norm/2
    else:
        for line in M:
            for M_ij in line:
                norm += M_ij**2
    return math.sqrt(norm)



def str_matrix(X):
    """Return a str of the 2D list or array X."""
    strM = []
    for i in X:
        strI = []
        for j in i:
            strI.append(' {0:10.6f} '.format(j)\
                        if abs(j) > 1.0E-7 else
                        (' ' + '-'*10 + ' '))
        strM.append(''.join(strI))
    return '\n'.join(strM)



def _get_orbitals_from_Molpro_output(output_name):
    """Load (last) orbitals from Molpro output.
    
    Parameters:
    -----------
    output_name (str)
        Molpro output
    
    Behaviour:
    ----------
    This function loads always the last RHF or UHF orbitals.
    
    Returns:
    --------
    See get_orbitals
    """
    reading_orbitals = False
    is_RHF = None
    n_orb = None
    coef_a = None
    coef_b = None
    logger.info('File: %s', output_name)
    with open(output_name, 'r') as f: 
        for l in f:
            if 'NUMBER OF CONTRACTIONS' in l:
                n_orb = int(l.split()[3])
                if n_orb > 10:
                    raise Exception ('Probably not valid for more than 10 orbitals')
            if 'MOLECULAR ORBITALS' in l:
                is_RHF = True
                coef_a = np.zeros((n_orb,n_orb))
                reading_orbitals = True
                cur_coef = coef_a
            if 'ELECTRON ORBITALS FOR POSITIVE SPIN' in l:
                is_RHF = False
                coef_a = np.zeros((n_orb,n_orb))
                reading_orbitals = True
                cur_coef = coef_a
            if 'ELECTRON ORBITALS FOR NEGATIVE SPIN' in l:
                is_RHF = False
                coef_b = np.zeros((n_orb,n_orb))
                reading_orbitals = True
                cur_coef = coef_b
            if 'HOMO' in l:
                reading_orbitals = False
            if reading_orbitals:
                lspl = l.split()
                if lspl:
                    is_line_orb = re.match('(\d+)\.\d+$', lspl[0])
                    if is_line_orb is not None:
                        for i in range(n_orb):
                            cur_coef[i][int(is_line_orb.group(1))-1] = float(l[24+i*10:24+(i+1)*10])
    logger.info('coeff_a:\n' + str(coef_a))
    if not is_RHF:
        logger.info('coeff_b:\n' + str(coef_b))
    if is_RHF:
        return coef_a
    else:
        return coef_a, coef_b





def str_Jac_Hess(J, H, wf):
    """Return str of Jacobian (J) and the Hessian (H) of wave function wf."""
    JHstr = []
    JHstr.append('Jacobian:')
    for i, x in enumerate(J):
        exc_from, exc_to = get_i_a_from_pos_in_jac(i, wf.n_alpha, wf.beta_shift)
        JHstr.append(('[{0:3d}] = {1:.6f}; spin = {2:s}; K_{3:d}^{4:d}'.\
                      format(i, x,
                             'a' if i<wf.beta_shift else 'b',
                             exc_from, exc_to)))
    JHstr.append('-'*50)
    JHstr.append('Hessian:')
    for i,II in enumerate(H):
        exc_from_i, exc_to_i = get_i_a_from_pos_in_jac(i, wf.n_alpha, wf.beta_shift)
        for j,x in enumerate(II): 
            exc_from_j, exc_to_j = get_i_a_from_pos_in_jac(j, wf.n_alpha, wf.beta_shift)
            JHstr.append(' [{0:3d},{1:3d}] = {2:.6f}; K_{3:d}^{4:d} [{5:s}]; K_{6:d}^{7:d} [{8:s}]'.\
                         format(i, j,  x,
                                exc_from_i, exc_to_i, 'a' if i<wf.beta_shift else 'b',
                                exc_from_j, exc_to_j, 'a' if j<wf.beta_shift else 'b'))
    return('\n'.join(JHstr))




def _get_Slater_Det_from_FCI_line(l, line_number, orb_dim, n_irrep, Ms):
    """only for Ms=0!!"""
    lspl = l.split()
    final_occ = []
    try:
        coeff = float(lspl[0])
        occ = list(map(lambda x: int(x)-1, lspl[1:]))
    except Exception as e:
        raise dGrMolproInputError("Error when reading FCI configuration. Exception was:\n"
                                  + str(e),
                                  line = l, line_number = line_number)
    n_cur_spirrep = 0
    cur_spirrep = 0
    n = 0
    sum_prev_orb_dim = 0
    print('orb_dim: ', orb_dim._occupation)
    print('my occ: ', occ)
    for i, orb in enumerate(occ):
        print('ini iter -->','i=',i,'orb=',orb)
        if orb >= orb_dim[cur_spirrep] or i == (len(occ) + int(2*Ms))//2:
            print('inside if:','i=', i,'orb=', orb, 'orbdim', orb_dim[cur_spirrep], 'n=',n)
            final_occ.append(np.array(list(map(lambda x:x-sum_prev_orb_dim, occ[i-n:i])), dtype=int))
            sum_prev_orb_dim += orb_dim[cur_spirrep]
            cur_spirrep += 1
            if i == (len(occ) + int(2*Ms))//2:
                sum_prev_orb_dim = 0
                while cur_spirrep < n_irrep:
                    final_occ.append(np.array([], dtype=int))
                    cur_spirrep += 1
            else:
                while orb > sum_prev_orb_dim:
                    print('adding empty:',
                          'orb=',orb,
                          'sum_prev_orb_dim=',sum_prev_orb_dim,
                          'cur_spirrep=',cur_spirrep)
                    final_occ.append(np.array([], dtype=int))
                    sum_prev_orb_dim += orb_dim[cur_spirrep]
                    cur_spirrep += 1
            n = 1
        else:
            n += 1
        print('end iter -->', final_occ)
    i += 1
    print(i,n,sum_prev_orb_dim)
    final_occ.append(np.array(list(map(lambda x:x-sum_prev_orb_dim, occ[i-n:i])), dtype=int))
    sum_prev_orb_dim += orb_dim[cur_spirrep]
    cur_spirrep += 1
    while cur_spirrep < 2 * n_irrep:
        final_occ.append(np.array(occ[0:0], dtype=int))
        cur_spirrep += 1
    return Slater_Det(c = float(lspl[0]),
                      occupation = final_occ)



    def compare_Jac_Hess(self, fout=sys.stdout, U=None):
        """Make Jacobian and Hessian analytically and numerically and compare.
        
        TODO: implement U, the basis
        """
        with logtime('Making Jacobian and Hessian') as T_Jac_Hess:
            Jac, Hess = self.make_Jac_Hess_overlap()
        with logtime('Making Jacobian and Hessian numerically') as T_Jac_Hess:
            num_Jac, num_Hess = self.construct_Jac_Hess(analytic=False)
        allclose_J = np.allclose(Jac, num_Jac)
        allclose_H = np.allclose(Hess, num_Hess)
        sep='\n' + '='*30 + '\n'
        sep2='\n' + '-'*30 + '\n'
        logger.info(sep
                    + 'Jacobian allclose: %s' + sep2
                    + 'Analytic Jacobian:\n%r\n' + sep2
                    + 'Numeric Jacobian:\n%r\n' + sep2
                    + 'Anal Jac == Num Jac:\n%r\n' + sep2
                    + 'Anal Jac - Num Jac:\n%r\n' + sep2,
                    allclose_J,
                    Jac, num_Jac, Jac == num_Jac, Jac - num_Jac)
        logger.info(sep
                    + 'Jacobian allclose: %s' + sep2
                    + 'Analytic Hessian:\n%r\n' + sep2
                    + 'Numeric Hessian:\n%r\n' + sep2
                    + 'Anal Hess == Num Hess:\n%r\n' + sep2
                    + 'Anal Hess - Num Hess:\n%r\n' + sep2,
                    allclose_J,
                    Hess, num_Hess, Hess == num_Hess, Hess - num_Hess)
        if allclose_H and allclose_H:
            fout.write('Success! Analytic and numeric Jac. and Hess. are the same!!\n')
        else:
            fout.write('\nWARNING! Comparisons of Jac. and Hess have failed!!\n')
