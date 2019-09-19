
def distance_to_det(wf, U, thresh_cI=1E-10, assume_orth = False):
    """Calculates the distance to the determinant U
     
    See dGr_FCI_Molpro.Molpro_FCI_Wave_Function.distance_to_det
     """
    if isinstance(U, tuple):
        Ua, Ub = U
    else:
        Ua = Ub = U
    for det in wf.all_dets():
        if abs(det.c) < thresh_cI:
            continue
        if isinstance(det, genWF.Ref_Det):
            f0_a = _calc_fI(Ua, get_I(wf.n_alpha))
            f0_b = _calc_fI(Ub, get_I(wf.n_beta))
            f = det.c * f0_a * f0_b
        else:
            try:
                f
            except NameError:
                raise NameError('First determinant has to be genWF.Ref_Det!')
        if isinstance(det, genWF.Singly_Exc_Det):
            if det.spin > 0:
                f += det.c * _calc_fI(Ua, get_I(wf.n_alpha, det.i, det.a)) * f0_b
            else:
                f += det.c * _calc_fI(Ub, get_I(wf.n_beta, det.i, det.a)) * f0_a
        elif isinstance(det, genWF.Doubly_Exc_Det):
            if det.spin_ia * det.spin_jb < 0:
                f += (det.c
                      * _calc_fI(Ua, get_I(wf.n_alpha, det.i, det.a))
                      * _calc_fI(Ub, get_I(wf.n_beta,  det.j, det.b)))
            elif det.spin_ia > 0:
                f += (det.c * f0_b
                      * _calc_fI(Ua, get_I(wf.n_alpha,
                                                [det.i, det.j],
                                                sorted([det.a, det.b]))))
            else:
                f += (det.c * f0_a
                      * _calc_fI(Ub, get_I(wf.n_beta,
                                                [det.i, det.j],
                                                sorted([det.a, det.b]))))
    if not assume_orth:
        Da = linalg.det(np.matmul(Ua.T, Ua))
        Db = linalg.det(np.matmul(Ub.T, Ub))
        f /= math.sqrt(Da * Db)
    return f


def get_ABC_matrices(wf, U, thresh_cI=1E-10):
    """Calculates the arrays A,B,C needed for Absil's algorithm

    See dGr_FCI_Molpro.Molpro_FCI_Wave_Function.distance_to_det
    """
    if isinstance(U, tuple):
        Ua, Ub = U
        restricted = False
    else:
        Ua = Ub = U
        restricted = True
    K, na = Ua.shape
    nb = Ub.shape[1]
    if na != nb:
        raise NotImplementedError('We need both Ua and Ub with same shape!')
    if K != wf.orb_dim:
        raise ValueError('First dimension of U must match orb_dim!')
    if na != wf.n_alpha:
        raise ValueError('Second dimension of Ua must match the n_alpha!')
    if nb != wf.n_beta:
        raise ValueError('Second dimension of Ua must match the n_beta!')
    ABshape = Ua.shape*2
    Aa_a = np.zeros(ABshape)
    Aa_b = np.zeros(ABshape)
    Ba = np.zeros(ABshape)
    Ca = np.zeros(Ua.shape)
    if not restricted:
        Ab_a = np.zeros(ABshape)
        Ab_b = np.zeros(ABshape)
        Bb = np.zeros(ABshape)
        Cb = np.zeros(Ub.shape)
    for det in wf.all_dets():
        if abs(det.c) < thresh_cI:
            continue
        if isinstance(det, genWF.Ref_Det):
            Ia = range(wf.n_alpha)
            Ib = range(wf.n_beta)
        elif isinstance(det, genWF.Singly_Exc_Det):
            if det.spin > 0:
                Ia = get_I(wf.n_alpha, det.i, det.a)
                Ib = range(wf.n_beta)
            else:
                Ia = range(wf.n_alpha)
                Ib = get_I(wf.n_beta, det.i, det.a)
        elif isinstance(det, genWF.Doubly_Exc_Det):
            if det.spin_ia * det.spin_jb < 0:
                Ia = get_I(wf.n_alpha, det.i, det.a)
                Ib = get_I(wf.n_beta,  det.j, det.b)
            elif det.spin_ia > 0:
                Ia = get_I(wf.n_alpha,
                           [det.i, det.j],
                           sorted([det.a, det.b]))
                Ib = range(wf.n_beta)
            else:
                Ia = range(wf.n_alpha)
                Ib = get_I(wf.n_beta,
                           [det.i, det.j],
                           sorted([det.a, det.b]))
        Fa = _calc_fI(Ua, Ia)
        Fb = _calc_fI(Ub, Ib)
        Proj_a = np.identity(K)
        Ga = np.zeros(Ua.shape)
        if not restricted:
            Gb = np.zeros(Ub.shape)
            Proj_b = np.identity(K)
        for k in range(K):
            for l in range(na):
                Hkl_a = np.zeros(Ua.shape)
                if not restricted:
                    Hkl_b = np.zeros(Ub.shape)
                for i in range(K):
                    Proj_a[i,k] -= np.dot(Ua[i,:], Ua[k,:])
                    if not restricted:
                        Proj_b[i,k] -= np.dot(Ub[i,:], Ub[k,:])
                    for j in range(na):
                        if j != l:
                            Hkl_a[i,j] = _calc_H(Ua, Ia, i, j, k, l)
                            if not restricted:
                                Hkl_b[i,j] = _calc_H(Ub, Ib, i, j, k, l)
                        else:
                            Ba[i,j,k,l] += det[0] * Fa * Fb * Proj_a[i,k]
                            if not restricted:
                                Bb[i,j,k,l] += det[0] * Fa * Fb * Proj_b[i,k]
                Ga[k,l] = _calc_G(Ua, Ia, k, l)
                Gb[k,l] = _calc_G(Ub, Ib, k, l)
#                 Ga[k,l] += np.dot(Hkl_a[:,na-1], Ua[:,na-1])
#                 if restricted:
#                     Gb[k,l] = _calc_G(Ub, Ib, k, l)
#                 else:
#                     Gb[k,l] += np.dot(Hkl_b[i,nb-1], Ub[i,nb-1])
                Aa_a[k,l,:,:] += det[0] * Fb * np.matmul(Proj_a, Hkl_a)
                if not restricted:
                    Ab_b[k,l,:,:] += det[0] * Fa * np.matmul(Proj_b, Hkl_b)
        det_G_FU = det[0] * (Ga - Fa * Ua)
        Ca += Fb * det_G_FU
        Aa_b += np.multiply.outer(det_G_FU, Gb)
        if not restricted:
            det_G_FU = det[0] * (Gb - Fb * Ub)
            Cb += Fa * det_G_FU
            Ab_a += np.multiply.outer(det_G_FU, Ga)
    if restricted:
        return (Aa_a, Aa_b), Ba, Ca
    else:
        return ((Aa_a, Aa_b),
                (Ab_a, Ab_b)), (Ba, Bb), (Ca, Cb)

def generate_lin_system_from_ABC(A, B, C, U):
    """Given the matrices A, B, C, reshape to get the linear system
    
    Behaviour:
    
    From the matrices A, B, C, reshape them to get the
    matrix (a 2d array) B_minus_A and the vector (a 1D array) C,
    such that
    
    B_minus_A @ eta = C
    
    is the linear system to calculate eta, in the step of
    Absil's Newton-Grassmann optimisation
    
    Limitations:
    
    It assumes that the number of alpha and beta electrons are the same
    
    Parameters:
    
    A   (2-tuple of 4D array, for rescricted)
        (A^a_a, A^a_b) = (A_same, A_mix)
        
        (2-tuple of 2-tuples of 4D arrays, for unrestricted)
        ((A^a_a, A^a_b),
         (A^b_a, A^b_b))
    
    B   (4D array, for rescricted)
        B^a
        (2-tuple of 4D array, for unrescricted)
        (B^a, B^b)
    
    C   (2D array, for rescricted)
        C^a
        (2-tuple of 2D array, for unrescricted)
        (C^a, C^b)
    
    U   (2D array, for rescricted)
        Ua
        (2-tuple of 2D array, for unrescricted)
        (Ua, Ub)
        The transformation matrix in this iteration
        
    Return:
    
    The 2D array A_minus_B and the 1D array Cn
    """
    restricted = not isinstance(C, tuple)
    # n = nubmer of electrons
    # K = nubmer of orbitals
    # nK = nubmer of electrons times the number of spatial orbitals
    if restricted:
        K = A[0].shape[0]
        n = A[0].shape[1]
    else:
        K = A[0][0].shape[0]
        n = A[0][0].shape[1]
    nK = n*K
    # test all entries and shapes?
    if restricted:
        Cn = np.zeros(nK + n)
        Cn[:nK] = np.ravel(C,order='C')
    else:
        Cn = np.zeros(2*(nK + n))
        Cn[:2*nK] = np.concatenate((np.ravel(C[0], order='C'),
                                    np.ravel(C[1], order='C')))
    if restricted:
        B_minus_A = np.zeros((nK + n, nK))
        B_minus_A[:nK,:] = np.reshape(B, (nK, nK), order='C')
        B_minus_A[:nK,:] -= np.reshape(A[0], (nK, nK), order='C')
        B_minus_A[:nK,:] -= np.reshape(A[1], (nK, nK), order='C')
        # --> Extra term due to normalisation
        B_minus_A[:nK,:] += 2*np.multiply.outer(Cn[:nK], np.ravel(U, order='C'))
        # --> Terms to guarantee orthogonality to U
        B_minus_A[nK:,:] += U.T
    else:
        B_minus_A = np.zeros((2*(nK + n), 2*nK))
        B_minus_A[:nK, :nK] = np.reshape(B[0],
                                         (nK, nK),
                                         order='C')
        B_minus_A[:nK, :nK] -= np.reshape(A[0][0],
                                          (nK, nK),
                                          order='C')
        B_minus_A[nK:2*nK, nK:] = np.reshape(B[1],
                                         (nK, nK),
                                         order='C')
        B_minus_A[nK:2*nK, nK:] -= np.reshape(A[1][1],
                                          (nK, nK),
                                          order='C')
        B_minus_A[:nK, nK:] -= np.reshape(A[0][1],
                                          (nK, nK),
                                          order='C')
        B_minus_A[nK:2*nK, :nK] -= np.reshape(A[1][0],
                                          (nK, nK),
                                          order='C')
        # --> Extra term due to normalisation
        B_minus_A[:nK, :nK] += np.multiply.outer(Cn[:nK],
                                                 np.ravel(U[0], order='C'))
        B_minus_A[:nK, nK:] += np.multiply.outer(Cn[:nK],
                                                 np.ravel(U[1], order='C'))
        B_minus_A[nK:2*nK, :nK] += np.multiply.outer(Cn[nK:2*nK],
                                                 np.ravel(U[0], order='C'))
        B_minus_A[nK:2*nK, nK:] += np.multiply.outer(Cn[nK:2*nK],
                                                 np.ravel(U[1], order='C'))
        # --> Terms to guarantee orthogonality to U
        ## Can be made more efficiente if order = 'F' is used!!!
        for iel in range(n):
            for iorb in range(K):
                B_minus_A[2*nK     + iel,      iel + n*iorb] = U[0][iorb,iel]
                B_minus_A[2*nK + n + iel, nK + iel + n*iorb] = U[1][iorb,iel]
    return B_minus_A, Cn
