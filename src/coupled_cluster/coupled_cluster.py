""" CCSD closed shell
"""

import numpy as np


def make_u(wf,no,nv):
    """Generate the u matrix:
        u_{ij}^{ab}=2t_{ij}^{ab}-t_{ji}^{ab}

    Parameters:
    -----------

    wf (IntermNormWaveFunction)
        A wave function containing the double excitation amplitudes Matrix

    Return:
    -------

    u (1D np.ndarray)

    """
    u = np.zeros(len(wf.amplitudes[no*nv:]))
    for ij in range(no*(no+1)//2):
        for a in range(nv):
            for b in range(nv):
                u[ij*nv**2 + a*nv + b] = 2*wf.amplitudes[no*nv+ij*nv**2 + a*nv + b] - wf.amplitudes[no*nv+ij*nv**2 + b*nv + a]
    return u

def make_L(g,n):
    """Generate the intermediary L matrix:
          L_{pqrs} = 2*g_{pqrs}-g_{psrq}
       This implementation does suport t1-transformed matrices.

    Parameters:
    -----------

    g (1D np.ndarray)
       The two-electron integral matrix in atomic or molecular basis set.

    n (int)
       Numbers total of orbitals.

    Return:
    -------

    L (1D np.ndarray)

    """
    #TODO: Test the results
    L = 2*g
    pq = -1
    for p in range(n):
        for q in range(n):
            if p < q:
                continue
            pq += 1
            rs = -1
            for r in range(n):
                for s in range(n):
                    if r < s:
                        continue
                    rs += 1
                    if pq < rs:
                        continue
                    pqrs = (rs + pq * (pq + 1) // 2
                            if pq >= rs else
                            pq + rs * (rs + 1) // 2)
                    ps = s + p * (p + 1) // 2 if p >= s else p + s * (s + 1) // 2
                    rq = q + r * (r + 1) // 2 if r >= q else r + q * (q + 1) // 2
                    psrq = (rq + ps * (ps + 1) // 2
                            if ps >= rq else
                            ps + rq * (rq + 1) // 2)
                    L[pqrs]-= L[psrq]
    return L 

def make_L_ooov(g,no,nv):
    """Generate a piece of the intermediary L matrix:
          L_{ijka} = 2*g_{ijka}-g_{iakj}
       There is one symmetry in this matrix:
          L_{ijka} = L_{jika}
       This implementation does suport t1-transformed matrices.

    Parameters:
    -----------

    g (1D np.ndarray)
       The two-electron integral matrix in atomic or molecular basis set.

    no (int)
       Numbers of occupied orbitals.

    nv (int)
       Numbers of virtual orbitals.

    Return:
    -------

    L (1D np.ndarray)

    """
    #TODO: Test the results
    for ij in range(no*(no+1)//2):
        i,j = divmod(ij,no)
        if j < i:
            j = i + j
        for k in range(no):
            kj = k + j * (j + 1) // 2 if j >= k else j + k * (k + 1) // 2
            for a in range(nv):
                ka = k + (a + no) * (a + no + 1) // 2
                ia = i + (a + no) * (a + no + 1) // 2
                ijka = ij + ka * (ka + 1) // 2
                iakj = kj + ia * (ia + 1) // 2
                L[ij*no*nv+k*nv+a] = 2*g[ijka] - g[iakj]
    return L

def make_L_ovov(g,no,nv):
    """Generate a piece of the intermediary L matrix:
          L_{iajb} = 2*g_{iajb}-g_{ibja}
       There is one symmetry in this matrix:
          L_{iajb} = L_{jbia}
       This implementation does suport t1-transformed matrices.

    Parameters:
    -----------

    g (1D np.ndarray)
       The two-electron integral matrix in atomic or molecular basis set.

    no (int)
       Numbers of occupied orbitals.

    nv (int)
       Numbers of virtual orbitals.

    Return:
    -------

    L (1D np.ndarray)

    """
    #TODO: Test the results
    L = np.zeros(no*(no+1)*nv**2//2)
    for ij in range(no*(no+1)//2):
        i,j = divmod(ij,no)
        if j < i:
            j = i + j
        for a in range(nv):
            ia = i + (a + no) * (a + no + 1) // 2
            for b in range(nv):
                jb = j + (b + no) * (b + no + 1) // 2
                ib = i + (b + no) * (b + no + 1) // 2
                ja = j + (a + no) * (a + no + 1) // 2
                ab = b + no + (a + no) * (a + no + 1) // 2 if a >= b else a + no + (b + no) * (b + no + 1) // 2
                iajb = jb + ia * (ia + 1) // 2
                ibja = ib + ja * (ja + 1) // 2 if ja >= ib else ja + ib * (ib + 1) // 2
                L[ij*nv**2+a*nv+b] = 2*g[iajb]-g[ibja]
    return L

def make_F(h,g,no):
    """Generate the Fock inactive matrix:
          F_{mn} = h_{mn}+\sum_{l}(2*g_{mnll}-g_{mlln})
       This implementation does suport t1-transformed matrices.

    Parameters:
    -----------

    h (2d np.array)
       The one-electron integral matrix in molecular basis set.

    g (1d np.array)
       The two-electron integral matrix in molecular basis set.
       Only the non-symmetric terms.

    Return:
    -------

    F (2d np.array)
      The complete Fock matrix
    
    """
    #TODO: Do more tests
    n = len(h)
    F = h
    for p in range(n):
        for q in range(n):
            pq = q + p * (p + 1) // 2 if p >= q else p + q * (q + 1) // 2
            for l in range(no):
                ll = l + l * (l + 1) // 2 
                pqll = (ll + pq * (pq + 1) // 2
                        if pq >= ll else
                        pq + ll * (ll + 1) // 2)
                pl = l + p * (p + 1) // 2 if p >= l else p + l * (l + 1) // 2
                lq = q + l * (l + 1) // 2 if l >= q else l + q * (q + 1) // 2
                pllq = (lq + pl * (pl + 1) // 2
                        if pl >= lq else
                        pl + lq * (lq + 1) // 2)
                F[p,q]+= 2*g[pqll]-g[pllq]
    return F


def energy(wf,Eref,g):
    """Calculates the Coupled-Cluster energy

    Parameters:
    -----------

    wf (IntermNormWaveFunction)
        A wave function containing the single and double excitation amplitudes matrices.

    Eref (float)
        The reference energy

    g (1d np.array)
       The two-electron integral matrix in molecular basis set.
       Only the non-symmetric terms.

    Returns:
    --------

    CC_E (float)
        The coupled-Cluster energy
    """
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    L =  make_L_ovov(g,no,nv) 
    corr_E = 0
    for ij in range(no*(no+1)//2):
        i,j = divmod(ij,no)
        if j < i:
            j = i + j
        for a in range(nv):
            ia = i + (a + no) * (a + no + 1) // 2
            for b in range(nv):
                jb = j + (b+ no) * (b + no + 1) // 2
                if i == j and a == b:
                    corr_E+= (wf.amplitudes[no*nv+ij*nv**2+a*nv+b]+wf.amplitudes[i*nv+a]*wf.amplitudes[j*nv+b])*L[ij*nv**2+a*nv+b]
                else:
                    corr_E+= 2*(wf.amplitudes[no*nv+ij*nv**2+a*nv+b]+wf.amplitudes[i*nv+a]*wf.amplitudes[j*nv+b])*L[ij*nv**2+a*nv+b]
    return Eref + corr_E

 
def test_conv(omega,threshold=1.0E-8):
    """Check if all residuals are below the the threshold.
       If their are, returns True.
       If at least one if above the threshold, returns False.
       If None return False

    Parameters:
    -----------

    omega (1D np.ndarray)
        Matrix containing all one- and two-electron residual values

    threshold (float, optional, default=1.0E-8)
        Threshold to the residual values, if all values are belows it the 
calculation converged.

    Return:
    -------
    
    conv_status (bool)
        If True, the calculation converged.
        If False, does not converged.

    """
    if omega is None:
        return False
    for i in range(len(omega)):
        if abs(omega[i]) > threshold:
            return False
    return True
 
def t1_1e_transf_oo(matrix,wf):
    """ t1-tranformation for one-electron integrals (h, F or S matrix)
        running both indexes in the occ orbitals.
    """
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    t1_matrix = np.zeros((no,no))
    for i in range(no):
        for j in range(no):
            t1_matrix[i,j] = matrix[i,j]
            for c in range(nv):
                t1_matrix[i,j] += matrix[i,c+no]*wf.amplitudes[j*nv+c]
    return t1_matrix

def t1_1e_transf_vo(matrix,wf):
    """ t1-tranformation for one-electron integrals (h, F or S matrix)
        running the index in the virtual and the second in the occ orbitals.
    """
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    t1_matrix = np.zeros((nv,no))
    for i in range(no):
        for a in range(nv):
            t1_matrix[a,i] = matrix[a+no,i]
            for k in range(no):
                t1_matrix[a,i]-= matrix[k,i]*wf.amplitudes[k*nv+a]
                tmp = 0
                for c in range(nv):
                    tmp+= matrix[k,c+no]*wf.amplitudes[i*nv+c]
                t1_matrix[a,i]-= tmp*wf.amplitudes[k*nv+a] 
            for c in range(nv):
                t1_matrix[a,i]+= matrix[a+no,c+no]*wf.amplitudes[i*nv+c]
    return t1_matrix 
     
def t1_1e_transf_vv(matrix,wf):
    """ t1-tranformation for one-electron integrals (h, F or S matrix)
        running both indexes in the virtual orbitals.
    """
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    t1_matrix = np.zeros((nv,nv))
    for a in range(nv):
        for b in range(nv):
            t1_matrix[a,b] = matrix[a+no,b+no]
            for k in range(no):
                t1_matrix[a,b] -= matrix[k,b+no]*wf.amplitudes[k*nv+a]
    return t1_matrix

def t1_2e_transf_oooo(matrix,wf):
    """ t1-tranformation for two-electron integrals (g or L matrix).
        There is the symmetry ijkl = klij.
    """
    #TODO: Test the results
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    t1_matrix = np.zeros((no**4+no**2)//2)
    count = -1
    for ij in range(no**2):
        i,j = divmod(ij,no)
        for kl in range(no**2):
            if ij >= kl:
                count+= 1
                k,l = divmod(kl,no)
                ijkl = kl + ij * (ij + 1) // 2
                mn = j + i * (i + 1) // 2 if i >= j else i + j * (j + 1) // 2
                pq = l + k * (k + 1) // 2 if k >= l else k + l * (l + 1) // 2
                mnpq = (pq + mn * (mn + 1) // 2
                        if mn >= pq else
                        mn + pq * (pq +1) // 2)
                t1_matrix[ijkl]+= matrix[mnpq]
                for e in range(nv):
                    ie = i + (e + no)*(e + no + 1)//2
                    iekl = pq + ie * (ie + 1) // 2
                    t1_matrix[ijkl]+= matrix[iekl]*wf.amplitudes[j*nv+e]
                    ke = k + (e + no)*(e + no +1)//2
                    tmp = 0
                    for f in range(nv):
                        kf = k + (f + no)*(f + no +1)//2
                        iekf = (kf + ie * (ie + 1) // 2
                                if ie >= kf else
                                ie + kf * (kf +1) // 2)
                        tmp+= matrix[iekf]*wf.amplitudes[j*nv+e]
                    ijke = mn + ke * (ke + 1) // 2
                    t1_matrix[ijkl]+= (matrix[ijke]+tmp)*wf.amplitudes[l*nv+e]
    return t1_matrix

def t1_2e_transf_ooov(matrix,wf):
    """ t1-tranformation for two-electron integrals (g or L matrix).
        There is no symmetry whitin this matrix. It is equal to the ovoo matrix.
    """
    #TODO: Test the results
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    t1_matrix = np.zeros((no,nv,nv,nv))
    for i in range(no):
        for j in range(no):
            for k in range(no):
                for a in range(nv):
                    ij = i + j * (j + 1) // 2 if i >= j else j + i * (i + 1) //2
                    ka = k + (a + no) * (a + no + 1) // 2
                    ijka = ij + ka * (ka + 1) // 2
                    t1_matrix[i,j,k,a]+= matrix[ijka]
                    for e in range(nv):
                        ie = i + (e + no) * (e + no + 1) // 2
                        ieka = (ka + ie * (ie + 1) // 2
                                if ie >= ka else
                                ie + ka * (ka + 1) // 2)
                        t1_matrix[i,j,k,a]+= wf.amplitudes[j*nv+e]*matrix[ieka]

    return t1_matrix

def t1_2e_transf_oovv(matrix,wf):
    """ t1-tranformation for two-electron integrals (g or L matrix).
        There is no symmetry whitin this matrix. It is equal to the vvoo matrix.
    """
    #TODO: Test the results
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    t1_matrix = np.zeros((no,no,nv,nv))
    for i in range(no):
        for j in range(no):
            ij = i + j * (j + 1) // 2 if j >= i else j + i * (i + 1) // 2
            for a in range(nv):
                for b in range(nv):
                    ab = b + no + (a + no) * (a + no + 1) // 2 if a >= b else a + no + (b + no) * (b + no + 1) // 2
                    ijab = ij + ab * (ab + 1) // 2
                    t1_matrix[i,j,a,b] = matrix[ijab]
                    for e in range(nv):
                        ie = i + (e + no) * (e + no + 1) // 2
                        ieab = ie + ab * (ab + 1) // 2
                        tmp = 0
                        for l in range(no):
                            lb = l + (b + no) * (b + no + 1) // 2
                            le = l + (e + no) * (e + no + 1) // 2
                            if e == 0:
                                ijlb = ij + lb * (lb + 1) // 2
                                t1_matrix[i,j,a,b]-= matrix[ijlb]*wf.amplitudes[l*nv+a]
                            ielb = (lb + ie * (ie + 1) // 2
                                    if ie >= lb else
                                    ie + lb * (lb + 1) // 2)
                            tmp-= matrix[ielb]*wf.amplitudes[l*nv+a]
                        t1_matrix[i,j,a,b]+= (matrix[ieab] + tmp)*wf.amplitudes[j*nv+e]
    return t1_matrix

def t1_2e_transf_voov(matrix,wf):
    """ t1-tranformation for two-electron integrals (g or L matrix).
        There is no symmetry whitin this matrix. It is equal to the ovvo matrix.
    """
    #TODO: Test the results
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    t1_matrix = np.zeros((nv,no,no,nv))
    for i in range(no):
        for j in range(no):
            ij = i + j * (j + 1) // 2 if j >= i else j + i * (i + 1) // 2
            for a in range(nv):
                ai = i + (a + no) * (a + no + 1) // 2
                for b in range(nv):
                    jb = j + (b + no) * (b + no + 1) // 2
                    aijb = (jb + ai * (ai + 1) // 2
                            if ai >= jb else
                            ai + jb * (jb + 1) // 2)
                    t1_matrix[a,i,j,b] = matrix[aijb]
                    for e in range(nv):
                        tmp = 0
                        for l in range(no):
                            if e == 0:
                                li = l + i * (i + 1) // 2 if i >= l else i + l * (l + 1) // 2 
                                lijb = li + jb * (jb + 1) // 2
                                t1_matrix[a,i,j,b]-= matrix[lijb]*wf.amplitudes[l*nv+a]
                            le = l + (e + no) * (e + no + 1) // 2
                            lejb = (jb + le * (le + 1) // 2
                                    if le >= jb else
                                    le + jb * (jb + 1) // 2)
                            tmp-=matrix[lejb]*wf.amplitudes[l*nv+a]
                        ae = e + no + (a + no) * (a + no + 1) // 2 if a >= e else a + no + (e + no) * (e + no + 1) // 2 
                        aejb = jb + ae * (ae + 1) // 2
                        t1_matrix[a,i,j,b]+= (matrix[aejb] + tmp)*wf.amplitudes[i*nv+e]
    return t1_matrix 
                     

def t1_2e_transf_vovo(matrix,wf):
    """ t1-tranformation for two-electron integrals (g or L matrix).
        Symmetry: aibj == bjai. Only around half of the values must be calculated.
        Warning: This function returns a array with the same structure as Omega2.
    """
    #TODO: Test the results
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    t1_matrix = np.zeros((no+1)*no*nv**2//2)
    for ij in range(no*(no+1)//2):
        i,j = divmod(ij,no)
        if j < i:
            j = i + j
        for a in range(nv):
            ai = i + (a + no) * (a + no + 1) // 2
            for b in range(nv):
                bj = j + (b + no) * (b + no + 1) // 2
                aibj = (bj + ai * (ai + 1) // 2
                        if ai >= bj else
                        ai + bj * (bj + 1) // 2)
                t1_matrix[ij*nv**2+b*nv+a] = matrix[aibj]
                for e in range(nv): 
                    ae = e + no + (a + no) * (a + no + 1) // 2 if a >= e else a + no + (e + no) * (e + no + 1) // 2
                    be = e + no + (b + no) * (b + no + 1) // 2 if b >= e else b + no + (e + no) * (e + no + 1) // 2
                    aebj = bj + ae * (ae + 1)//2
                    t1_matrix[ij*nv**2+b*nv+a]+= matrix[aebj]*wf.amplitudes[i*nv+e]    
                    aibe = ai + be * (be + 1)//2
                    t1_matrix[ij*nv**2+b*nv+a]+= matrix[aibe]*wf.amplitudes[j*nv+e]    
                    for l in range(no):
                        li = i + l * (l + 1) // 2 if l >= i else l + i * (i + 1) // 2
                        lj = j + l * (l + 1) // 2 if l >= j else l + j * (j + 1) // 2
                        le = l + (e + no) * (e + no + 1) // 2
                        if e == 0:
                            libj = li + bj * (bj + 1) // 2
                            t1_matrix[ij*nv**2+b*nv+a]-= matrix[libj]*wf.amplitudes[l*nv+a]    
                            ailj = lj + ai * (ai + 1) // 2
                            t1_matrix[ij*nv**2+b*nv+a]-= matrix[ailj]*wf.amplitudes[l*nv+b]    
                        lebj = (bj + le * (le + 1)//2
                                if le >= bj else
                                le + bj * (bj + 1)//2)
                        t1_matrix[ij*nv**2+b*nv+a]-= matrix[lebj]*wf.amplitudes[i*nv+e]*wf.amplitudes[l*nv+a]   
                        aelj = lj + ae * (ae + 1 )//2
                        t1_matrix[ij*nv**2+b*nv+a]-= matrix[aelj]*wf.amplitudes[i*nv+e]*wf.amplitudes[l*nv+b]   
                        libe = li + be * (be + 1)//2
                        t1_matrix[ij*nv**2+b*nv+a]-= matrix[libe]*wf.amplitudes[j*nv+e]*wf.amplitudes[i*nv+a]   
                        aile = (ai + le * (le + 1)//2
                                if le >= ai else
                                le + ai * (ai + 1)//2)
                        t1_matrix[ij*nv**2+b*nv+a]-= matrix[aile]*wf.amplitudes[j*nv+e]*wf.amplitudes[l*nv+b]   
                        for k in range(no):
                            kj = j + k * (k + 1) // 2 if k >= j else k + j * (j + 1) // 2
                            ke = k + (e + no) * (e + no + 1) // 2
                            if e == 0:
                                likj = (kj + li * (li + 1) // 2
                                        if li >= kj else
                                        li + kj * (kj + 1) // 2)
                                t1_matrix[ij*nv**2+b*nv+a]+= matrix[likj]*wf.amplitudes[l*nv+a]*wf.amplitudes[k*nv+b]
                            lekj = kj + le * (le + 1 ) // 2
                            t1_matrix[ij*nv**2+b*nv+a]+= matrix[lekj]*wf.amplitudes[l*nv+a]*wf.amplitudes[k*nv+b]*wf.amplitudes[i*nv+e]
                            like = li + ke * (ke +1 ) // 2
                            t1_matrix[ij*nv**2+b*nv+a]+= matrix[like]*wf.amplitudes[l*nv+a]*wf.amplitudes[k*nv+b]*wf.amplitudes[j*nv+e]
                            for f in range(nv):
                                bf = f + no + (b + no) * (b + no + 1) // 2 if b >= f else b + no + (f + no) * (f + no + 1) // 2
                                lf = l + (f + no) * (f + no + 1) // 2
                                kf = k + (f + no) * (f + no + 1) // 2
                                if k == 0:
                                    if l == 0:
                                        aebf = (bf + ae * (ae + 1) // 2
                                                if ae >= bf else
                                                ae + bf * (bf + 1) // 2)
                                        t1_matrix[ij*nv**2+b*nv+a]+= matrix[aebf]*wf.amplitudes[i*nv+e]*wf.amplitudes[j*nv+f]
                                    lebf = le + bf * (bf + 1) // 2
                                    t1_matrix[ij*nv**2+b*nv+a]-= matrix[lebf]*wf.amplitudes[l*nv+a]*wf.amplitudes[i*nv+e]*wf.amplitudes[j*nv+f]
                                    aelf = lf + ae * (ae + 1) // 2
                                    t1_matrix[ij*nv**2+b*nv+a]-= matrix[aelf]*wf.amplitudes[l*nv+b]*wf.amplitudes[i*nv+e]*wf.amplitudes[j*nv+f]
                                lekf = (kf + le * (le + 1) // 2
                                        if le >= kf else
                                        le + kf * (kf + 1) // 2)
                                t1_matrix[ij*nv**2+b*nv+a]+= matrix[lekf]*wf.amplitudes[l*nv+a]*wf.amplitudes[k*nv+b]*wf.amplitudes[i*nv+e]*wf.amplitudes[j*nv+f]
    return t1_matrix
                                   
def t1_2e_transf_vovo_test(matrix,wf):
    """ t1-tranformation for two-electron integrals (g or L matrix).
        Symmetry: aibj == bjai. Only around half of the values must be calculated.
        Warning: This function returns a array with the same structure as Omega2.
        Other way to organize the multiplications and sums
    """
    #TODO: Test the results
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    t1_matrix = np.zeros((no+1)*no*nv**2//2)
    for ij in range(no*(no+1)//2):
        i,j = divmod(ij,no)
        if j < i:
            j = i + j
        for a in range(nv):
            ai = i + (a + no) * (a + no + 1) // 2
            for b in range(nv):
                bj = j + (b + no) * (b + no + 1) // 2
                aibj = (bj + ai * (ai + 1) // 2
                        if ai >= bj else
                        ai + bj * (bj + 1) // 2)
                t1_matrix[ij*nv**2+b*nv+a] = matrix[aibj]
                tmp_al = 0
                tmp_bl = 0
                for l in range(no):
                    li = i + l * (l + 1) // 2 if l >= i else l + i * (i + 1) // 2
                    lj = j + l * (l + 1) // 2 if l >= j else l + j * (j + 1) // 2
                    tmp_lk = 0
                    for k in range(no):
                        kj = j + k * (k + 1) // 2 if k >= j else k + j * (j + 1) // 2
                        tmp_kle = 0
                        if k == 0:
                            tmp_ale = 0
                            tmp_ble = 0
                        for e in range(nv): 
                            ke = k + (e + no) * (e + no + 1) // 2
                            ae = e + no + (a + no) * (a + no + 1) // 2 if a >= e else a + no + (e + no) * (e + no + 1) // 2
                            be = e + no + (b + no) * (b + no + 1) // 2 if b >= e else b + no + (e + no) * (e + no + 1) // 2
                            le = l + (e + no) * (e + no + 1) // 2
                            tmp_klef = 0
                            for f in range(nv):
                                kf = k + (f + no) * (f + no + 1) // 2
                                if k == 0:
                                    if l == 0:
                                        lekf = (kf + le * (le + 1) // 2
                                                if le >= kf else
                                                le + kf * (kf + 1) // 2)
                                        tmp_klef+= matrix[lekf]*wf.amplitudes[j*nv+f]
                            lekj = kj + le * (le + 1 ) // 2
                            tmp_kle+= (tmp_klef+matrix[lekj])*wf.amplitudes[i*nv+e]
                            like = li + ke * (ke +1 ) // 2
                            tmp_kle+= matrix[like]*wf.amplitudes[j*nv+e]
                            if k == 0 and l == 0:
                                tmp_ef = 0
                                for f in range(nv):
                                     bf = f + no + (b + no) * (b + no + 1) // 2 if b >= f else b + no + (f + no) * (f + no + 1) // 2
                                     aebf = (bf + ae * (ae + 1) // 2
                                             if ae >= bf else
                                             ae + bf * (bf + 1) // 2)
                                     tmp_ef+= matrix[aebf]*wf.amplitudes[j*nv+f]
                                aebj = bj + ae * (ae + 1) // 2
                                t1_matrix[ij*nv**2+b*nv+a]+= (matrix[aebj]+tmp_ef)*wf.amplitudes[i*nv+e]    
                                aibe = ai + be * (be + 1) // 2
                                t1_matrix[ij*nv**2+b*nv+a]+= matrix[aibe]*wf.amplitudes[j*nv+e]    
                            if k == 0:
                                tmp_alef = 0
                                tmp_blef = 0
                                for f in range(nv):
                                     lf = l + (f + no) * (f + no + 1) // 2
                                     bf = f + no + (b + no) * (b + no + 1) // 2 if b >= f else b + no + (f + no) * (f + no + 1) // 2
                                     lebf = le + bf * (bf + 1) // 2
                                     tmp_alef-= matrix[lebf]*wf.amplitudes[j*nv+f]
                                     aelf = lf + ae * (ae + 1) // 2
                                     tmp_blef-= matrix[aelf]*wf.amplitudes[j*nv+f]
                                lebj = (bj + le * (le + 1)//2
                                        if le >= bj else
                                        le + bj * (bj + 1)//2)
                                tmp_ale+= (tmp_alef - matrix[lebj])*wf.amplitudes[i*nv+e]
                                libe = li + be * (be + 1)//2
                                tmp_ale-= matrix[libe]*wf.amplitudes[j*nv+e]
                                aelj = lj + ae * (ae + 1 )//2
                                tmp_ble+= (tmp_blef - matrix[aelj])*wf.amplitudes[i*nv+e]
                                aile = (ai + le * (le + 1)//2
                                        if le >= ai else
                                        le + ai * (ai + 1)//2)
                                tmp_ble-= matrix[aile]*wf.amplitudes[j*nv+e]
                        likj = (kj + li * (li + 1) // 2
                                if li >= kj else
                                li + kj * (kj + 1) // 2)
                        tmp_lk+= (matrix[likj]+tmp_kle)*wf.amplitudes[k*nv+b]
                    libj = li + bj * (bj + 1) // 2
                    t1_matrix[ij*nv**2+b*nv+a]+= (tmp_lk+tmp_ale-matrix[libj])*wf.amplitudes[l*nv+a]    
                    ailj = lj + ai * (ai + 1) // 2
                    t1_matrix[ij*nv**2+b*nv+a]+= (tmp_ble-matrix[ailj])*wf.amplitudes[l*nv+b]    

    return t1_matrix
                                   
                               
    

def t1_2e_transf_ovvv(matrix,wf):
    """ t1-tranformation for two-electron integrals (g or L matrix).
        There is no symmetry whitin this matrix. It is equal to the vvov matrix.
    """
    #TODO: Test the results
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    t1_matrix = np.zeros((no,nv,nv,nv))
    for i in range(no):
        for a in range(nv):
            for b in range(nv):
                for c in range(nv):
                    ia = i + (a + no) * (a + no + 1) // 2
                    bc = c + no + (b + no) * (b + no + 1) // 2 if b >= c else b + no + (c + no) * (c + no + 1) // 2
                    iabc = ia + bc * (bc + 1) // 2
                    t1_matrix[i,a,b,c]+= matrix[iabc]
                    for l in range(no):
                        lc = l + (c + no) * (c + no + 1) // 2
                        ialc = (lc + ia * (ia + 1) // 2
                                if ia >= lc else
                                ia + lc * (lc + 1) // 2)
                        t1_matrix[i,a,b,c]-= wf.amplitudes[l*nv+b]*matrix[ialc]
                        
    return t1_matrix

def t1_2e_transf_vvvv(matrix,wf):
    """ t1-tranformation for two-electron integrals (g or L matrix).
        There is the symmetry abcd = cdab.
    """
    #TODO: Test the results
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    t1_matrix = np.zeros((nv**4+nv**2)//2)
   # t1_matrix = np.zeros((nv**4+2*nv**3+3*nv**2+2*nv)//8)
    for ab in range(nv**2):
        a,b = divmod(ab,nv)
        for cd in range(nv**2):
            if ab >= cd:
                abcd = cd + ab * (ab + 1) //2
                c,d = divmod(cd,nv)
                mn = b + no + (a + no) * (a + no + 1) // 2 if a >= b else a + no + (b + no) * (b + no + 1) // 2
                pq = d + no + (c + no) * (c + no + 1) // 2 if c >= d else c + no + (d + no) * (d + no + 1) // 2
                mnpq = (pq + mn * (mn + 1) // 2
                        if mn >= pq else
                        mn + pq * (pq +1) // 2)
                t1_matrix[abcd]+= matrix[mnpq]
                for l in range(no):
                    lb = l + (b + no)*(b + no + 1)//2
                    ld = l + (d + no) * (d + no +1) // 2
                    abld = ld + mn * (mn + 1) // 2
                    t1_matrix[abcd]-= matrix[abld]*wf.amplitudes[l*nv+c]
                    tmp = 0
                    for k in range(no):
                        kc = k + (c + no) * (c + no +1) // 2
                        lbkc = (kc + lb * (lb + 1) // 2
                                if lb >= kc else
                                lb + kc * (kc + 1) // 2)
                        tmp+= matrix[lbkc]*wf.amplitudes[k*nv+c]
                    lbcd = lb + pq * (pq + 1) // 2
                    t1_matrix[abcd]+= (tmp-matrix[lbcd])*wf.amplitudes[l*nv+a]
    return t1_matrix

def make_L_t1_voov(g_voov,g,wf):
    #TODO: Test the results
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    g_voov*= 2
    for i in range(no):
        for j in range(no):
            ij = i + j * (j + 1) // 2 if j >= i else j + i * (i + 1) // 2
            for a in range(nv):
                for b in range(nv):
                    ab = a + no + (b + no) * (b + no + 1) // 2 if b >= a else b + no + (a + no) * (a + no + 1) // 2
                    abij = ij + ab * (ab + 1) // 2
                    g_voov[a,i,j,b]-= g[abij]
                    for e in range(nv):
                        ie = i + (e + no) * (e + no + 1) // 2
                        tmp = 0
                        for l in range(no):
                            lb = l + (b + no) * (b + no + 1) // 2
                            if e == 0:
                                lbij = ij + lb * (lb + 1) // 2
                                g_voov[a,i,j,b]+= g[lbij]*wf.amplitudes[l*nv+a]
                            lbie = (lb + ie * (ie + 1) // 2
                                    if ie >= lb else
                                    ie + lb * (lb + 1) // 2)
                            tmp-= g[lbie]*wf.amplitudes[l*nv+a]
                        abie = ie + ab * (ab + 1) // 2
                        g_voov[a,i,j,b]-= (g[abie] + tmp)*wf.amplitudes[j*nv+e]
    return g_voov
                    
    

def equation(wf,F,g):
    """Calculate the residual from the singles and doubles configurations.

    Parameters:
    -----------
    
    wf (IntermNormWaveFunction)
        A wave function containing the single and double excitation amplitudes matrices.
    
    F (2d np.array)
       The one-electron Fock matrix in molecular basis set.

    g (1d np.array)
       The two-electron integral matrix in molecular basis set.
       Only the non-symmetric terms.

    Return:
    -------

    omega1 (1d np.array)
        Matrix containing all one electron residual values.

    omega2 (1d np.array)
        Matrix containing all two electron residual values.

    """
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    u = make_u(wf,no,nv)
    if wf.wf_type == 'CCSD':
        omega1 = _res_t1_singles(wf,u,F,g)
        omega2 = _res_t1_doubles(wf,u,F,g)
    elif cc_wf.wf_type == 'CCD':
        omega1 = 0
        omega2 = _res_doubles(wf,u,F,g)
    else:
        raise ValueError(
            'CC version '+cc_wf.level+' unknown.')
    
    return np.append(omega1,omega2),updater(omega1,omega2,F,no,nv)

def updater(omega1,omega2,F,no,nv):
    """Calculate the value to update eache amplitude for the next interation
    """
    update_val = np.zeros(no*nv+no*(no+1)*nv**2//2)
    for i in range(no):
        for a in range(nv):
            update_val[i*nv+a] = -omega1[i*nv+a]/(F[a+no,a+no]-F[i,i])
            for j in range(no):
                if j > i:
                    continue
                ij = j + i * (i + 1) // 2
                for b in range(nv):
                    update_val[no*nv+ij*nv**2+a*nv+b] = -omega2[ij*nv**2+a*nv+b]/(F[a+no,a+no]+F[b+no,b+no]-F[i,i]-F[j,j])
    return update_val

def _res_t1_singles(wf,u,F,g):
    """Calculate the residual from the singles using the T1-transformation.
    """
    #TODO: More tests
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    omega1 = t1_1e_transf_vo(F,wf).flatten()
    _res_sing_A1(omega1,t1_2e_transf_ovvv(g,wf),u,no,nv)
    _res_sing_B1(omega1,t1_2e_transf_ooov(g,wf),u,no,nv)
    _res_sing_C1(omega1,F,u,no,nv)
    return omega1

def _res_sing_A1(omega,g,u,no,nv):
    #TODO: Test the results
    for i in range(no):
        for a in range(nv):
            for k in range(no):
                ki = k + i * (i + 1) // 2 if i >= k else i + k * (k + 1) // 2
                for c in range(nv):
                    for d in range(nv):
                        omega[i*nv+a]+=u[ki*nv**2+c*nv+d]*g[k,c,a,d]

def _res_sing_B1(omega,g,u,no,nv):
    #TODO: Test the results
    for i in range(no):
        for a in range(nv):
            for k in range(no):
                for l in range(no):
                    kl = k + l * (l + 1) // 2 if l >= k else l + k * (k + 1) // 2
                    for c in range(nv):
                        omega[i*nv+a]-=u[kl*nv**2+a*nv+c]*g[k,i,l,c]

def _res_sing_C1(omega,F,u,no,nv):
    #TODO: Test the results
    for i in range(no):
        for a in range(nv):
            for k in range(no):
                ik = k + i * (i + 1) // 2 if i >= k else i + k * (k + 1) // 2
                for c in range(nv):
                    omega[i*nv+a]-=u[ik*nv**2+a*nv+c]*F[k,c+no]


def _res_t1_doubles(wf,u,F,g):
    """Calculate the residual from the doubles using the T1-transformation.
    """
    no = len(wf.ref_orb)//2
    nv = (len(wf.orb_dim)-len(wf.ref_orb))//2
    omega2 = t1_2e_transf_vovo(g,wf)
    _res_doub_A2pp(omega2,t1_2e_transf_vvvv(g,wf),wf,no,nv)
    _res_doub_B2(omega2,t1_2e_transf_oooo(g,wf),g,wf,no,nv)
    _res_doub_C2(omega2,t1_2e_transf_oovv(g,wf),g,wf,no,nv)
#    _res_doub_D2(omega2,t1_2e_transf_voov_from_L(make_L_voov(g),make_L_vvov(g),make_L_ooov(g),make_L_ovov(g),wf),make_L_ovov(g),u)
    ## We need more than the L_voov matrix for the t1-transf.
    ## Is it better to transform g or make more L "slices"?
    ## Needed: L_voov, L_vvov, L_ooov and L_ovov
    _res_doub_D2(omega2,make_L_t1_voov(t1_2e_transf_voov(g,wf),g,wf),make_L_ovov(g,no,nv),u,no,nv) #Maybe I can do all the transformations at once
    _res_doub_E2(omega2,t1_1e_transf_vv(F,wf),t1_1e_transf_oo(F,wf),g,wf,u,no,nv)
    return omega2

def _res_doub_A2pp(omega,gt1,wf,no,nv):
    #TODO: Test the results
    for ij in range(no*(no+1)//2):
        for ac in range(nv**2):
            for bd in range(nv**2):
                acbd = (ac + bd * (bd + 1) // 2
                        if bd >= ac else
                        bd + ac * (bd + 1) // 2)
                omega[ij*nv**2+ac//nv*nv+bd//nv]+= wf.amplitudes[no*nv+ij*nv**2+ac%nv*nv+bd%nv]*gt1[acbd]

                        
def _res_doub_B2(omega,gt1,g,wf,no,nv):
    #TODO: Test the results
    for ij in range(no*(no+1)//2):
        i,j = divmod(ij,no)
        if j < i:
            j = i + j
        for a in range(nv):
            for b in range(nv):
                for k in range(no):
                    for l in range(no):
                        tmp = 0
                        for c in range(nv):
                            for d in range(nv):
                                kc = k + (c + no) * (c  + no + 1) // 2
                                ld = l + (d + no) * (d + no + 1) // 2
                                kcld = (ld + kc * (kc + 1) // 2
                                        if kc >= ld else
                                        kc + ld * (ld + 1) // 2)
                                tmp+= g[kcld]*wf.amplitudes[no*nv+ij*nv**2+c*nv+d]
                        kl = k + l * (l + 1) // 2 if l >= k else l + k * (k + 1) // 2 
                        ki = i + k * (k + 1) 
                        lj = j + l * (l + 1)  
                        kilj = (lj + ki * (ki + 1) // 2
                                if ki >= lj else
                                ki + lj * (lj + 1) // 2)
                        omega[ij*nv**2+a*nv+b]+= (tmp + gt1[kilj])*wf.amplitudes[nv*no+kl*nv**2+a*nv+b]
 
def _res_doub_C2(omega,gt1,g,wf,no,nv):
    """ This function calculates the Omega_{ij}^{ab(C2)}+Omega_{ji}^{ba(C2)} term
    """
    #TODO: Test the results
    for ij in range(no*(no+1)//2):
        i,j = divmod(ij,no)
        if j < i:
            j = i + j
        for a in range(nv):
            for b in range(nv):
                if a != b and i != j:
                    for k in range(no):
                        ki = k + i * (i + 1) // 2 if i >= k else i + k * (k + 1) // 2 
                        kj = k + j * (j + 1) // 2 if j >= k else j + k * (k + 1) // 2 
                        for c in range(nv):
                            omega[ij*nv**2+a*nv+b]-= wf.amplitudes[no*nv+ki*nv**2+b*nv+c]*gt1[k,j,a,c]
                            omega[ij*nv**2+a*nv+b]-= wf.amplitudes[no*nv+kj*nv**2+a*nv+c]*gt1[k,i,b,c]
                            tmp_1 = 0
                            tmp_2 = 0
                            for l in range(no):
                                li = l + i * (i + 1) // 2 if i >= l else i + l * (l + 1) // 2 
                                lj = l + j * (j + 1) // 2 if j >= l else j + l * (l + 1) // 2 
                                for d in range(nv):
                                    kd = k + (d + no) * (d  + no + 1) // 2
                                    lc = l + (c + no) * (c + no + 1) // 2
                                    kdlc = (kd + lc * (lc + 1) // 2
                                            if lc >= kd else
                                            lc + kd * (kd + 1) // 2)
                                    tmp_1+= wf.amplitudes[no*nv+li*nv**2+a*nv+d]*g[kdlc]
                                    tmp_2+= wf.amplitudes[no*nv+lj*nv**2+a*nv+d]*g[kdlc]
                            omega[ij*nv**2+a*nv+b]+= wf.amplitudes[no*nv+kj*nv**2+b*nv+c]*(tmp_1-gt1[k,i,a,c])/2
                            omega[ij*nv**2+a*nv+b]+= wf.amplitudes[no*nv+ki*nv**2+a*nv+c]*(tmp_2-gt1[k,j,b,c])/2
                elif a == b and i != j:
                    for k in range(no):
                        ki = k + i * (i + 1) // 2 if i >= k else i + k * (k + 1) // 2 
                        kj = k + j * (j + 1) // 2 if j >= k else j + k * (k + 1) // 2 
                        for c in range(nv):
                            omega[ij*nv**2+a*nv+b]-=3* wf.amplitudes[no*nv+ki*nv**2+a*nv+c]*gt1[k,j,a,c]/2
                            tmp = 0
                            for l in range(no):
                                li = l + i * (i + 1) // 2 if i >= l else i + l * (l + 1) // 2 
                                for d in range(nv):
                                    kd = k + (d + no) * (d  + no + 1) // 2
                                    lc = l + (c + no) * (c + no + 1) // 2
                                    kdlc = (kd + lc * (lc + 1) // 2
                                            if lc >= kd else
                                            lc + kd * (kd + 1) // 2)
                                    tmp+= wf.amplitudes[no*nv+li*nv**2+a*nv+d]*g[kdlc]
                            omega[ij*nv**2+a*nv+b]+= 3*wf.amplitudes[no*nv+kj*nv**2+a*nv+c]*(tmp-gt1[k,i,a,c])/2
                elif a != b and i == j:
                    for k in range(no):
                        ki = k + i * (i + 1) // 2 if i >= k else i + k * (k + 1) // 2 
                        for c in range(nv):
                            omega[ij*nv**2+a*nv+b]-=3* wf.amplitudes[no*nv+ki*nv**2+a*nv+c]*gt1[k,i,a,c]/2
                            tmp = 0
                            for l in range(no):
                                li = l + i * (i + 1) // 2 if i >= l else i + l * (l + 1) // 2 
                                for d in range(nv):
                                    kd = k + (d + no) * (d  + no + 1) // 2
                                    lc = l + (c + no) * (c + no + 1) // 2
                                    kdlc = (kd + lc * (lc + 1) // 2
                                            if lc >= kd else
                                            lc + kd * (kd + 1) // 2)
                                    tmp+= wf.amplitudes[no*nv+li*nv**2+a*nv+d]*g[kdlc]
                            omega[ij*nv**2+a*nv+b]+= 3*wf.amplitudes[no*nv+ki*nv**2+b*nv+c]*(tmp-gt1[k,i,a,c])/2
                else:
                    for k in range(no):
                        ki = k + i * (i + 1) // 2 if i >= k else i + k * (k + 1) // 2 
                        for c in range(nv):
                            tmp = 0
                            for l in range(no):
                                li = l + i * (i + 1) // 2 if i >= l else i + l * (l + 1) // 2 
                                for d in range(nv):
                                    kd = k + (d + no) * (d  + no + 1) // 2
                                    lc = l + (c + no) * (c + no + 1) // 2
                                    kdlc = (kd + lc * (lc + 1) // 2
                                            if lc >= kd else
                                            lc + kd * (kd + 1) // 2)
                                    tmp+= wf.amplitudes[no*nv+li*nv**2+a*nv+d]*g[kdlc]
                            omega[ij*nv**2+a*nv+b]+= 3*wf.amplitudes[no*nv+ki*nv**2+a*nv+c]*(tmp/2-gt1[k,i,a,c])

def  _res_doub_D2(omega,Lt1,L,u,no,nv):
    """ This function calculates the Omega_{ij}^{ab(D2)}+Omega_{ji}^{ba(D2)} term
    """
    #TODO: Test the results
    for ij in range(no*(no+1)//2):
        i,j = divmod(ij,no)
        if j < i:
            j = i + j
        for a in range(nv):
            for b in range(nv):
                if i == j and a == b:
                    for c in range(nv):
                        for k in range(no):
                            ik = k + i * (i + 1)//2 if i >= k else i + k * (k + 1)//2
                            tmp = 0
                            for l in range(no):
                                lk = k + l * (l + 1)//2 if l >= k else l + k * (k + 1)//2
                                il = l + i * (i + 1)//2 if i >= l else i + l * (l + 1)//2
                                for d in range(nv):
                                    tmp+= u[il*nv**2+a*nv+d]*L[lk*nv**2+d*nv+c]
                            omega[ij*nv**2+a*nv+b]+= u[ik*nv**2+a*nv+c]*(Lt1[a,i,k,c]+tmp/2)
                else:
                    for c in range(nv):
                        ik = k + i * (i + 1)//2 if i >= k else i + k * (k + 1)//2
                        jk = k + j * (j + 1)//2 if j >= k else j + k * (k + 1)//2
                        for k in range(no):
                            omega[ij*nv**2+a*nv+b]+= u[ik*nv**2+a*nv+c]*Lt1[b,j,k,c]/2
                            tmp = 0
                            for l in range(no):
                                il = l + i * (i + 1)//2 if i >= l else i + l * (l + 1)//2
                                for d in range(nv):
                                    tmp+= u[il*nv**2+a*nv+d]*L[lk*nv**2+d*nv+c]
                            omega[ij*nv**2+a*nv+b]+= u[jk*nv**2+b*nv+c]*(Lt1[a,i,k,c]+tmp)/2
                            
def _res_doub_E2(omega,Ft1vv,Ft1oo,g,wf,u,no,nv):
    """ This function calculates the Omega_{ij}^{ab(E2)}+Omega_{ji}^{ba(E2)} term
    """
    #TODO: Test the results
    for ij in range(no*(no+1)//2):
        i,j = divmod(ij,no)
        if j < i:
            j = i + j
        for a in range(nv):
            for b in range(nv):
                if i != j and a != b:
                    for c in range(nv):
                        tmp_1 = 0
                        tmp_2 = 0
                        for kl in range(no*(no+1)//2):
                            k,l = divmod(kl,no)
                            if l < k:
                                l = k + l
                            kc = k + (c + no) * (c + no + 1) // 2
                            for d in range(nv):
                                ld = l + (d + no) * (d  + no + 1) // 2
                                ldkc = (ld + kc * (kc + 1) // 2
                                        if kc >= ld else
                                        kc + ld * (ld + 1) // 2)
                                g_tmp = g[ldkc]
                                tmp_1+= u[kl*nv**2+b*nv+d]*g_tmp
                                tmp_2+= u[kl*nv**2+a*nv+d]*g_tmp
                        omega[ij*nv**2+a*nv+b]+= wf.amplitudes[no*nv+ij*nv**2+a*nv+c]*(Ft1vv[b,c]-tmp_1)
                        omega[ij*nv**2+a*nv+b]+= wf.amplitudes[no*nv+ij*nv**2+c*nv+b]*(Ft1vv[a,c]-tmp_2)
                    for k in range(no):
                        tmp_1 = 0
                        tmp_2 = 0
                        ik = k + i * (i + 1)//2 if i >= k else i + k * (k + 1)//2
                        jk = k + j * (j + 1)//2 if j >= k else j + k * (k + 1)//2
                        for l in range(no):
                            lj = l + j * (j + 1)//2 if j >= l else j + l * (l + 1)//2
                            li = l + i * (i + 1)//2 if i >= l else i + l * (l + 1)//2
                            for c in range(nv):
                                lc = l + (c + no) * (c + no + 1) // 2
                                for d in range(nv):
                                    kd = k + (d + no) * (d  + no + 1) // 2
                                    kdlc = (kd + lc * (lc + 1) // 2
                                            if lc >= kd else
                                            lc + kd * (kd + 1) // 2)
                                g_tmp = g[kdlc]
                                tmp_1+= u[lj*nv**2+c*nv+d]*g_tmp
                                tmp_2+= u[li*nv**2+c*nv+d]*g_tmp
                        omega[ij*nv**2+a*nv+b]-= wf.amplitudes[no*nv+ik*nv**2+a*nv+b]*(Ft1oo[k,j]+tmp_1)
                        omega[ij*nv**2+a*nv+b]-= wf.amplitudes[no*nv+jk*nv**2+b*nv+a]*(Ft1oo[k,i]+tmp_2)
                elif i != j and a == b:
                    for c in range(nv):
                        tmp = 0
                        for kl in range(no*(no+1)//2):
                            k,l = divmod(kl,no)
                            if l < k:
                                l = k + l
                            kc = k + (c + no) * (c + no + 1) // 2
                            for d in range(nv):
                                ld = l + (d + no) * (d  + no + 1) // 2
                                ldkc = (ld + kc * (kc + 1) // 2
                                        if kc >= ld else
                                        kc + ld * (ld + 1) // 2)
                                tmp+= u[kl*nv**2+b*nv+d]*g[ldkc]
                        omega[ij*nv**2+a*nv+b]+= (wf.amplitudes[no*nv+ij*nv**2+a*nv+c]+wf.amplitudes[no*nv+ij*nv**2+c*nv+a])*(Ft1vv[a,c]-tmp)
                    for k in range(no):
                        tmp_1 = 0
                        tmp_2 = 0
                        ik = k + i * (i + 1)//2 if i >= k else i + k * (k + 1)//2
                        jk = k + j * (j + 1)//2 if j >= k else j + k * (k + 1)//2
                        for l in range(no):
                            lj = l + j * (j + 1)//2 if j >= l else j + l * (l + 1)//2
                            li = l + i * (i + 1)//2 if i >= l else i + l * (l + 1)//2
                            for c in range(nv):
                                lc = l + (c + no) * (c + no + 1) // 2
                                for d in range(nv):
                                    kd = k + (d + no) * (d  + no + 1) // 2
                                    kdlc = (kd + lc * (lc + 1) // 2
                                            if lc >= kd else
                                            lc + kd * (kd + 1) // 2)
                                g_tmp = g[kdlc]
                                tmp_1+= u[lj*nv**2+c*nv+d]*g_tmp
                                tmp_2+= u[li*nv**2+c*nv+d]*g_tmp
                        omega[ij*nv**2+a*nv+b]-= wf.amplitudes[no*nv+ik*nv**2+a*nv+b]*(Ft1oo[k,j]+tmp_1)
                        omega[ij*nv**2+a*nv+b]-= wf.amplitudes[no*nv+jk*nv**2+b*nv+a]*(Ft1oo[k,i]+tmp_2)
                elif i == j and a != b:
                    for c in range(nv):
                        tmp_1 = 0
                        tmp_2 = 0
                        for kl in range(no*(no+1)//2):
                            k,l = divmod(kl,no)
                            if l < k:
                                l = k + l
                            kc = k + (c + no) * (c + no + 1) // 2
                            for d in range(nv):
                                ld = l + (d + no) * (d  + no + 1) // 2
                                ldkc = (ld + kc * (kc + 1) // 2
                                        if kc >= ld else
                                        kc + ld * (ld + 1) // 2)
                                g_tmp = g[ldkc]
                                tmp_1+= u[kl*nv**2+b*nv+d]*g_tmp
                                tmp_2+= u[kl*nv**2+a*nv+d]*g_tmp
                        omega[ij*nv**2+a*nv+b]+= wf.amplitudes[no*nv+ij*nv**2+a*nv+c]*(Ft1vv[b,c]-tmp_1)
                        omega[ij*nv**2+a*nv+b]+= wf.amplitudes[no*nv+ij*nv**2+c*nv+b]*(Ft1vv[a,c]-tmp_2)
                    for k in range(no):
                        tmp = 0
                        ik = k + i * (i + 1)//2 if i >= k else i + k * (k + 1)//2
                        for l in range(no):
                            li = l + i * (i + 1)//2 if i >= l else i + l * (l + 1)//2
                            for c in range(nv):
                                lc = l + (c + no) * (c + no + 1) // 2
                                for d in range(nv):
                                    kd = k + (d + no) * (d  + no + 1) // 2
                                    kdlc = (kd + lc * (lc + 1) // 2
                                            if lc >= kd else
                                            lc + kd * (kd + 1) // 2)
                                tmp+= u[li*nv**2+c*nv+d]*g[kdlc]
                        omega[ij*nv**2+a*nv+b]-= (wf.amplitudes[no*nv+ik*nv**2+a*nv+b]+wf.amplitudes[no*nv+ik*nv**2+b*nv+a])*(Ft1oo[k,i]+tmp)
                else:
                    for c in range(nv):
                        tmp = 0
                        for kl in range(no*(no+1)//2):
                            k,l = divmod(kl,no)
                            if l < k:
                                l = k + l
                            kc = k + (c + no) * (c + no + 1) // 2
                            for d in range(nv):
                                ld = l + (d + no) * (d  + no + 1) // 2
                                ldkc = (ld + kc * (kc + 1) // 2
                                        if kc >= ld else
                                        kc + ld * (ld + 1) // 2)
                                tmp+= u[kl*nv**2+b*nv+d]*g[ldkc]
                        omega[ij*nv**2+a*nv+b]+= 2*wf.amplitudes[no*nv+ij*nv**2+a*nv+c]*(Ft1vv[a,c]-tmp)
                    for k in range(no):
                        tmp = 0
                        ik = k + i * (i + 1)//2 if i >= k else i + k * (k + 1)//2
                        for l in range(no):
                            li = l + i * (i + 1)//2 if i >= l else i + l * (l + 1)//2
                            for c in range(nv):
                                lc = l + (c + no) * (c + no + 1) // 2
                                for d in range(nv):
                                    kd = k + (d + no) * (d  + no + 1) // 2
                                    kdlc = (kd + lc * (lc + 1) // 2
                                            if lc >= kd else
                                            lc + kd * (kd + 1) // 2)
                                tmp+= u[li*nv**2+c*nv+d]*g[kdlc]
                        omega[ij*nv**2+a*nv+b]-= 2*wf.amplitudes[no*nv+ik*nv**2+a*nv+a]*(Ft1oo[k,i]+tmp)
                    
                        
def _res_doub_C2_old(omega,gt1,g,wf,no,nv):
    """ This function calculates only the Omega_{ij}^{ab(C2)} term
    """
    #TODO: Test the results
    for ij in range(no*(no+1)//2):
        i = ij // no
        j = ij % no
        for a in range(nv):
            for b in range(nv):
                tmp_1 = 0
                tmp_2 = 0
                for k in range(no):
                    ki = k + i * (i + 1) // 2 if i >= k else i + k * (k + 1) // 2 
                    kj = k + j * (j + 1) // 2 if j >= k else j + k * (k + 1) // 2 
                    for c in range(nv):
                        tmp_i = 0
                        tmp_j = 0
                        for d in range(nv):
                            for l in range(no):
                                li = l + i * (i + 1) // 2 if i >= l else i + l * (l + 1) // 2 
                                lj = l + j * (j + 1) // 2 if j >= l else j + l * (l + 1) // 2 
                                kd = k + (d + no) * (d  + no + 1) // 2
                                lc = l + (c + no) * (c + no + 1) // 2
                                kdlc = (kd + lc * (lc + 1) // 2
                                        if lc >= kd else
                                        lc + kd * (kd + 1) // 2)
                                tmp_i-= wf.amplitudes[no*nv+li*nv**2+a*nv+d]*g[kdlc]
                                tmp_j-= wf.amplitudes[no*nv+lj*nv**2+a*nv+d]*g[kdlc]
                        tmp_1-= (2*gt1[k,i,a,c]+tmp_i)*wf.amplitudes[no*nv+kj*nv**2+b*nv+c]
                        tmp_2-= (2*gt1[k,j,a,c]+tmp_j)*wf.amplitudes[no*nv+ki*nv**2+b*nv+c]
                omega[ij*nv**2+a*nv+b]+= tmp_1/4+tmp_2/2


def _res_doubles(wf,u,F,g):
    #TODO: All 
    omega2 = get_2e_vovo(g)
    _res_doub_A2pp(omega2,get_2e_vvvv(g),wf)
    _res_doub_B2(omega2,get_2e_oooo(g),g,wf)
    _res_doub_C2(omega2,get_2e_oovv(g),g,wf)
    _res_doub_D2(omega2,make_L_voov(g),make_L_ovov(g),u)
    _res_doub_E2(omega2,get_1e_vv(F),t1_1e_transf_oo(F,wf),g,wf,u)

