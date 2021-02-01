""" CCSD closed shell
"""

import numpy as np


def make_u(wf}:
    """Generate the u matrix:
        u_{ij}^{ab}=2t_{ij}^{ab}-t_{ji}^{ab}

    Parameters:
    -----------

    wf (IntermNormWaveFunction)
        A wave function containing the double excitation amplitudes Matrix

    Return:
    -------

    u (??)
    

    """

def make_L(g):
    """Generate the intermediary L matrix:
          L_{pqrs} = 2*g_{pqrs}-g_{psrq}

    Parameters:
    -----------

    g (??)
       The two-electron integral matrix in atomic or molecular basis set.

    Return:
    -------

    L (??)
        An 4 index object or N**2 2 index ones (to define).

    """

def make_F(h,g_or_L,use_L=False):
    """Generate the Fock inactive matrix:
          F_{mn} = h_{mn}+\sum_{l}(2*g_{mnll}-g_{mlln})
       or
          F_{mn} = h_{mn}+\sum_{l}L_{mnll}

    Parameters:
    -----------

    h (??)
       The one-electron integral matrix in atomic or molecular basis set.

    g_or_L (??)
       The two-electron integral matrix in atomic or molecular basis set
       or the intermediate matrix L

    use_L (bool, optional, default=False)
       If True, treats g_or_L as L
       If False, treats g_or_L as g

    Return:
    -------

    F (??)

    """

def energy(wf,Eref):
    """Calculates the Coupled-Cluster energy

    Parameters:
    -----------

    wf (IntermNormWaveFunction)
        A wave function containing the single and double excitation amplitudes matrices.

    Eref (float)
        The reference energy

    Returns:
    --------

    CC_E (float)
        The coupled-Cluster energy
   

    """
   
def test_conv(omega1,omega2,threshold=1.0E-8)
    """Check if all residuals are below the the threshold.
       If at least one of the omega matrices is None, it returns False.

    Parameters:
    -----------

    omega1 (??)
        Matrix containing all one electron residual values

    omega2 (??)
        Matrix containing all two electron residual values

    threshold (float, optional, default=1.0E-8)
        Threshold to the residual values, if all values are belows it the 
calculation converged.

    Return:
    -------
    
    conv_status (bool)
        If True, the calculation converged.
        If False, does not converged.

    """
 
def get_t1_transf(h,g,wf):
    """Use the T1-transformation in the one- and two-electron matrices:

    Parameters:
    -----------

    h (??)
       The one-electron integral matrix in atomic or molecular basis set.

    g (??)
       The two-electron integral matrix in atomic or molecular basis set.
    
    wf (IntermNormWaveFunction)
        A wave function containing the single excitation amplitudes matrices.

    Return:
    -------

    ht1 (??)
       The T1-transformed one-electron integral matrix in atomic or molecular basis set.

    gt1 (??)
       The T1-transformed two-electron integral matrix in atomic or molecular basis set.

    """
        


def equation(wf,h,g,F,L):
    """Calculate the residual from the singles and doubles configurations.

    Parameters:
    -----------
    
    wf (IntermNormWaveFunction)
        A wave function containing the single and double excitation amplitudes matrices.
    
    h (??)
       The one-electron integral matrix in atomic or molecular basis set

    g (??)
       The two-electron integral matrix in atomic or molecular basis set

    F (??)
       The intermediate matrix F, if inter_matrix=False it is not used

    L (??)
       The intermediate matrix L, if inter_matrix=False it is not used

    Return:
    -------

    omega1 (??)
        Matrix containing all one electron residual values

    omega2 (??)
        Matrix containing all two electron residual values

    """
    if wf.level == 'SD':
        ht1,gt1 = get_t1_transf(h,g,cc_wf)
        if cc_wf.inter_matrix:
            u = make_u(wf)
            F = make_t1_F(ht1,gt1,F)
            L = make_t1_L(gt1,L)
       omega1 = _res_singles(??)  ##Probably divide this in two cases with or without inter
       omega2 = _res_doubles(??)
    elif cc_wf.level == 'D':
        if cc_wf.inter_matrix:
            u = make_u(wf)
        omega1 = 0
        omega2 = _res_doubles()
    else:
        raise ValueError(
            'CC version '+cc_wf.level+' unknown.')

    return omega1,omega2

def _res_singles(book=True):
    """Calculate the residual from the singles.

    TODO decide which formula will be used
    """
    if book == True:
        tmp=2*np.einsum('ikac,kcll->ai',t,g[:no,no:,:no,:no])
        tmp+=np.einsum('kicd,adkc->ai',t,g[no:,no:,:no,no:])
        tmp+=np.einsum('ikac,kc->ai',t,h[:no,no:])
        tmp+=np.einsum('aill->ai',g[no:,:no,:no,:no])
        tmp-=np.einsum('klac,kilc->ai',t,g[:no,:no,:no,no:])
        tmp-=np.einsum('kiac,kcll->ai',t,g[:no,no:,:no,:no])
        tmp*=2
        tmp+=np.einsum('lkac,kilc->ai',t,g[:no,:no,:no,no:])
        tmp+=np.einsum('kiac,kllc->ai',t,g[:no,:no,:no,no:])
        tmp-=np.einsum('ikcd,adkc->ai',t,g[no:,no:,:no,no:])
        tmp-=np.einsum('kiac,kc->ai',t,h[:no,no:])
        tmp-=np.einsum('alli->ai',g[no:,:no,:no,:no])
        tmp+=h[no:,:no]
    else:
        tmp=2*np.einsum('ikac,kcll->kaic',t,g[:no,no:,:no,:no])
        tmp+=np.einsum('ikac,kc->kaic',t,h[:no,no:])
        tmp-=np.einsum('kiac,kcll->kaic',t,g[:no,no:,:no,:no])
        tmp-=np.einsum('ikac,kllc->kaic',t,g[:no,:no,:no,no:])
            tmp+=np.einsum('kicd,adkc->kaic',t,g[no:,no:,:no,no:])
            tmp-=np.einsum('klac,kilc->kaic',t,g[:no,:no,:no,no:])
            tmp*=2
            tmp+=np.einsum('kiac,kllc->kaic',t,g[:no,:no,:no,no:])
            tmp-=np.einsum('ikcd,adkc->kaic',t,g[no:,no:,:no,no:])
            tmp-=np.einsum('kiac,kc->kaic',t,h[:no,no:])
            tmp+=np.einsum('lkac,kilc->kaic',t,g[:no,:no,:no,no:])
            tmp=np.einsum('kaic->kai',tmp)
            tmp-=np.einsum('akki->kai',g[no:,:no,:no,:no])
            tmp+=2*np.einsum('aikk->kai',g[no:,:no,:no,:no])
            tmp=np.einsum('kai->ai',tmp)
            tmp+=h[no:,:no]
        return tmp
def _res_doubles(self):
    """
    """
    tmp=-np.einsum('ljbc,kiad,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=np.einsum('liac,kjbd,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp*=0.5
    tmp+=np.einsum('kibd,ljac,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('kjad,libc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('jlbc,kiad,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('ljbc,ikad,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('ljbd,kiac,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('ilad,jkbc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('ilac,kjbd,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('liac,jkbd,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('liad,kjbc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=np.einsum('kjbc,kiac->ijab',t,g[:no,:no,no:,no:])
    tmp-=np.einsum('kiac,kjbc->ijab',t,g[:no,:no,no:,no:])
    tmp+=np.einsum('kjbc,acki->ijab',t,g[no:,no:,:no,:no])
    tmp+=np.einsum('kiac,bckj->ijab',t,g[no:,no:,:no,:no])
    tmp*=0.5
    tmp+=np.einsum('klab,ijcd,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('kjbd,liac,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('kiad,ljbc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('ijad,klbc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('ikab,jldc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('jibd,klac,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=np.einsum('jkba,ildc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=np.einsum('jlbc,ikad,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=np.einsum('jlbd,ikac,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=np.einsum('ilac,jkbd,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=np.einsum('ilad,jkbc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=np.einsum('liad,jkbc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=np.einsum('kibc,kjac->ijab',t,g[:no,:no,no:,no:])
    tmp-=np.einsum('kjac,kibc->ijab',t,g[:no,:no,no:,no:])
    tmp-=np.einsum('jkbc,acki->ijab',t,g[no:,no:,:no,:no])
    tmp-=np.einsum('kjbc,aikc->ijab',t,g[no:,:no,:no,no:])
    tmp-=np.einsum('ikac,bckj->ijab',t,g[no:,:no,:no,no:])
    tmp-=np.einsum('kiac,bjkc->ijab',t,g[no:,:no,:no,no:])
    tmp+=np.einsum('ijac,bc->ijab',t,h[no:,no:])
    tmp+=np.einsum('jibc,ac->ijab',t,h[no:,no:])
    tmp+=np.einsum('ikab,kj->ijab',t,h[:no,:no])
    tmp+=np.einsum('jkba,ki->ijab',t,h[:no,:no])
    tmp+=np.einsum('ijcd,acbd->ijab',t,g[no:,no:,no:,no:])
    tmp+=np.einsum('klab,kilj->ijab',t,g[:no,:no,:no,:no])
    tmp-=np.einsum('ijac,bkkc->ijab',t,g[no:,:no,:no,no:])
    tmp-=np.einsum('jibc,akkc->ijab',t,g[no:,:no,:no,no:])
    tmp-=np.einsum('ikab,kllj->ijab',t,g[no:,:no,:no,no:])
    tmp-=np.einsum('jkba,klli->ijab',t,g[no:,:no,:no,no:])
    tmp+=2*np.einsum('jlbd,ikac,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=2*np.einsum('ijad,lkbc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=2*np.einsum('ikab,ljdc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=2*np.einsum('jibd,lkac,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp-=2*np.einsum('jkba,lidc,kcld->ijab',t,t,g[:no,no:,:no,no:])
    tmp+=2*np.einsum('jkbc,aikc->ijab',t,g[no:,:no,:no,no:])
    tmp+=2*np.einsum('ikac,bjkc->ijab',t,g[no:,:no,:no,no:])
    tmp+=2*np.einsum('ijac,bckk->ijab',t,g[no:,no:,:no,:no])
    tmp+=2*np.einsum('jibc,ackk->ijab',t,g[no:,no:,:no,:no])
    tmp+=2*np.einsum('ikab,kjll->ijab',t,g[:no,:no,:no,:no])
    tmp+=2*np.einsum('jkba,kill->ijab',t,g[:no,:no,:no,:no])
    tmp+=g[no:,:no,no:,:no]
    return tmp 


