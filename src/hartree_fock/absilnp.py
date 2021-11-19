import sys
import numpy as np
from scipy import linalg
np.set_printoptions(threshold=sys.maxsize)

def build_fock(Ps, Pt, h, g):
    """Compute Fock matrix.
    
    F_ij = H_ij + [(Ps)_lk + (Pt)_lk] (ij|L)(L|lk) - (Ps)_lk (ik|L)(L|lj)
    """
    Fock = np.array(h)

    tmp = np.einsum('ij,Lij->L', Ps, g)
    Fock += np.einsum('L,Lkl->kl', tmp, g)
    tmp = np.einsum('ij,Lij->L', Pt, g)
    Fock += np.einsum('L,Lkl->kl', tmp, g)
    tmp = np.einsum('ij,Lkj->Lik', Ps, g)
    Fock -= np.einsum('Lik,Lil->kl', tmp, g)

    return Fock

def grad_fock(W, F, Ps, g):
    grad = 2 * (F @ W)
    n, N = W.shape

    tmp = np.einsum('Lji,ib->jbL', g, W)
    tmp2 = np.einsum('jk,Lak->ajL', Ps, g)
    grad += np.einsum('ajL,jbL->ab', tmp2, tmp)
    tmp = np.einsum('Lki,ib->kbL', g, W)
    tmp2 = np.einsum('jk,Laj->akL', Ps, g)
    grad += np.einsum('akL,kbL->ab', tmp2, tmp)

    return grad

def hess(v, g):

    tmp = np.einsum('Lqs,s->Lq', g, v)
    tmpa = np.einsum('Lq,q->L', tmp, v)
    tmp2 = np.einsum('Lai,L->ai', g, tmpa)
    tmp3 = np.einsum('Las,s->La', g, v)
    tmp4 = np.einsum('Lqi,q->Li', g, v)
    tmp5 = np.einsum('La,Li->ai', tmp3, tmp4)

    return tmp2 - tmp5
