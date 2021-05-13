"""Class to run an iteration step in Hartree Fock optimisation


"""
import logging

import numpy as np
from scipy import linalg

from util.variables import sqrt2
from input_output.log import logtime
from . import util

logger = logging.getLogger(__name__)
loglevel = logging.getLogger().getEffectiveLevel()


class HartreeFockStep():
    """The iteration step in a Hartree-Fock optimisation
    
    """
    def __init__(self):
        self.n_DIIS = 0
        self.n_occ = 0
        self.n_occ_alpha = 0
        self.n_occ_beta = 0
        self.integrals = None
        self.orb = None
        self.energy = None
        self.grad = None
        self.Dmat = None
        self.Fock = None
        self.gradNorm = None

    def initialise(self, step_type):
        if step_type == 'RH-SCF':
            self.i_DIIS = -1
            self.grad = np.zeros((self.n_occ_alpha,
                                  len(self.orb) - self.n_occ_alpha,
                                  max(self.n_DIIS, 1)))
            self.Dmat = np.zeros((len(self.orb),
                                  len(self.orb),
                                  max(self.n_DIIS, 1)))
        elif step_type == 'densMat-SCF':
            pass
        elif step_type == 'Absil':
            self.grad = np.zeros((len(self.orb),
                                  self.n_occ))
            self.integrals.g.transform_to_ijkl()
        elif step_type == 'orb_rot-Newton':
            pass
        else:
            raise ValueError("Unknown type of Hartree-Fock step: "
                             + step_type)
        
    def roothan_hall(self, i_SCF):
        """Roothan-Hall procedure as described, e.g, in Szabo
        
        """
        if i_SCF < self.n_DIIS:
            cur_n_DIIS = i_SCF
        else:
            cur_n_DIIS = self.n_DIIS
        if self.n_DIIS > 0:
            logger.info('current n DIIS:', cur_n_DIIS)

        with logtime('Form the density matrix'):
            self.Dmat[:, :, self.i_DIIS] = 2 * np.einsum(
                'pi,qi->pq',
                self.orb[0][:, :self.n_occ],
                self.orb[0][:, :self.n_occ])
        logger.debug('Density matrix:\n%r', self.Dmat[:, :, self.i_DIIS])

        if self.n_DIIS > 0 and cur_n_DIIS > 0:
            with logtime('DIIS step'):
                util.calculate_DIIS(
                    self.Dmat, self.grad, cur_n_DIIS, self.i_DIIS)

        with logtime('Form Fock matrix'):
            # F[mn] = h[mn] + Dmat[rs]*(g[mnrs] - g[mrsn]/2)
            # The way that einsum is made matters a lot in the time
            Fock = np.array(self.integrals.h)
            tmp = np.einsum('rs,Frs->F',
                            self.Dmat[:, :, self.i_DIIS],
                            self.integrals.g._integrals)
            Fock += np.einsum('F,Fmn->mn',
                              tmp,
                              self.integrals.g._integrals)
            tmp = np.einsum('rs,Fms->Frm',
                            self.Dmat[:, :, self.i_DIIS],
                            self.integrals.g._integrals)
            Fock -= np.einsum('Frm,Frn->mn',
                              tmp,
                              self.integrals.g._integrals) / 2
        logger.debug('Fock matrix:\n%r', Fock)

        with logtime('Calculate Energy'):
            self.energy = np.tensordot(self.Dmat[:, :, self.i_DIIS],
                                       self.integrals.h + Fock) / 2
            self.one_el_energy = np.tensordot(self.Dmat[:, :, self.i_DIIS],
                                              self.integrals.h)
            self.two_el_energy = np.tensordot(self.Dmat[:, :, self.i_DIIS],
                                              Fock - self.integrals.h) / 2
        logger.info(
            'Electronic energy: %f\nOne-electron energy: %f'
            + '\nTwo-electron energy: %f',
            self.energy, self.one_el_energy, self.two_el_energy)

        self.i_DIIS += 1
        if self.i_DIIS >= self.n_DIIS:
            self.i_DIIS = 0
        if self.n_DIIS > 0:
            logger.info('current self.i_DIIS:', self.i_DIIS)
        with logtime('Fock matrix in the MO basis'):
            F_MO = self.orb[0].T @ Fock @ self.orb[0]
            self.grad[:, :, self.i_DIIS] = F_MO[:self.n_occ, self.n_occ:]
            self.gradNorm = linalg.norm(self.grad[:, :, self.i_DIIS]) * sqrt2
            logger.info('Gradient norm = %f', self.gradNorm)
            self.orb.energies = np.array([F_MO[i, i]
                                          for i in range(len(self.orb))])
            if loglevel <= logging.INFO:
                logger.info('Current orbital energies:\n%r', self.orb.energies)

        with logtime('Fock matrix in the orthogonal basis'):
            Fock = self.integrals.X.T @ Fock @ self.integrals.X
            e, C = linalg.eigh(Fock)
    #        e, C = linalg.eig(Fock)
    #        e_sort_index = np.argsort(e)
    #        e = e[e_sort_index]
    #        C = C[:, e_sort_index]
            # ----- Back to the AO basis
            self.orb[0][:, :] = self.integrals.X @ C

    def density_matrix_scf(self, i_SCF):
        raise NotImplementedError("Density matrix based SCF")

    def newton_absil(self, i_SCF):
        N = self.n_occ
        n = len(self.orb)
        Gh = np.zeros((n, N)) # parte de um elétron do gradiente
        Gg = np.zeros((len(self.orb), # parte de dois elétrons do gradiente
                       self.n_occ))
        GEij = np.zeros((n,N)) # derivada direcional na direção E_{ij}
        D = np.zeros((n*N,n*N)) # a matriz da derivada direcional em si
        X = self.orb[0][0][:,:self.n_occ]
        # ponto inicial
        col = 0
        aux = np.zeros((n,n)) # matriz com os termos c_{qk}*c_{sk} do gradiente de dois elétrons
        tmp = 0
        self.energy = 0
        
        for a in range(0,n):
            for b in range(0,N):                
                for p in range(0,n):
                    Gh[a][b] += X[p][b]*(self.integrals.h[a][p] + self.integrals.h[p][a]) # parte de um elétron do gradiente
                    self.energy += X[a][b]*X[p][b]*self.integrals.h[a][p]

                for q in range(0,n):
                    for s in range(0,n):
                        for k in range(0,N):
                            if k != b:
                                aux[q][s] += X[q][k]*X[s][k]
                        tmp += (aux[q][s] *
                                (self.integrals.g[a,q,a,s]
                                - self.integrals.g[a,q,s,a]))
                        for j in range(k+1,N):
                            self.energy += (X[a][k]*X[b][j]*X[q][k]*X[s][j]*
                                            (self.integrals.g[a,b,q,s]
                                             - self.integrals.g[a,b,s,q]))
                            
                tmp2 = tmp
                tmp *= 2*X[a][b]
                tmp3 = 0

                for p in range(0,n):
                    if p != a:
                        for q in range(0,n):
                            for s in range(0,n):
                                tmp += (X[p][b] * aux[q][s] *
                                        (2*self.integrals.g[a,q,p,s]
                                         - self.integrals.g[a,q,s,p]
                                         - self.integrals.g[p,q,s,a]))
                                tmp3 += tmp / X[p][b]

                Gg[a][b] = tmp
                tmp = 0
                
                for i in range(0,n):
                    for j in range(0,N):
                        if i == a and j == b:
                            GEij[i][j] += 2*self.integrals.h[a][a] + 2*tmp2
                        if i != a and j == b:
                            GEij[i][j] += self.integrals.h[a][i]+self.integrals.h[i][a]+tmp3
                        if j != b:
                            GEij[i][j] += (4*X[a][b]*X[i][j] *
                                           self.integrals.g[a,i,a,i]
                                           - self.integrals.g[a,i,i,a])
                            for q in range(0,n):
                                if q != i:
                                    GEij[i][j] += (2*X[a][b]*X[q][j]*
                                                   (2*self.integrals.g[a,i,a,q]
                                                    - self.integrals.g[a,i,q,a]
                                                    - self.integrals.g[a,q,i,a]))
                            for p in range(0,n):
                                if p != a:
                                    for q in range(0,n):
                                        GEij[i][j] += (X[p][b]*X[q][j]*
                                                (2*self.integrals.g[a,i,p,q]
                                                 - self.integrals.g[a,i,q,p]
                                                 - self.integrals.g[p,i,q,a]
                                                 + 2*self.integrals.g[a,q,p,i]
                                                 - self.integrals.g[a,q,i,p]
                                                 - self.integrals.g[p,q,i,a]))
                            
                D[:,[col]] = np.reshape(GEij, (n*N,1), 'F') # pode dar erro se fizer com 'F', mas deveria ser o jeito correto pois estou mantendo esse padrão em todos os lugares
                GEij = np.zeros((n,N))
                col += 1
                
        self.grad = Gh + Gg
        R = (np.identity(n) - X @ np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X)) @ self.grad
        R = np.reshape(R, (n*N, 1), 'F')
        eta = np.linalg.solve(D,R)
        eta = np.reshape(eta, (n,N), 'F')
        u, s, v = np.linalg.svd(eta, full_matrices=False)
        s = np.diag(s)
        X = X @ np.transpose(v) @ np.cos(s) + u @ np.sin(s)
        self.orb[0][0][:,:self.n_occ] = X
        self.gradNorm = np.linalg.norm(self.grad[:,:])
        
        #raise NotImplementedError("Caio, this is up to you!")
    
    def newton_orb_rot(self, i_SCF):
        raise NotImplementedError("As described in Helgaker's book")
