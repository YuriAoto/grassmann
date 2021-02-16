'''Residual terms an IntermNormWaveFunction (V2) must be used.
'''
import cython
from cython.parallel import prange


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing
#@cython.cdivision(True)     # Use C division
def calc_A2( double[:,:,:,:] g,
             double[:] amp,
             double[:] res,
             int no):
    '''Doule residual A2 term.
    '''
    cdef int a,b,ini_ij,ij,c,d,nv,nwf,nv2,i,j

    nv = g.shape[0]-no
    
    nv2 = nv*nv
    nwf_2 = amp.shape[0]//(nv2)    
    nwf = amp.shape[0] 

    i=0
    j=0
    for ij in range(nwf_2): 
        ini_ij = ij*nv2
        for a in range(nv):
            for b in range(nv):
                res[ini_ij+a*nv+b]=g[a,i,b,j]
        if i < j:
            i+=1
        else:
            i=0
            j+=1


    with nogil:
#        for ij in prange(nwf):
        for ij in prange(0,nwf,nv2):
#            ini_ij = ij*nv2
            for a in range(nv):
                for b in range(nv):
                    for c in range(nv):
                        for d in range(nv):
        #                    res[ini_ij+a*nv+b]+=amp[ini_ij+c*nv+d]*g[a,c,b,d]
                            res[ij+a*nv+b]+=amp[ij+c*nv+d]*g[a,c,b,d]





  

