""" CCSD closed shell
"""

import numpy as np

class Coupled_Cluster(self,   ##Tirar
                      version='SD',
                      T1=True,
                      inter_matrix=False):
    """
    """
    self.E=0
    self.version=version
    self.T1=T1
    self.inter_matrix=inter_matrix

    def u_matrix(self}:
    """
    """

    def L_matrix(self}:
    """
    """

    class CCSD(self):
        """
        """
        def energy(self):
            """
            """
            
        def get_t1_transf(self):
            """
            """
            
 
 
 
        class equation(self):
            """
            """
            def res_singles(book=True):
                """
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
            def res_doubles(self):
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


