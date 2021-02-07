'''
Representations of Dynamic Systems
'''

from pyma import utils
import numpy as np
from numpy.linalg import eig, inv
import itertools

class DynamicSystem():
    '''
    Base Class for Dynamic Systems
    '''

class SpatialModel(DynamicSystem):
    '''
    Dynamic System in Spatial Coordinates - M, C, K
    '''

    def __init__(self,M=None, C=None, K=None):
        
        self.dofs = None
        self._M = None
        self._C = None
        self._K = None


        self.M = M
        self.C = C
        self.K = K

    def dofs_update(self,A):
        if A is not None:
            if self.dofs == None or (A.shape[0] == self.dofs):
                self.dofs = A.shape[0]
            else:
                raise(ValueError)

    @property
    def M(self):
        return self._M
    
    @M.setter
    @utils.num_to_np
    @utils.valid_system_matrix
    def M(self,M):
        self.dofs_update(M)
        self._M = M
        self.update_modal()

    @property
    def C(self):
        return self._C
    
    @C.setter
    @utils.num_to_np
    @utils.valid_system_matrix
    def C(self,C):
        self.dofs_update(C)
        self._C = C
        self.update_modal()

    @property
    def K(self):
        return self._K
    
    @K.setter
    @utils.num_to_np
    @utils.valid_system_matrix
    def K(self,K):
        self.dofs_update(K)
        self._K = K
        self.update_modal()

    def first_order_form(self):
        MI = inv(self.M)
        bottom = np.hstack((-MI@self.K, -MI@self.C))
        top = np.eye(self.dofs,self.dofs*2,self.dofs)
        return np.vstack((top,bottom))

    def update_modal(self):
        # Compute modal properties
        if self.M is not None and self.K is not None:
            lam_ud, phi = eig(inv(self.M)@self.K)
            # Orthonormal modes
            Mrr = phi.T @ (self.M @ phi)
            phi = phi/np.diag(Mrr)
            if self.C is not None:
                lam, _ = eig(self.first_order_form())
                self.wn = np.abs(lam[::2])
                self.zeta = -np.real(lam[::2])/self.wn
            else:
                self.wn = np.sqrt(lam_ud)
                self.zeta = np.zeros_like(lam_ud)
            self.Phi = phi
        else:
            self.wn = None
            self.zeta = None
            self.Phi = None

    def frf(self,w=None,J=-1,K=-1):
        '''
        Compute complex receptance FRF

        Response at k due to excitation at j: -1 implies all
        Frequencies specified in w

        '''

        if J == -1:
            J = np.arange(0,self.dofs)

        if K == -1:
            K = np.arange(0,self.dofs)

        if w == None:
            w = np.linspace(0,1.2*self.wn.max(),1024)

        if isinstance(J,int):
            J = [J-1]

        if isinstance(K,int):
            K = [K-1]

        FRF = np.empty((len(J),len(K),len(w)),dtype=np.clongdouble)

        for a in J:
            for b in K:
                FRF[a,b] = self._frf(a,b,w)

        return FRF

    def _frf(self,jj,kk,w):

        frf = np.zeros(len(w),dtype=np.clongdouble)
        for r in range(self.dofs):
            rAjk = self.Phi[r,jj] * self.Phi[r,kk]
            den =  self.wn[r]**2 - w**2 + 1j*2*self.zeta[r]*self.wn[r]*w
            frf = frf + rAjk/den
        return frf

    def as_modal(self):

        if self.C is None:
            # Undamped system
            return ModalModel(Omega=self.wn,Phi=self.Phi)
        else:
            return ModalModel(Omega=self.wn, Phi=self.Phi, Zeta=self.zeta)
        

class ModalModel(DynamicSystem):
    '''
    Dynamic System in Modal Coordinates - Omega, Phi
    '''

    def __init__(self,Omega=None, Phi=None, Zeta=None):
        
        self.Omega = Omega
        self.Phi = Phi
        self.Zeta = Zeta
