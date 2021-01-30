'''
Representations of Dynamic Systems
'''

from pyma import utils
import numpy as np
from numpy.linalg import eig, inv

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
            if self.C is not None:
                lam, _ = eig(self.first_order_form())
                self.wn = np.abs(lam[::2])
                self.zeta = -np.real(lam[::2])/self.wn
            else:
                self.wn = lam_ud
                self.zeta = np.zeros_like(lam_ud)
            self.Phi = phi
        else:
            self.wn = None
            self.zeta = None
            self.Phi = None

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
