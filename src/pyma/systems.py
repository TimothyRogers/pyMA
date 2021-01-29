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

    @property
    def C(self):
        return self._C
    
    @C.setter
    @utils.num_to_np
    @utils.valid_system_matrix
    def C(self,C):
        self.dofs_update(C)
        self._C = C

    @property
    def K(self):
        return self._K
    
    @K.setter
    @utils.num_to_np
    @utils.valid_system_matrix
    def K(self,K):
        self.dofs_update(K)
        self._K = K

    def first_order_form(self):
        MI = inv(self.M)
        bottom = np.hstack((-MI@self.K, -MI@self.C))
        top = np.eye(self.dofs,self.dofs*2,self.dofs)
        return np.vstack((top,bottom))


    def as_modal(self):

        if self.C is None:
            # Undamped system
            lam, phi = eig(inv(self.M)@self.K)
            return ModalModel(Omega=np.sqrt(lam),Phi=phi)
        else:
            MI = inv(self.M)
            lam, phi = eig(self.first_order_form())
            lam = lam[::2]
            ms = np.real(lam*phi[:self.dofs,::2])
            return ModalModel(Omega=np.abs(lam), Phi=ms, Zeta=(-np.real(lam)/np.abs(lam)))
        

    


class ModalModel(DynamicSystem):
    '''
    Dynamic System in Modal Coordinates - Omega, Phi
    '''

    def __init__(self,Omega=None, Phi=None, Zeta=None):
        
        self.Omega = Omega
        self.Phi = Phi
        self.Zeta = Zeta
