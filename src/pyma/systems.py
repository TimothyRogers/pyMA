'''
Representations of Dynamic Systems
'''

from pyma import utils
import numpy as np
from numpy.linalg import eig, inv
from scipy.linalg import expm
import itertools

class DynamicSystem():
    '''
    Base Class for Dynamic Systems
    '''

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
            w = np.linspace(0,1.2*self.Omega.max(),1024)

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
            den =  self.Omega[r]**2 - w**2 + 1j*2*self.Zeta[r]*self.Omega[r]*w
            frf = frf + rAjk/den
        return frf


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
                self.Omega = np.abs(lam[::2])
                self.Zeta = -np.real(lam[::2])/self.Omega
            else:
                self.Omega = np.sqrt(lam_ud)
                self.Zeta = np.zeros_like(lam_ud)
            self.Phi = phi
        else:
            self.Omega = None
            self.Zeta = None
            self.Phi = None

    
    def as_modal(self):

        if self.C is None:
            # Undamped system
            return ModalModel(Omega=self.Omega,Phi=self.Phi)
        else:
            return ModalModel(Omega=self.Omega, Phi=self.Phi, Zeta=self.Zeta)
        

class ModalModel(DynamicSystem):
    '''
    Dynamic System in Modal Coordinates - Omega, Phi
    '''

    def __init__(self,Omega=None, Phi=None, Zeta=None):
        
        self.Omega = Omega
        self.Phi = Phi
        self.Zeta = Zeta


class StateSpace(DynamicSystem):
    '''
    Dynamic system as a linear Gaussian SSM

    If continuous is True:
        A is continuous time transition matrix
        B is continuous time control matrix
        C is observation matrix
        D is shoot through matrix
        Q is L*q*L' where q is white noise spectral density
        R is measurement noise covariance

    Else:
        A is discrete time transition matrix
        B is discrete time control matrix
        C is observation matrix
        D is shoot through matrix
        Q is process noise covariance
        R is measurement noise covariance

    dt - sample time default 1 Hz

    '''

    def __init__(self, A=None, B=None, C=None, D=None, Q=None, R=None, continuous=True, dt=1):
        #self.__dict__.update(kwargs) # Set kwargs as instance attributes
        pass


    def frf(self,w=None,J=None,K=None):
        # Need to add conversion here
        raise NotImplementedError()

    def discretise(self):

        if self.continuous:

            # Using a matrix fraction decomposition to discretize
            Phi = expm(self.dt*np.block([
                [self.A,           self.Q  ],
                [np.zeros((2,2)), -self.A.T]
            ]))
            A = Phi[0:self._Dx,0:self._Dx]
            Q = Phi[0:self._Dx,self._Dx+1:] @ A.T
            if self.B is not None:
                B = inv(self.A) @ (A-np.eye(2)) @ self.B
                self.B = B
            
            self.A = A
            self.Q = Q

