# Polynomial Based Identification Algorithms

from pyma import utils
import numpy as np
from numpy.linalg import pinv, inv, svd


class PolynomialID():

    def __init__(self, order=None):

        self._Di = None
        self._Do = None
        self._Nf = None

        self.order = order

        pass

    def solve_polynomial(self,A,b):
        # Currently only OLS available
        return inv(A.T @ A) @ (A.T @ b)

    def companion_matrix(self, w):

        # Construct a companion matrix for a set of coeffs w
        # Assuming the coefficient of the max order is 1
        
        D = w.shape[0]
        C = np.vstack((np.eye(D-1,D,1),w.T))
        return C

    def roots(self, w):

        return np.linalg.eig(self.companion_matrix(w))[0]

    def poles_to_modal(self, lam, dt):

        mu = np.log( lam ) / dt
        wn = np.abs(mu)
        zeta = -np.real(mu) / wn
        return {'wn': wn, 'zeta': zeta, 'lam':mu}

        


class PolynomialTime(PolynomialID):

    def __init__(self, order=None):

        super().__init__(order)

        pass

class ComplexExponential(PolynomialTime):

    def __init__(self, order=None):

        super().__init__(order)


    def __call__( self, h=None, dt=1 ):

        id = []
        N = h.shape[0]

        # Iterating over the impulse response functions
        for hh in h.T:
            # Set up linear regression
            hank = utils.block_hankel( hh[:,None].T, self.order )
            A = hank[:-1,:].T
            b = hank[-1:,:].T
            # Solve
            w = self.solve_polynomial(A,b)
            Lam = self.roots(w)
            Lam = np.sort(Lam)
            #Lam = Lam[::2] # Remove conjugate pairs
            id.append(self.poles_to_modal( Lam , dt ))
            

            #eL = np.exp(id[-1:][0]['lam'][None,:] * np.arange(0,len(hh))[:,None]*dt)
            eL = Lam ** (np.arange(0,len(hh))[:,None])
            
            Apq = self.solve_polynomial(eL,hh[:,None])
            id[-1:][0]['Apq'] = Apq
                
        return id


class LSCE(PolynomialTime):

    def __init__(self, order=None):

        super().__init__(order)


    def __call__( self, h=None, dt=1 ):

        N, D = h.shape
        Np = N-self.order
        A = np.zeros((D*Np,self.order))
        b = np.zeros((D*Np,1))

        # Set up linear regression
        for ii, hh in enumerate(h.T):
            hank = utils.block_hankel( hh[:,None].T, self.order )
            A[ii*Np:(ii+1)*Np,:] = hank[:-1,:].T
            b[ii*Np:(ii+1)*Np,:] = hank[-1:,:].T
        # Solve
        w = self.solve_polynomial(A,b)
        Lam = self.roots(w)
        Lam = np.sort(Lam)
        #Lam = Lam[::2] # Remove conjugate pairs
        modal = self.poles_to_modal( Lam , dt )

        #Solve for Modal Participation
        Apq = self.solve_polynomial((Lam **  (np.arange(0,len(h))[:,None])),h)

        # Build id
        id = []
        for kk in range(D):
            id.append(modal.copy())
            id[-1:][0]['Apq'] = Apq[:,kk]

        return id
        





class PolynomialFrequency(PolynomialID):

    def __init__(self, order=None):
        super().__init__(order)


class RFP(PolynomialFrequency):

    def __init__(self, order=None):
        super().__init__(order)

    def __call__(self, H, w):

        m = self.order -1
        n = self.order

        P = (1j * w) ** np.arange(0,self.order)
        PH = np.conj(P).T
        T = (H * P)
        TH = np.conj(T).T
        W = (H * ((1j * w) ** self.order))

        X = -np.real( PH @ T )
        Y =  np.real( PH @ P )
        Z =  np.real( TH @ T )

        G =  np.real( PH @ W )
        F = -np.real( TH @ W )

        A = np.block([[Y,X],[X.T,Z]])
        b = np.vstack((G,F))

        YI = inv(Y)
        ZI = inv(Z)

        #theta = np.linalg.solve(A,b)
        AI = np.block(
            [[inv(Y - X @ ZI @ X.T), np.zeros_like(X)],
             [np.zeros_like(X), inv(Z - X.T @ YI @ X)]]) \
             @ np.block(
                 [[np.identity(self.order),-X@ZI],
                 [-X.T@YI,np.identity(self.order)]])

        theta = AI @ b

        theta = np.linalg.solve(A,b)
        #Lam = self.roots(np.flip(np.split(theta,2)[1]))
        Lam = np.roots(np.flip(np.append(theta[self.order:,0],1)))
        return Lam
        