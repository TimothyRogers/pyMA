'''
Subspace Identification 
'''

from pyma import utils
import numpy as np
from numpy.linalg import pinv, inv, svd
import scipy as sp

class Subspace():

    def __init__(self, y=None, u=None, order=2):
        '''
        Inputs: y     - output data - np.ndarray T x Do
                u     - input data  - np.ndarray T x Di
                order - model order - int/list of ints/ np.array of ints
        '''
        self._order = 0
        self._u     = None
        self._y     = None
        self._Di    = 0
        self._Do    = 0
        
        self.u = u
        self.y = y
        self.order = order

    
    def LQ(self,A):

        R = np.linalg.qr(A.T,'r').T
        return R

    def _update_hankels(self):
        
        if self.u is not None:
            U = utils.block_hankel(self.u.T,(self.order*2)-1)
            self._Up = U[:self.order*self._Di,:]
            self._Uf = U[self.order*self._Di:,:]
        else:
            self._Up = None
            self._Uf = None


        if self.y is not None:
            Y = utils.block_hankel(self.y.T,(self.order*2)-1)
            self._Yp = Y[:self.order*self._Do,:]
            self._Yf = Y[self.order*self._Do:,:]

        if self.u is not None and Y.shape[1] != U.shape[1]:
            raise ValueError
        else:
            self._N = Y.shape[1]


    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = order
        self._update_hankels()

    @property 
    def u(self):
        return self._u

    @u.setter
    def u(self,u):
        self._u = u
        if u is not None:
            self._Di = u.shape[1]
            self._update_hankels 

    @property 
    def y(self):
        return self._y

    @y.setter
    def y(self,y):
        self._y = y
        if y is not None:
            self._Do = y.shape[1]
            self._update_hankels 


class MOESP(Subspace):

    
    def __init__(self, y=None, u=None, order=2):

        super().__init__(y=y, u=u, order=order)


    def __call__(self):

        # LQ decomposition
        k = self.order*2
        R = self.LQ(np.vstack((self._Up,self._Uf,self._Yp,self._Yf))/self._Up.shape[1])
        L11 = R[:k*self._Di,:k*self._Di]
        L21 = R[k*self._Di:,:k*self._Di]
        L22 = R[k*self._Di:,k*self._Di:]

        # SVD decomposition
        U, S, _ = np.linalg.svd(L22)

        # No. sing vals
        n = 4

        # Extended observability        
        Ok = U[:,:n] @ np.diag(np.sqrt(S[:n]))

        # C is first Do block rows
        C = Ok[:self._Do,:]

        # A from Eq 6.42 Katayama
        A = np.linalg.pinv(Ok[:-self._Do, :]) @ Ok[self._Do:,:]

        # Finding B and D by least squares
        # Building linear matrices
        #MBT = U[:,n:].T @ (L21 @ np.linalg.inv(L11))
        MBT = U[:,n:].T @ (R[k*self._Di+self._Do:,:] @ np.linalg.pinv(R[self._Di:2*self._Di,:]))
        U2T = U[:,n:].T
        LL = np.zeros((k*(k*self._Do-n),self._Do+n))
        MM = np.zeros((k*(k*self._Do-n),self._Di))
        for i in range(2*self.order):
            MM[i*(k*self._Do-n):(i+1)*(k*self._Do-n),:] = MBT[:,i*self._Di:(i+1)*self._Di]
            LL[i*(k*self._Do-n):(i+1)*(k*self._Do-n),:] = np.hstack((
                U2T[:,i*self._Do:(i+1)*self._Do],
                U2T[:,(i+1)*self._Do:] @ Ok[:-(i+1)*self._Do,:]
            ))

        DB = np.linalg.pinv(LL) @ MM
        D = DB[:self._Do,:]
        B = DB[self._Do:,:]

        return A, B, C, D

        

class ORT(Subspace):

    def __init__(self, y=None, u=None, order=2):

        super().__init__(y=y, u=u, order=order)

    def __call__(self):

        # Decompose Eq 9.48 Katayama
        R = self.LQ(np.vstack((self._Uf,self._Up,self._Yp,self._Yf)))

        # Extract appropriate matrix 
        L42 = R[self.order*(self._Di*2+self._Do),self.order*self._Di:self.order*self._Di*2]

        # SVD Eq 9.52 Katayama
        U, S, V = np.linalg.svd(L42)

        # Number of interesting SVs
        n = 4

        Ok = U[:,:n] @ np.diag(S[:n])

        # Eq 9.53 Katayama
        Ad = np.linalg.pinv(Ok[:-self._Do,:]) @ Ok[self._Do,:]
        Cd = Ok[:self._Do,:]

class SSI(Subspace):

    def __init__(self, y=None, u=None, order=2):

        super().__init__(y=y, u=u, order=order)


    def __call__(self):

        # LQ Decomposition
        R = self.LQ(np.vstack((
            self._Up,
            self._Uf,
            self._Yp,
            self._Yf
        )))

        # All LQ Decomposed
        Yf = R[self.order*(self._Di*2+self._Do):,:]
        Uf = R[self.order*self._Di:self.order*self._Di*2,:self.order*self._Di*2]
        Wp = np.vstack((
            R[:self.order*self._Di,:],
            R[self.order*self._Di*2:self.order*(self._Di*2+self._Do),:]
        ))
        # Perpendicular future
        Yfp = np.hstack((
            Yf[:,:self.order*self._Di*2] - (Yf[:,:self.order*self._Di*2] @ np.linalg.pinv(Uf)) @ Uf,
            Yf[:,self.order*self._Di*2:]
        ))
        # Perpendicular past
        Wpp = np.hstack((
            Wp[:,:self.order*self._Di*2] - (Wp[:,:self.order*self._Di*2] @ np.linalg.pinv(Uf)) @ Uf,
            Wp[:,self.order*self._Di*2:]
        )) 

        # Oblique Projection 
        Oi = (Yfp @ np.linalg.pinv(Wpp)) @ Wp

        # Oi @ Pi_{Uf perpendicular}
        OiP = np.hstack((
            Oi[:,:self.order*self._Di*2] - (Oi[:,:self.order*self._Di*2] @ np.linalg.pinv(Uf) ) @ Uf,
            Oi[:,self.order*self._Di*2:]
        ))
        
        # SVD 
        U, S, _ = np.linalg.svd(OiP)

        # Number of SVs
        n = 4

        # Extended Observability
        Gamma = U[:,:n] @ np.diag(np.sqrt(S[:n])) 
        GammaUBar = Gamma[:self._Do,:]
        GammaI = np.linalg.pinv(Gamma)
        GammaUBarI = np.linalg.pinv(GammaUBar)

        # Now to solve the big equation for A and C
        #TODO Finish this...
        AC = np.vstack((
            GammaUBarI @ R[self.order*(self._Di*2+self._Do)+self._Do:,:self.order*(self._Di*2+self._Do)+self._Do],
            R[self.order*(self._Di*2+self._Do):self.order*(self._Di*2+self._Do)+self._Do,:self.order*(self._Di*2+self._Do)+self._Do]
        )) @ np.linalg.pinv(np.vstack((
            np.hstack((
                GammaI @ Yf,
                np.zeros(n,self._Do)
            )),
            R[self.order*self._Di:2*self.order*self._Di,:self.order*(2*self._Di+self._Do)+self._Do]
        )))

class CCA(Subspace):

    def __init__(self, y=None, u=None, order=2):

        super().__init__(y=y, u=u, order=order)

    def __call__(self):

        R = self.LQ(np.vstack((
            self._Uf,
            self._Up,
            self._Yp,
            self._Yf
        )))

        Suu = R[:self.order*self._Di,:self.order*self._Di]
        Sup = R[:self.order*self._Di,self.order*self._Di:self.order*(self._Di*2+self._Do)]
        Sup = R[:self.order*self._Di,self.order*(self._Di*2+self._Do):]

        

class VODMSTO3(Subspace):

    def __init__(self, y=None, u=None, order=2):

        super().__init__(y=y, u=u, order=order)


    def __call__(self):

        # i = order
        # l = Do
        # j = N-2*i+1

        # Lower Triangular QR Factor
        R = self.LQ(np.vstack((self._Yp,self._Yf)))
        #R = R[:2*self.order*self._Do,:2*self.order*self._Do]

        # Ob_i and Ob_{i-1}
        Ob = R[self._Do*self.order:,:self._Do*self.order]
        Obm = R[self._Do*(self.order+1):,:self._Do*(self.order+1)]

        # Weighting for CVA
        W1 = self.LQ(R[self._Do*self.order:,:].T)
        W1 = W1[:self._Do*self.order,:self._Do*self.order]

        # SVD of O
        U, S, _ = np.linalg.svd(np.linalg.pinv(W1) @ Ob)

        # Also for CVA
        U = W1 @ U

        # Model order - do not hardcode forever Tim
        n = 4
        # Truncate SVD
        U = U[:,:n]
        S = S[:n]

        # Gamma and Gamma_
        Gamma = U @ np.diag(np.sqrt(S))
        Gamma_ = Gamma[:-self._Do,:]

        # State vector estimates
        Xhat = np.linalg.pinv(Gamma) @ Ob
        Xhatp = np.linalg.pinv(Gamma_) @ Obm

        # Solve Least Squares
        Tl = np.vstack((
                    Xhatp,
                     R[self._Do*self.order:self._Do*(self.order+1),:self._Do*(self.order+1)]
                     ))
        Tr = np.hstack((
                    Xhat,
                    np.zeros((n,self._Do))
                ))
            
        AC = Tl @ np.linalg.pinv(Tr)

        # A and C Matrices
        A = AC[:n,:n]
        C = AC[n:n+self._Do,:n]

        # Q matrix
        resid = Tl - AC @ Tr
        Pi = resid @ resid.T / resid.shape[1] # CHECK THIS
        Q = Pi[:n,:n]
        S = Pi[:n,n:]
        R = Pi[n:,n:]

        # Decouple system
        RI = np.linalg.inv(R)
        #A = A - S @ ( RI @ C)
        #Q = Q - S @ ( RI @ S.T)

        # Solve Lyapunov for Sig
        Sig = sp.linalg.solve_discrete_lyapunov(A,Q)
        G = A @ (Sig @ C.T) + S
        Lam0 = C @ (Sig @ C.T) + R

        # P and K
        # P = A P A' + (G - A P C') (L0 - C P C')^{-1} (G - A P C')' 
        # 0 = A P A' - P - (A P C' + G) (C P C' + L0)^{-1} ( A P C' + G)' 
        # So in scipy function 
        # a = A
        # b = C
        # q = np.zeros_like(A)
        # r = Lam0
        # s = G
        P = sp.linalg.solve_discrete_are(
            a=A.T,
            q=np.zeros_like(A),
            b=C.T,
            r=Lam0.T,
            s=G)
        R = Lam0 - C@(P@C.T)
        K = (G - A@(P@C.T))@np.linalg.inv(R)

        return A, C, Q







class VODM3(Subspace):

    def __init__(self, y=None, u=None, order=2):

        super().__init__(y=y, u=u, order=order)

    def __call__(self):

        W = np.vstack((self._Up,self._Uf,self._Yp,self._Yf))
        R = self.LQ(W/np.sqrt(self._N))

        m = self._Di
        l = self._Do
        i = self.order

        # Yf
        Rf = R[(2*m+l)*i:,:] 
        # Uf
        Ru = R[m*i:2*m*i,:2*m*i]
        # Up Yp
        Rp = np.hstack((
            R[:m*i,:],
            R[2*m*i:(2*m+l)*i,:]    
        ))

        # Now perpendicular projections
        Rfperp = np.hstack((
            Rf[:,:m*i*2] - np.linalg.lstsq(Rf[:,:2*m*i].T, Ru.T)[0] @ Ru.T,
            Rf[:,m*i*2:2*(m+l)*i]
        ))
        Rpperp = np.hstack((
            Rp[:,:m*i*2] - (Rp[:,:2*m*i] @ pinv(Ru)) @ Ru,
            Rp[:,m*i*2:2*(m+l)*i]
        ))
        
        # Oblique projection
        Ob = (Rfperp @ pinv(Rpperp)) @ Rp
        Ob = np.hstack((
            Ob[:,:m*i*2] - (Ob[:,:m*i*2] @ pinv(Ru))@Ru,
            Ob[:,m*i*2:2*(m+l)*i]
        ))

        U, S, V = svd(Ob)
        n = 4

        U1 = U[:,:n]

        Gam = U1 @ np.diag(np.sqrt(S[:n]))
        Gam_min = Gam[:-l,:]

        Tr = np.vstack((
            np.hstack((pinv(Gam) @ Rf, np.zeros((n,1)))),
            R[m*i:2*m*i,:]
        ))

        Tl = np.vstack((
            pinv(Gam_min) @ Rf,
            Rf
        ))

        S = np.linalg.lstsq(Tl,Tr)


        #A = S[:n,:n]
        #C = S[n:n+l,:n]
        #res = LHS - S @ RHS

        # L = R[(m*i*2+l*i):,:(m*i*2+l*i)] @ np.linalg.pinv(R[:(m*i*2+l*i),:(m*i*2+l*i)])
        # LUp = L[:,:m*i]
        # LYp = L[:,2*l*i:]

        # Pi = np.eye(2*m*i) - R[m*i:2*m*i,:2*m*i].T @ np.linalg.inv(R[m*i:2*m*i,:2*m*i] @ R[m*i:2*m*i,:2*m*i].T) @ R[m*i:2*m*i,:2*m*i]

        # O = np.hstack((
        #     (LUp @ R[:m*i,:2*m*i] + LYp @ R[2*m*i:2*m*i+l*i,:2*m*i]) @ Pi, \
        #     LYp @ R[2*m*i:2*m*i+l*i,2*m*i:2*m*i+l*i]
        # ))
        # U, S, V = np.linalg.svd(O,full_matrices=False)

        # Gam_i = U @ np.diag(np.sqrt(S))
        # Gam_im1 = Gam_i[:-l,:]

        # Tl = np.vstack((
        #         np.linalg.pinv(Gam_im1) @ R[(2*m*i+l*(i+1)):,:(2*m*i+l*(i+1))], \
        #         R[(2*m*i+l*i):(2*m*i+l*(i+1)),:(2*m*i+l*(i+1))]
        #     ))

        # Tr = np.vstack((
        #         np.linalg.pinv(Gam_i) @ R[(2*m*i+l*i):,:(2*m*i+l*(i+1))], \
        #         R[m*i:m*i*2,:(2*m*i+l*(i+1))]
        #     ))

        # S = Tl @ np.linalg.pinv(Tr)
        # A = S[:self.order,:self.order]
        # C = S[self.order:self.order+self._Do,:self.order]

        # GG = C
        # for k in range(i+1):
        #     GG = np.vstack((GG,np.vstack(GG[-i:,:]@A)))

        # #P = Tl - np.vstack((A,C)) @ Tr[:(-l*i),:]
        # #Q = Tr[:l*i,:]

        # res = ( Tl - S @ Tr )
        # Pi = res@res.T

        # return A, C