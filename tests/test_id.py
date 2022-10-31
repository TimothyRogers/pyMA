'''
Test identification methods
'''

import pytest
import numpy as np
from numpy import pi
import scipy as sp
from matplotlib import pyplot as plt

from pyma import systems
from pyma import utils
from pyma import subspace
from pyma import polynomial

@pytest.fixture(params=[3,4,6,10])
def default_system(request):

    k = 1e4
    c = 20
    m = 1

    M = m*np.eye(request.param)
    K = np.diag(-k*np.ones(request.param-1),k=-1) + \
         np.diag(-k*np.ones(request.param-1),k=1) + \
         np.diag(2*k*np.ones(request.param),k=0)    
    #C = np.diag(-c*np.ones(request.param-1),k=-1) + \
    #     np.diag(-c*np.ones(request.param-1),k=1) + \
    #     np.diag(2*c*np.ones(request.param),k=0)    
    C = 1e-4*K + 0.1*M
         
    MI = np.linalg.inv(M)
    wn, Phi = np.linalg.eig(MI @ K)
    wn = np.sqrt(wn)
    Mrr = np.diag(Phi.T @ (M @ Phi))
    PhiTilde = Phi / Mrr
    zeta = np.diag(PhiTilde.T @ ( C @ PhiTilde )) / 2 / wn
    
    return M, C, K, wn, zeta, Phi

@pytest.fixture
def default_spatial_system(default_system):
    
    M, C, K, wn, zeta, Phi = default_system
    sys = systems.SpatialModel(M=M, C=C, K=K)

    return sys


@pytest.fixture
def default_test_data(default_system):
    fs = 128
    dt = 1/fs

    M, C, K, _, _, _ = default_system
    dofs = M.shape[0]

    MI = np.linalg.inv(M)
    A = np.block([[np.eye(dofs,dofs*2,dofs)],[-MI@K, -MI@C]])
    B = np.vstack((np.zeros_like(MI),MI))
    C = np.block([[-MI@K, -MI@C]])
    D = MI
    Q = np.zeros((4,4))
    Q[3,3] = 1e-6
    R = 0.001*np.eye(dofs)

    #ssm = systems.StateSpace(A=A, B=B, C=C, D=D, Q=Q, R=R, dt=dt)
    ssm = systems.StateSpace(A=A, B=B, C=C, D=D, Q=Q, R=R, dt=dt)
    ssm.discretise()

    np.random.seed(1)
    t = np.arange(0,20,dt)
    wn = np.abs(np.linalg.eig(A)[0])
    #U = 0.01*np.tile(np.sin(25*t)+np.random.standard_normal(t.shape),(2,1))
    freqs = np.random.rand(1000,1)*(1.5*max(wn)-0.2*min(wn)) + 0.2*min(wn)
    U = np.vstack((
        utils.generate_multisine(freqs, t),
        utils.generate_multisine(freqs, t)
    ))

    _, Y = ssm.simulate(u=U,T=20*fs-1)

    #U = U[:,1:].T
    #Y = Y[:,1:].T
    
    return Y, U, ssm


'''
def test_ssi(default_system, default_test_data):
    M, C, K, wn, zeta, Phi = default_system
    Y, U, ssm = default_test_data

    alg = subspace.SSI(y=Y[:,1:].T, u=U[:,1:].T, order=M.shape[0]*4)
    AA,BB,CC,DD = alg()
    #alg()
    wnest = np.abs(np.log(np.linalg.eig(AA)[0])*128)
    zest = -np.real(np.log(np.linalg.eig(AA)[0])*128)/wnest

    ssm_est = systems.StateSpace(A=AA,B=BB,C=CC,D=DD,dt=1/128,continuous=False)


    _, Y2 = ssm_est.simulate(u=U,T=20*128-1)

'''

def test_RFP(default_spatial_system):

    fs = 512
    dt = 1/fs
    secs = 1

    w = np.arange(0 , fs/2 + secs/fs , 1/secs)

    sys = default_spatial_system
    H = sys.frf(w=w)
    #H = H + 1**-6*np.random.standard_normal(H.shape) + + 1**-6*np.random.standard_normal(H.shape)*1j

    N = H.shape[2]
    alg = polynomial.RFP(len(sys.Omega)*2)
    id = alg(np.squeeze(H[0,0,:]).T[:,None]/1,w[:,None]/1)
    print(id)


def test_CE(default_spatial_system):

    fs = 512
    dt = 1/fs
    secs = 10

    w = np.arange(0 , fs/2 , 1/secs)

    sys = default_spatial_system
    H = sys.frf(w=w)
    h = np.fft.irfft(H*fs,axis=2)[:,:,:fs] # IRF should really by in the system object
    h = np.squeeze(h[:,0,:]).T


    alg = polynomial.ComplexExponential(order=len(sys.Omega)*2)
    id = alg(h[10:,:], dt)

    assert(np.allclose(id[-1:][0]['wn'][::2]/2/pi,sys.Omega,rtol=0.1)) # Nat Freq within 5 %
    assert(np.allclose(id[-1:][0]['zeta'][::2],sys.Zeta,rtol=0.5)) # Nat Freq within 50 %

    
    plt.figure()
    Hest = np.zeros_like(H)
    for ii,jk in enumerate(id):
        R = len(jk['wn'])
        for rr in range(R):
            Hest[ii,0,:] += jk['Apq'][rr]/(1j*w*2*np.pi - jk['lam'][rr])
        
        plt.subplot(len(sys.Omega),1,ii+1)
        plt.semilogy(w,np.abs(H[ii,0,:]))
        plt.semilogy(w,np.abs(Hest[ii,0,:]))

    plt.show()
    
def test_LSCE(default_spatial_system):

    fs = 512
    dt = 1/fs
    secs = 10

    w = np.arange(0 , fs/2 , 1/secs)

    sys = default_spatial_system
    H = sys.frf(w=w)
    h = np.fft.irfft(H*fs,axis=2)[:,:,:fs] # IRF should really by in the system object
    h = np.squeeze(h[:,0,:]).T


    alg = polynomial.LSCE(order=len(sys.Omega)*2)
    id = alg(h[10:,:], dt)

    assert(np.allclose(id[-1:][0]['wn'][::2]/2/pi,sys.Omega,rtol=0.05)) # Nat Freq within 5 %
    assert(np.allclose(id[-1:][0]['zeta'][::2],sys.Zeta,rtol=1)) # Nat Freq within 100 %

    
    plt.figure()
    Hest = np.zeros_like(H)
    for ii,jk in enumerate(id):
        R = len(jk['wn'])
        for rr in range(R):
            Hest[ii,0,:] += jk['Apq'][rr]/(1j*w*2*np.pi - jk['lam'][rr])
        
        plt.subplot(len(sys.Omega),1,ii+1)
        plt.semilogy(w,np.abs(H[ii,0,:]))
        plt.semilogy(w,np.abs(Hest[ii,0,:]))

    plt.show()


    
