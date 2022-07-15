'''
Testing Systems
'''
import pytest
import numpy as np
import scipy as sp

from pyma import systems
from pyma import utils

@pytest.fixture
def default_system():
    
    M = np.eye(2)
    K = np.array([[20, -10],[-10, 20]])
    C = 0.05*M + 0.001*K

    return M, C, K

def test_spatial_model(default_system):

    M, C, K = default_system

    model = systems.SpatialModel(M=M,C=C,K=K)

    # Property assignment
    assert((model.M == M).all())
    assert((model.C == C).all())
    assert((model.K == K).all())

    # Check DOFS
    with pytest.raises(ValueError):
            model.M =  np.array([[3,2,1],[2,3,2],[1,2,3]])

    # Undamped Modal
    model = systems.SpatialModel(M=M,C=None,K=K)
    modal = model.as_modal()
    model.update_modal()

    assert(np.isclose(np.sort(modal.Omega**2),np.sort([10,30])).all())
    assert(np.isclose(modal.Phi/modal.Phi[0,:],np.array([[1,1],[-1,1]])).all()) 

    # Damped Modal
    model = systems.SpatialModel(M=M,C=C,K=K)
    modal = model.as_modal()
    model.update_modal()

    assert((model.Omega == modal.Omega).all())
    assert((model.Zeta == modal.Zeta).all())
    assert(np.isclose(model.Phi/model.Phi[0,:],modal.Phi/modal.Phi[0,:]).all())

    assert(np.isclose(np.sort(modal.Omega**2),np.sort([10,30])).all())
    assert(np.isclose(modal.Phi/modal.Phi[0,:],np.array([[1,1],[-1,1]])).all())

    model = systems.SpatialModel(M=1,C=20,K=1e4)
    model.as_modal()


def test_frf(default_system):
    
    M, C, K = default_system

    wn, phi = np.linalg.eig(np.linalg.inv(M)@K)
    wn = np.sqrt(wn)
    Mrr = np.diag(phi.T @ (M @ phi))
    phi = phi/Mrr
    Krr = np.diag(phi.T @ (K @ phi))
    Crr = np.diag(phi.T @ (C @ phi))
    
    w = np.linspace(0,1.2*wn.max(),1024)

    model = systems.SpatialModel(M=M,C=C,K=K)
    frf = model.frf(J=1,K=0)
    frf = model.frf()

    frf_test = np.zeros((2,2,1024),dtype=np.clongdouble)


    for j in range(2):
        for k in range(2):
            for d in range(2):
                frf_test[j,k,:] = frf_test[j,k,:] + (phi[d,j] * phi[d,k]) / (Krr[d] - (w**2)*Mrr[d] + 1j*w*Crr[d])
    
    assert(np.allclose(frf_test,frf))


def test_ssm(default_system):

    M, C, K = default_system

    MI = np.linalg.inv(M)
    model = systems.SpatialModel(M=M,C=C,K=K)

    A = model.first_order_form()
    B = np.block([[np.zeros_like(M)],[np.linalg.inv(M)]])
    C = np.hstack((-MI@K,-MI@C))
    D = MI

    dt = 1e-2 # 1000Hz sample freq

    # Deterministic SSM
    ssm = systems.StateSpace(A=A, B=B, C=C, D=D, dt=dt)

    with pytest.raises(NotImplementedError):
        ssm.frf()

    x, y = ssm.simulate(T=2000)

    assert( (ssm.A == A).all())
    assert( (ssm.B == B).all())
    assert( (ssm.C == C).all())
    assert( (ssm.D == D).all())

    ssm.discretise()

    assert( (ssm.A == sp.linalg.expm(dt*A)).all() )

    # Check eigenvalues are preserved in discretisation
    assert( np.allclose(np.linalg.eig(A)[0], np.log(np.linalg.eig(ssm.A)[0]) / dt) )


    # Stochastic SSM
    L = np.flipud(np.eye(4,2))
    q = np.eye(2)
    Q = L @ (q @ L.T)
    R = np.eye(2)
    ssm = systems.StateSpace(A=A, B=B, C=C, D=D, Q=Q, R=R, dt=dt)

    ssm.discretise()
    np.random.seed(1)
    with pytest.raises(utils.SimulationError):
        ssm.simulate()
    x, y = ssm.simulate(T=3e4)

    t = np.arange(0,10,dt)
    u = np.block([[np.zeros((1,len(t)))],[np.sin(15*2*np.pi*t)]])
    x, y = ssm.simulate(u=u)
    x, y = ssm.simulate(u=u, T=100)

    x0 = 10*np.ones(4)
    x, y = ssm.simulate(x0=x0,T=1e4)

    # Eventually we should handle lists as inputs instead of np.ndarrays
    with pytest.raises(ValueError):
        ssm.simulate(u=[1]*100)