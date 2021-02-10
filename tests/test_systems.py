'''
Testing Systems
'''
import pytest
import numpy as np

from pyma import systems

def test_spatial_model():

    M = np.eye(2)
    K = np.array([[20, -10],[-10, 20]])
    C = 0.05*M + 0.001*K

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


def test_frf():
    
    M = np.eye(2)
    K = np.array([[20, -10],[-10, 20]])
    C = 0.05*M + 0.001*K

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


