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

    assert(np.isclose(np.sort(modal.Omega**2),np.sort([10,30])).all())
    assert(np.isclose(modal.Phi/modal.Phi[0,:],np.array([[1,1],[-1,1]])).all()) 

    # Damped Modal
    model = systems.SpatialModel(M=M,C=C,K=K)
    modal = model.as_modal()

    assert(np.isclose(np.sort(modal.Omega**2),np.sort([10,30])).all())
    assert(np.isclose(modal.Phi/modal.Phi[0,:],np.array([[1,1],[-1,1]])).all())

    model = systems.SpatialModel(M=1,C=20,K=1e4)
    model.as_modal()

