import pytest
import numpy as np

from pyma import systems


@pytest.fixture(scope="session")
def default_system():

    M = np.eye(2)
    K = np.array([[20, -10], [-10, 20]])
    C = 0.05 * M + 0.001 * K

    return M, C, K


@pytest.fixture(scope="session")
def default_oma(default_system):

    M, C, K = default_system

    MI = np.linalg.inv(M)
    model = systems.SpatialModel(M=M, C=C, K=K)

    A = model.first_order_form()
    L = np.flipud(np.eye(4, 2))
    q = np.eye(2)
    Q = L @ (q @ L.T)
    R = 1e-5 * np.eye(2)

    dt = 1e-2  # 100Hz sample freq

    C = np.hstack((np.identity(2), np.zeros((2, 2))))

    ssm = systems.StateSpace(A=A, C=C, Q=Q, R=R, dt=dt)
    ssm.discretise()

    x, y = ssm.simulate(T=5e3)

    return y[:, 1:], ssm
