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
    B = np.block([[np.zeros_like(M)], [np.linalg.inv(M)]])

    L = np.flipud(np.eye(4, 2))
    q = np.eye(2)
    Q = L @ (q @ L.T)
    R = 1e-5 * np.eye(2)

    dt = 1e-2  # 100Hz sample freq

    ssm = systems.StateSpace(A=A, B=B, Q=Q, R=R, dt=dt)

    x, y = ssm.simulate(T=5e3)

    return y
