import numpy as np
from scipy.linalg import expm
from pyma import postprocess


def test_ss_to_modal(default_system):

    M, C, K = default_system

    lam, Phi = np.linalg.eig(np.linalg.solve(M, K))

    wn = np.sqrt(lam)
    ord = np.argsort(wn)
    wn = wn[ord]
    Mt = Phi.T.dot(M.dot(Phi))
    Kt = Phi.T.dot(K.dot(Phi))
    Ct = Phi.T.dot(C.dot(Phi))
    zeta = np.diag(Ct) / (2 * np.sqrt(np.diag(Kt * Mt)))
    zeta = zeta[ord]
    Phi = Phi[:, ord]
    dt = 0.01

    A = np.block(
        [
            [np.zeros((2, 2)), np.identity(2)],
            [-np.linalg.solve(M, K), -np.linalg.solve(M, C)],
        ]
    )
    A = expm(A * dt)

    C = np.hstack((np.eye(2), np.zeros((2, 2))))

    wnp, zp, Pp = postprocess.ss_to_modal(A, C, dt)
    Pp = Pp / Pp[0, :]
    Phi = Phi / Phi[0, :]

    assert np.allclose(wn, wnp[::2])
    assert np.allclose(zeta, zp[::2])
    assert np.allclose(Phi, Pp[:, ::2])


def test_modal_assurance(default_system):

    M, C, K = default_system

    lam, Phi = np.linalg.eig(np.linalg.solve(M, K))

    MAC = postprocess.modal_assurance(Phi, Phi)

    # One at a time without broadcasting
    single_mac = (
        lambda p1, p2: np.abs(np.inner(p1, p2)) ** 2
        / np.inner(p1, p1)
        / np.inner(p2, p2)
    )

    for i in range(Phi.shape[1]):
        for j in range(Phi.shape[1]):
            assert np.allclose(MAC[i, j], single_mac(Phi[:, i], Phi[:, j]))

    # MAC with itself = 1
    assert np.all(np.diag(MAC) == 1)

    # MAC invariant to scaling on the mode shapes
    Phi2 = Phi * 10

    MAC2 = postprocess.modal_assurance(Phi, Phi2)

    for i in range(Phi.shape[1]):
        for j in range(Phi.shape[1]):
            assert np.allclose(MAC2[i, j], single_mac(Phi2[:, i], Phi2[:, j]))

    assert np.allclose(MAC, MAC2)
    assert np.all(np.diag(MAC) == 1)
