""" 
Postprocessing tools 
"""

import numpy as np


def ss_to_modal(A: np.ndarray, C: np.ndarray, dt: float):

    lam, Phi = np.linalg.eig(A)
    mu = np.log(lam) / dt
    wn = np.abs(mu)
    zeta = -np.real(mu) / wn
    Phi = np.real(C.dot(Phi))

    # Sort by ascending eigenvalue
    ord = np.argsort(wn)
    wn = wn[ord]
    zeta = zeta[ord]
    Phi = Phi[:, ord]

    return wn, zeta, Phi


def modal_assurance(Phi1: np.ndarray, Phi2: np.ndarray):

    return (
        np.abs(Phi1.T.dot(Phi2)) ** 2
        / np.sum(Phi1**2, axis=0)[:, None]
        / np.sum(Phi2**2, axis=0)[None, :]
    )
