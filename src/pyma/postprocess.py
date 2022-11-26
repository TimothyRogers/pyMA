""" 
Postprocessing tools 
"""

import numpy as np
from pyma.utils import ModalType
from typing import Tuple, List


def ss_to_modal(A: np.ndarray, C: np.ndarray, dt: float) -> ModalType:
    """Convert from state-space form to modal properties

    Computing the modal properties from the state-space form of a system

    For R modes we get R natural frequencies and damping ratios in conjugate pairs and a P long vector
    for each mode shape for each mode such that Phi has shape (P, 2*R) again because of the conjugate pairs

    All modal properties are returned sorted in ascending natural frequency

    Args:
        A (np.ndarray): A matrix of the system ( 2*R, 2*R )
        C (np.ndarray): C matrix of the system ( P , 2*R )
        dt (float): sample time

    Returns:
        ModalType: modal properties (natural frequencies, damping ratios, mode shapes)
    """

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


def modal_assurance(Phi1: np.ndarray, Phi2: np.ndarray) -> np.ndarray:
    """Modal Assurance Criterion

    Compute the modal assurance criterion between two sets of mode shapes.

    Mode shapes must be a set of length P vectors stacked columnwise such that the first
    set contains R_1 mode shapes and the second set contains R_2 mode shapes.

    Args:
        Phi1 (np.ndarray): first set of mode shapes ( P, R_1 )
        Phi2 (np.ndarray): second set of mode shapes ( P, R_2 )

    Returns:
        np.ndarray: pairwise modal assurance criterion shape ( R_1, R_2 )
    """

    return (
        np.abs(Phi1.T.dot(Phi2)) ** 2
        / np.sum(Phi1**2, axis=0)[:, None]
        / np.sum(Phi2**2, axis=0)[None, :]
    )
