""" 
Preprocessing tools 
"""

import numpy as np
from scipy.fft import rfftfreq
from scipy.signal._spectral_py import _fft_helper
from typing import Optional, Tuple


def block_hankel(X: np.ndarray, order: int, N: Optional[int] = None) -> np.ndarray:
    """Compute block Hankel matrix

    Args:
        X (np.ndarray): Data matrix of size ( p x T ) with p channels and T measurments
        order (int): Order of the Hankel matrix, i.e. H of size ( p * order, ... )
        N (Optional[int], optional): Fixed width Hankel matrix width N. Defaults to T - p*order.

    Returns:
        np.ndarray: Block Hankel matrix size ( p * order, N )
    """

    if N is None:
        D, N = X.shape
        N -= order - 1
    else:
        D, _ = X.shape
    H = np.empty((D * order, N))

    for o in range(order):
        H[o * D : (o + 1) * D, :] = X[:, o : N + o]

    return H


def pairwise_csd(X: np.ndarray, opts: dict = {}) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Pairwise CSD in Array

    Args:
        X (np.ndarray): Array of data (p channels, N data)
        opts (dict, optional): options for the CSD. Defaults to {}.

    Returns:
        Tuple[np.ndarray, np.ndarray]: frequencys (n,) and pairwise CSD (n, p, p) with n length spectra
    """

    _default_opts = {"fs": 1.0, "window_length": 0.2, "overlap": 0.5}

    opts = _default_opts | opts
    p, N = X.shape

    nperseg = np.floor(N * opts["window_length"])
    noverlap = np.floor(nperseg * opts["overlap"])
    win = np.hanning(nperseg)[None, None, :]  # For now force Hanning window

    # Bit naughty scipy says this is internal only...
    Y = _fft_helper(
        X, win, lambda d: d, int(nperseg), int(noverlap), int(nperseg), "onesided"
    )
    # Compute pairwise CSD by broadcasting
    Y = np.mean(np.conjugate(Y[:, None, :, :]) * Y[None, :, :, :], axis=2) / (
        opts["fs"] * (win**2).sum()
    )  # Scaling for CSD
    if nperseg % 2:
        Y[:, :, 1:] *= 2
    else:
        Y[:, :, 1:-1] *= 2

    # Thanks Scipy again
    freqs = rfftfreq(int(nperseg), 1 / opts["fs"])

    return freqs, Y


def svs(X: np.ndarray, opts: dict = {}) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Singular Value Spectra

    Args:
        X (np.ndarray): Array of time series data (p channels, N data)
        opts (dict, optional): options for the SVS. Defaults to {}.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: frequencys (n,), mode shape estimates U (n, p, p) and spectra (n, p)
    """

    _default_opts = {"fs": 1.0, "window_length": 0.2, "overlap": 0.5}

    opts = _default_opts | opts
    p, N = X.shape

    freqs, Y = pairwise_csd(X, opts)
    U, S, _ = np.linalg.svd(np.moveaxis(Y, -1, 0))  # cycle to broadcast SVD

    return freqs, U.real, S
