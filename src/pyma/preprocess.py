""" 
Preprocessing tools 
"""

import numpy as np
from typing import Optional


def block_hankel(X: np.ndarray, order: int, N: Optional[int] = None):
    # Block hankel matrix order o

    if N is None:
        D, N = X.shape
        N -= order - 1
    else:
        D, _ = X.shape
    H = np.empty((D * order, N))

    for o in range(order):
        H[o * D : (o + 1) * D, :] = X[:, o : N + o]

    return H
