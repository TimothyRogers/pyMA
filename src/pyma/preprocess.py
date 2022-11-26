""" 
Preprocessing tools 
"""

import numpy as np
from typing import Optional


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
