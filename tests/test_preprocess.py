import numpy as np
from pyma import preprocess


def test_block_hankel():

    X = np.reshape(np.arange(0, 18), (6, 3), order="F")
    H = np.array(
        [
            [0, 1, 2, 3],
            [6, 7, 8, 9],
            [12, 13, 14, 15],
            [1, 2, 3, 4],
            [7, 8, 9, 10],
            [13, 14, 15, 16],
            [2, 3, 4, 5],
            [8, 9, 10, 11],
            [14, 15, 16, 17],
        ]
    )

    assert (preprocess.block_hankel(X.T, 2) == H).all()
