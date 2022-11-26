import numpy as np
from pyma import preprocess


def test_block_hankel():

    X = np.reshape(np.arange(0, 24), (8, 3), order="F")
    H = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [8, 9, 10, 11, 12, 13],
            [16, 17, 18, 19, 20, 21],
            [1, 2, 3, 4, 5, 6],
            [9, 10, 11, 12, 13, 14],
            [17, 18, 19, 20, 21, 22],
            [2, 3, 4, 5, 6, 7],
            [10, 11, 12, 13, 14, 15],
            [18, 19, 20, 21, 22, 23],
        ]
    )

    assert preprocess.block_hankel(X.T, 3)[0, 0] == X[0, 0]
    assert preprocess.block_hankel(X.T, 3)[-1, -1] == X[-1, -1]
    assert (preprocess.block_hankel(X.T, 3) == H).all()
    for i in range(6):
        assert (preprocess.block_hankel(X.T, 3, i) == H[:, :i]).all()
