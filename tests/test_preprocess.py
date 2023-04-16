import numpy as np
from scipy.signal import csd
from itertools import combinations_with_replacement, permutations

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


def test_pairwise_csd(default_oma):

    opts = {"fs": 1.0, "window_length": 0.2, "overlap": 0.5}

    x, sys = default_oma
    p, N = x.shape
    nperseg = np.floor(N * opts["window_length"])
    noverlap = np.floor(nperseg * opts["overlap"])

    freqs, G = preprocess.pairwise_csd(x)
    Gsp = np.empty_like(G)

    # Check against nested loop on scipy
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            freqs_sp, Gsp[i, j, :] = csd(
                x[i, :], x[j, :], nperseg=nperseg, noverlap=noverlap, detrend=False
            )

    assert np.allclose(freqs, freqs_sp)
    for i in range(G.shape[0]):
        assert np.allclose(G[i, i, :].imag, 0.0)
    # 5 % seems loose here but visually we look good
    assert np.allclose(Gsp, G, rtol=0.05)


def test_svs(default_oma):

    opts = {"fs": 1.0, "window_length": 0.2, "overlap": 0.5}

    x, sys = default_oma

    p, N = x.shape
    nperseg = np.floor(N * opts["window_length"])
    noverlap = np.floor(nperseg * opts["overlap"])

    # Defaults
    freqs, Phi, spec = preprocess.svs(x)

    assert np.all(np.imag(freqs) == 0)
    assert np.all(np.imag(Phi) == 0)
    assert np.all(np.imag(spec) == 0)
    assert np.all(Phi.shape == (nperseg / 2 + 1, p, p))
    assert np.all(spec.shape == (nperseg / 2 + 1, p))
    assert np.all(freqs.shape == (nperseg / 2 + 1,))
