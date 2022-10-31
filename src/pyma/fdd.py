# Frequency Domain Decomposition

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Any

from scipy.signal import csd

class SVS():
    """Singular Value Spectrum

    Object for working with singular value spectrum

    """

    opts = {
        "nfft"          : 1024,
        "noverlap"      : None,
        "scaling"       : "spectrum",
        "window"        : "hann"
    }

    def __init__(self, user_opts:dict={}) -> None:

        self.opts.update(user_opts)
        self.opts['nperseg'] = self.opts['nfft']
        

    def __call__(self, data:NDArray[np.float64]) -> Tuple[NDArray[np.float64],NDArray[np.float64]]:
        
        N_time, N_chan = data.shape

        # Compute Cross Spectral Density Matrix
        # It would be nice if there was a more efficient way to do this...
        G = np.empty((self.opts['nfft']//2+1, N_chan, N_chan))
        for i in np.arange(N_chan):
            for j in np.arange(i, N_chan):
                w, G[:,i,j] = csd(data[:,i], data[:,j], **self.opts)
                G[:,j,i] = G[:,i,j]
        
        U,S,_ = np.linalg.svd(G, hermitian=True)

        return S, U, w

class FDD():

    def __init__(self) -> None:
        pass


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    

