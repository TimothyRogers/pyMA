"""
Stochastic Subspace OMA Methods
"""

from . import OMA
from pyma.utils import lq, ModalType
from pyma.preprocess import block_hankel
from pyma.postprocess import ss_to_modal

import numpy as np
from typing import Any, Union, List, Tuple
from collections.abc import Iterable


class SSI(OMA):
    """Stochastic Subspace Based Operational Modal Analysis"""

    # Default options dictionary
    _opts = {
        "max_model_order": 10,  # Max number of modes to look for defines Hankel size
        "N": None,  # Fix Hankel width?
        "model_order": -1,  # Model orders to compute modes at
        "dt": 1,  # Sample time of data
    }

    def __init__(self, opts={}) -> None:
        """Setup SSI

        No setup required really, only thing is to update opts as in OMA

        Args:
            opts (dict, optional): User options. Defaults to {}, i.e. use defaults.
        """
        super().__init__(opts)

    def __call__(
        self, y: np.ndarray, *args: Any, **kwargs: Any
    ) -> Union[ModalType, List[ModalType]]:
        """Perform SSI OMA

        Analysis of the acquired dataset, this is presented as a call to avoid internalising the
        data. Eventually other SSI methods supported through this call in conjunction with the
        user options dict.

        Args:
            y (np.ndarray): measured (usually acceleration) data in ( P, T ) array for P channels

        Returns:
            Union[ModalType, List[ModalType]]: modal properties at requested model order(s)
        """

        return self._alg_A(y, *args, **kwargs)

    def _alg_A(
        self, y: np.ndarray, *args: Any, **kwargs: Any
    ) -> Union[ModalType, List[ModalType]]:
        """Use Algorithm A from Katayama

        This methodology solves least squares on the observability matrix for A and takes the top
        block row as the estimate of C.

        Args:
            y (np.ndarray):  measured (usually acceleration) data in ( P, T ) array for P channels

        Raises:
            ValueError: invalid model order provided

        Returns:
            Union[ModalType, List[ModalType]]: modal properties at requested model order(s)
        """

        # Number of output channels
        p = y.shape[0]

        # Build the block hankel matrix
        Y = block_hankel(y, order=self.opts["max_model_order"] * 2, N=self.opts["N"])

        # Get covariances and compute square roots
        Spp, Sfp, Sff = self._compute_covariances(Y / np.sqrt(Y.shape[1]), p)
        LL = np.linalg.cholesky(Sff)
        MM = np.linalg.cholesky(Spp)

        # SVD solves generalised eigenvalue problem for CCA
        # We don't need the controlability matrix so no V
        U, S, _ = np.linalg.svd(
            np.linalg.solve(LL, Sfp).dot(
                np.linalg.solve(MM, np.identity(MM.shape[0])).T
            )
        )

        # For a given model order compute modal properties
        # Construct Observability matrix excluding null space
        def modal_for_order(k):
            # Only compute even model orders as expect 2nd order systems
            Obs = LL.dot(U[:, : 2 * k]).dot(np.diag(np.sqrt(S[: 2 * k])))

            # Compute A and C matrices
            A = np.linalg.pinv(Obs[:-p, :]).dot(Obs[p:, :])
            C = Obs[:p, :]

            # Modal properties
            return ss_to_modal(A, C, self.opts["dt"])

        # Compute A and C matrix for model orders
        props = []
        if self.opts["model_order"] == -1:
            # If -1 model_order compute all up to max
            for k in range(self.opts["max_model_order"]):
                props.append(modal_for_order(k + 1))
        elif isinstance(self.opts["model_order"], Iterable):
            # User provided iterable of desired model orders
            for k in self.opts["model_order"]:
                props.append(modal_for_order(k))
        elif isinstance(self.opts["model_order"], int):
            # User asked for single model order
            props = modal_for_order(self.opts["model_order"])
        else:
            raise ValueError("Invalid model order option.")

        return props

    def _compute_covariances(
        self, Y: np.ndarray, p: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute covariance matrices from Hankel matrix

        Args:
            Y (np.ndarray): block Hankel matrix of the system [[Yp],[Yf]] ( 2*p*max_model_order, N)
            p (int): number of measurement channels, i.e. block size in Hankel

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: past-past, future-past, future-future covariances
        """

        # Lower triangular
        L = lq(Y)
        midpoint = self.opts["max_model_order"] * p

        # See Katayama pp. 227
        Spp = L[:midpoint, :midpoint].dot(L[:midpoint, :midpoint].T)
        Sfp = L[midpoint:, :midpoint].dot(L[:midpoint, :midpoint].T)
        Sff = L[midpoint:, :midpoint].dot(L[midpoint:, :midpoint].T) + L[
            midpoint:, midpoint:
        ].dot(L[midpoint:, midpoint:].T)

        return Spp, Sfp, Sff
