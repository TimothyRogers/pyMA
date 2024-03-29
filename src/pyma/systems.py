"""
Representations of Dynamic Systems
"""

from typing import Union, Tuple
from pyma import utils
import numpy as np
from numpy.linalg import eig, inv
from scipy.linalg import expm
import itertools


class DynamicSystem:
    """
    Base Class for Dynamic Systems
    """

    def __init__(self):

        self.dofs = None
        self._M = None
        self._C = None
        self._K = None

        self.Omega = []
        self.Phi = []
        self.Zeta = []

    def frf(
        self,
        w: np.ndarray = None,
        J: Union[int, list, np.ndarray] = -1,
        K: Union[int, list, np.ndarray] = -1,
    ) -> np.ndarray:
        """Compute complex receptance FRF


        Args:
            w (np.ndarray, optional): range of frequencys over which to compute the FRF,
                                      if unspecified up to 1.2*max(wn). Defaults to None.
            J (Union[int, list, np.ndarray], optional): Reponse location(s). Input -1 for all dofs. Defaults to -1.
            K (Union[int, list, np.ndarray], optional): Loading location(s). Input -1 for all dofs. Defaults to -1.

        Returns:
            np.ndarray: Returns [J,K,N] array where N is the number of frequency lines
        """

        if J == -1:
            J = np.arange(0, self.dofs)

        if K == -1:
            K = np.arange(0, self.dofs)

        if w is None:
            w = np.linspace(0, 1.2 * self.Omega.max(), 1024)

        if isinstance(J, int):
            J = [J - 1]

        if isinstance(K, int):
            K = [K - 1]

        FRF = np.empty((len(J), len(K), len(w)), dtype=np.clongdouble)

        for a in J:
            for b in K:
                FRF[a, b] = self._frf(a, b, w)

        return FRF

    def _frf(self, jj: int, kk: int, w: np.ndarray) -> np.ndarray:
        """Compute single complex receptance FRF

        Args:
            jj (int): measurement location
            kk (int): forcing location
            w (np.ndarray): frequencies to compute at

        Returns:
            np.ndarray: Complex receptance FRF as [N,] array where N is number of frequency lines
        """

        frf = np.zeros(len(w), dtype=np.clongdouble)
        for r in range(self.dofs):
            rAjk = self.Phi[r, jj] * self.Phi[r, kk]
            den = (
                self.Omega[r] ** 2 - w**2 + 1j * 2 * self.Zeta[r] * self.Omega[r] * w
            )
            frf = frf + rAjk / den
        return frf


class ModalModel(DynamicSystem):
    """
    Dynamic System in Modal Coordinates - Omega, Phi
    """

    def __init__(
        self, Omega: np.ndarray = None, Phi: np.ndarray = None, Zeta: np.ndarray = None
    ):
        """Initialise modal model

        Args:
            Omega (np.ndarray, optional): Natural frequencies. Defaults to None.
            Phi (np.ndarray, optional): Mode shapes. Defaults to None.
            Zeta (np.ndarray, optional): Damping ratios. Defaults to None.
        """

        self.Omega = Omega
        self.Phi = Phi
        self.Zeta = Zeta


class SpatialModel(DynamicSystem):
    """
    Dynamic System in Spatial Coordinates - M, C, K
    """

    def __init__(
        self, M: np.ndarray = None, C: np.ndarray = None, K: np.ndarray = None
    ):
        """Initialise a spatial dynamic model

        Args:
            M (np.ndarray, optional): Mass matrix. Defaults to None.
            C (np.ndarray, optional): Damping (viscous) matrix. Defaults to None.
            K (np.ndarray, optional): Stiffness matrix. Defaults to None.
        """

        self.dofs = None
        self._M = None
        self._C = None
        self._K = None

        self.M = M
        self.C = C
        self.K = K

    def dofs_update(self, A: np.ndarray):
        """Check or update number of degress of freedom

        Args:
            A (np.ndarray): System matrix, self.dofs square
        """

        if A is not None:
            if self.dofs == None or (A.shape[0] == self.dofs):
                self.dofs = A.shape[0]
            else:
                raise (ValueError)

    @property
    def M(self):
        """Mass matrix"""
        return self._M

    @M.setter
    @utils.num_to_np
    @utils.valid_system_matrix
    def M(self, M: Union[np.ndarray, float]):
        """Setter for Mass matrix

        Convert single number input to 2D numpy array and check valid

        Args:
            M (Union[np.ndarray,float]): Mass matrix
        """
        self.dofs_update(M)
        self._M = M
        self.update_modal()

    @property
    def C(self):
        """Damping matrix"""
        return self._C

    @C.setter
    @utils.num_to_np
    @utils.valid_system_matrix
    def C(self, C: Union[np.ndarray, float]):
        """Setter for damping matrix

        Convert single number input to 2D numpy array and check valid

        Args:
            C (Union[np.ndarray,float]): Mass matrix
        """
        self.dofs_update(C)
        self._C = C
        self.update_modal()

    @property
    def K(self):
        """Stiffnss matrix"""
        return self._K

    @K.setter
    @utils.num_to_np
    @utils.valid_system_matrix
    def K(self, K: Union[np.ndarray, float]):
        """Setter for stiffness matrix

        Convert single number input to 2D numpy array and check valid

        Args:
            K (Union[np.ndarray,float]): stiffness matrix
        """
        self.dofs_update(K)
        self._K = K
        self.update_modal()

    def first_order_form(self) -> np.ndarray:
        """Return system in first order form

        Returns:
            np.ndarray: First order system matrix [2R,2R] for R DOFs
        """
        MI = inv(self.M)
        return np.block(
            [
                [np.eye(self.dofs, self.dofs * 2, self.dofs)],
                [-MI @ self.K, -MI @ self.C],
            ]
        )

    def update_modal(self):
        """Update Modal Properties

        Compute the modal properties of the system dependent on what matrices are set.

        """

        # Compute modal properties
        if self.M is not None and self.K is not None:
            lam_ud, phi = eig(inv(self.M) @ self.K)
            # Orthonormal modes
            Mrr = phi.T @ (self.M @ phi)
            phi = phi / np.diag(Mrr)

            if self.C is not None:
                # If system is damped solve the eigenvalues of the extended matrix
                lam, _ = eig(self.first_order_form())
                lam.sort()
                self.Omega = np.abs(lam[::2])
                self.Zeta = -np.real(lam[::2]) / self.Omega
            else:
                # Classic undamped eigenvalue problem
                self.Omega = np.sqrt(lam_ud)
                self.Zeta = np.zeros_like(lam_ud)
            self.Phi = phi
        else:
            self.Omega = None
            self.Zeta = None
            self.Phi = None

    def as_modal(self) -> ModalModel:
        """Return a Modal Model

        Returns:
            ModalModel: Current system as modal model
        """

        if self.C is None:
            # Undamped system
            return ModalModel(Omega=self.Omega, Phi=self.Phi)
        else:
            return ModalModel(Omega=self.Omega, Phi=self.Phi, Zeta=self.Zeta)


class StateSpace(DynamicSystem):
    """
    Dynamic system as a linear Gaussian SSM

    If continuous is True:
        A is continuous time transition matrix
        B is continuous time control matrix
        C is observation matrix
        D is shoot through matrix
        Q is L*q*L' where q is white noise spectral density
        R is measurement noise covariance

    Else:
        A is discrete time transition matrix
        B is discrete time control matrix
        C is observation matrix
        D is shoot through matrix
        Q is process noise covariance
        R is measurement noise covariance

    dt - sample time default 1 Hz

    """

    def __init__(
        self,
        A: np.ndarray = None,
        B: np.ndarray = None,
        C: np.ndarray = None,
        D: np.ndarray = None,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        continuous: bool = True,
        dt: float = 1.0,
    ):

        self._A = None
        self._B = None
        self._C = None
        self._D = None
        self._Q = None
        self._R = None
        self._Dx = 0
        self._Du = 0
        self._Dy = 0

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q
        self.R = R

        self.continuous = continuous
        self.dt = dt

    @property
    def A(self):
        return self._A

    @A.setter
    @utils.verify_dims("Dx", "Dx")
    def A(self, A: np.ndarray):
        self._A = A

    @property
    def B(self):
        return self._B

    @B.setter
    @utils.verify_dims("Dx", "Du")
    def B(self, B: np.ndarray):
        self._B = B

    @property
    def C(self):
        return self._C

    @C.setter
    @utils.verify_dims("Dy", "Dx")
    def C(self, C: np.ndarray):
        self._C = C

    @property
    def D(self):
        return self._D

    @D.setter
    @utils.verify_dims("Dy", "Du")
    def D(self, D: np.ndarray):
        self._D = D

    @property
    def R(self):
        return self._R

    @R.setter
    @utils.num_to_np
    @utils.verify_dims("Dy", "Dy")
    def R(self, R: np.ndarray):
        self._R = R

    @property
    def Q(self):
        return self._Q

    @Q.setter
    @utils.num_to_np
    @utils.verify_dims("Dx", "Dx")
    def Q(self, Q: np.ndarray):
        self._Q = Q

    def frf(
        self,
        w: np.ndarray = None,
        J: Union[int, list, np.ndarray] = None,
        K: Union[int, list, np.ndarray] = None,
    ):
        """Compute receptance FRF of SSM

        Todo:
            * Not yet implemented

        Args:
            w (np.ndarray, optional): vector of frequencies to compute at. Defaults to None.
            J (Union[int, list, np.ndarray], optional): measurement dof(s). Defaults to None.
            K (Union[int, list, np.ndarray], optional): forcing dofs(s). Defaults to None.

        Raises:
            NotImplementedError: _description_
        """
        # Need to add conversion here
        raise NotImplementedError()

    def discretise(self):
        """Discretise continuous time SSM using matrix fractions"""

        if self.continuous:

            if self.Q is not None:
                # Stochastic case
                # Using a matrix fraction decomposition to discretize
                # Note: self.Q must equal L*q*L'
                Phi = expm(
                    self.dt
                    * np.block(
                        [[self.A, self.Q], [np.zeros((self._Dx, self._Dx)), -self.A.T]]
                    )
                )
                A = Phi[0 : self._Dx, 0 : self._Dx]
                Q = Phi[0 : self._Dx, self._Dx :] @ A.T
                if self.B is not None:
                    B = inv(self.A) @ (A - np.eye(self._Dx)) @ self.B
                    self.B = B

                self.A = A
                self.Q = Q
                self.continuous = False

            else:
                # Deterministic case
                A = expm(self.dt * self.A)
                if self.B is not None:
                    B = inv(self.A) @ (A - np.eye(self._Dx)) @ self.B
                    self.B = B
                self.A = A
                self.continuous = False

    def simulate(
        self, u: np.ndarray = None, T: int = None, x0: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate response of a dynamic system

        Todo:
            * input dimension checking
            * handling unexpected types for u

        Args:
            u (np.ndarray, optional): input signal time series. Defaults to None.
            T (int, optional): number of time steps to simulate. Defaults to None.
            x0 (np.ndarray, optional): initial conditions. Defaults to None.

        Raises:
            utils.SimulationError: invalid parameters for simulation, e.g. cannot determine length
            ValueError: invalid input type u

        Returns:
            _type_: _description_
        """

        T = int(T) if T is not None else T

        Du = self._Du if self._Du != 0 else 1
        if u is None:
            if T is None:
                raise utils.SimulationError("Zero time steps in simulation.")
            u = np.zeros((Du, T + 1))
        else:
            if T is None:
                if isinstance(u, np.ndarray):
                    # First input has to be x0 input
                    T = u.shape[1] - 1
                else:
                    raise ValueError

        # x is array from t=0 to t=T
        x = np.empty((self._Dx, T + 1))
        if x0 is None:
            x[:, 0] = np.zeros(self._Dx)
        else:
            x[:, 0] = x0

        # y0 is nan
        y = np.empty((self._Dy, T + 1))
        y[:, 0] = np.nan

        # Unpack arrays and deal with inputs etc
        A = self.A
        B = self.B if self.B is not None else np.zeros((self._Dx, Du))
        C = self.C
        D = self.D if self.D is not None else np.zeros((self._Dy, Du))

        # Lower cholesky of noises
        LQ = (
            np.linalg.cholesky(self.Q)
            if self.Q is not None
            else np.zeros((self._Dx, self._Dx))
        )
        LR = (
            np.linalg.cholesky(self.R)
            if self.R is not None
            else np.zeros((self._Dy, self._Dy))
        )

        # Actual simulations
        for tt in range(T):
            x[:, tt + 1] = A @ x[:, tt] + B @ u[:, tt] + LQ @ np.random.randn(self._Dx)
            y[:, tt + 1] = (
                C @ x[:, tt + 1] + D @ u[:, tt + 1] + LR @ np.random.randn(self._Dy)
            )

        return x, y
