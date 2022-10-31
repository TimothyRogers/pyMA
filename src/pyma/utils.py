from types import NoneType
from typing import Any, Optional, Tuple, Union
import numpy as np

from pyma.systems import DynamicSystem

"""
Utilities
"""


def block_hankel(X: np.ndarray, order: int, N: Optional[int] = None):
    # Block hankel matrix order o

    if N is None:
        D, N = X.shape
        N -= order
    else:
        D, _ = X.shape
    H = np.empty((D * (order + 1), N))

    for o in range(order + 1):
        H[o * D : (o + 1) * D, :] = X[:, o : N + o]

    return H


def generate_multisine(freqs: np.ndarray, t: np.ndarray):
    # Generate a random phase multisine of frequencies in freqs

    X = np.sin(freqs[0] * t + np.random.rand() * 2 * np.pi)
    for f in freqs[1:]:
        X += np.sin(f * t + np.random.rand() * 2 * np.pi)

    return X


"""
Decorators
"""


def num_to_np(func: function):
    # Decorator that makes all number inputs 2D arrays
    def wrapper(*args: Any, **kwargs: Any):
        args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, int) or isinstance(a, float):
                args[i] = np.array([a])[:, None]
        args = tuple(args)
        for key, val in kwargs.items():
            if isinstance(val, int) or isinstance(val, float):
                kwargs[key] = np.array([val])[:, None]
        return func(*args, **kwargs)

    return wrapper


def is_square(func: function):
    def wrapper(*args: Optional[np.ndarray], **kwargs: Any):
        if args[1] is None or (
            len(args[1].shape) == 2 and args[1].shape[0] == args[1].shape[1]
        ):
            return func(*args, **kwargs)
        else:
            raise ValueError

    return wrapper


def is_symmetric(func):
    def wrapper(*args: Optional[np.ndarray], **kwargs: Any):
        if args[1] is None or np.all(args[1] == args[1].T):
            return func(*args, **kwargs)
        else:
            raise ValueError

    return wrapper


def verify_dims(*dargs: Union[int, str]):
    # Makes sure matrices get set at the right dimensions
    def wrapper(func: function):
        def wrapped_func(
            *args: Tuple[DynamicSystem, Union[int, np.ndarray, None]], **kwargs: Any
        ):

            # Dynamically get dims if strings
            dims = [
                getattr(args[0], "_" + d) if isinstance(d, str) else d for d in dargs
            ]

            if isinstance(args[1], np.ndarray):
                arg_dims = args[1].shape
            elif isinstance(args[1], int) or isinstance(args[1], float):
                arg_dims = [1]
            elif args[1] is None:
                arg_dims = None
            else:
                raise ValueError

            # Update zero dims
            for i, d in enumerate(dims):
                if d == 0 and arg_dims is not None:
                    dims[i] = arg_dims[i]
                    if isinstance(dargs[i], str):
                        setattr(args[0], "_" + dargs[i], arg_dims[i])

            # Verifying the dims
            if args[1] is None:
                return func(*args, **kwargs)
            elif isinstance(args[1], np.ndarray) and (
                len(args[1].shape) == len(dims)
                and all([d1 == d2 for d1, d2 in zip(args[1].shape, dims)])
            ):
                return func(*args, **kwargs)
            elif (
                (isinstance(args[1], int) or isinstance(args[1], float))
                and len(dims) == 1
                and dims[0] == 1
            ):
                return func(*args, **kwargs)
            else:
                raise ValueError

        return wrapped_func

    return wrapper


def is_positive_definite(func: function):
    def wrapper(*args: Any, **kwargs: Any):
        if args[1] is None or np.all(np.linalg.eigvals(args[1]) > 0):
            return func(*args, **kwargs)
        else:
            raise ValueError

    return wrapper


def valid_system_matrix(func: function):
    @is_square
    @is_symmetric
    @is_positive_definite
    def wrapper(*args: Any, **kwargs: Any):
        return func(*args, **kwargs)

    return wrapper


#%%

"""
Errors
"""


class SimulationError(Exception):
    def __init__(self, msg: str):
        self.message = msg
