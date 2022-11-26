from typing import Any, Generic, Optional, Tuple, Union, Callable, TypeVar, List
import numpy as np

T = TypeVar("T")

"""
Utilities
"""


def lq(A: np.ndarray) -> np.ndarray:
    """LQ Decomposition

    Note:
        Function returns lower triangular L matrix only not Q

    Args:
        A (np.ndarray): Matrix to be decomposed

    Returns:
        np.ndarray: L matrix of A = LQ
    """

    return np.linalg.qr(A.T, mode="r").T


def generate_multisine(freqs: np.ndarray, t: np.ndarray):
    # Generate a random phase multisine of frequencies in freqs

    X = np.sin(freqs[0] * t + np.random.rand() * 2 * np.pi)
    for f in freqs[1:]:
        X += np.sin(f * t + np.random.rand() * 2 * np.pi)

    return X


"""
Decorators
"""


def num_to_np(func: Callable):
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


def is_square(func: Callable):
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
    def wrapper(func: Callable):
        def wrapped_func(*args: Tuple[T, Union[int, np.ndarray, None]], **kwargs: Any):

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


def is_positive_definite(func: Callable):
    def wrapper(*args: Any, **kwargs: Any):
        if args[1] is None or np.all(np.linalg.eigvals(args[1]) > 0):
            return func(*args, **kwargs)
        else:
            raise ValueError

    return wrapper


def valid_system_matrix(func: Callable):
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


"""
Types
"""

ModalType = Tuple[List[float], List[float], np.ndarray]
