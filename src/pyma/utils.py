'''
Utilities
'''

import numpy as np

def is_square(func):
    def wrapper(*args,**kwargs):
        if args[1] is None or (len(args[1].shape) == 2 and args[1].shape[0] == args[1].shape[1]):
            return func(*args, **kwargs)
        else:
            raise ValueError
    return wrapper

def is_symmetric(func):
    def wrapper(*args,**kwargs):
        if args[1] is None or np.all(args[1] == args[1].T):
            return func(*args, **kwargs)
        else:
            raise ValueError
    return wrapper

def is_positive_definite(func):
    def wrapper(*args,**kwargs):
        if args[1] is None or np.all(np.linalg.eigvals(args[1]) > 0):
            return func(*args, **kwargs)
        else:
            raise ValueError
    return wrapper

def valid_system_matrix(func):
    @is_square
    @is_symmetric
    @is_positive_definite
    def wrapper(*args,**kwargs):
       return func(*args, **kwargs)
    return wrapper


