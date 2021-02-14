'''
Utilities
'''

import numpy as np

def num_to_np(func):
    # Decorator that makes all number inputs 2D arrays
    def wrapper(*args, **kwargs):
        args = list(args)
        for i,a in enumerate(args):
            if isinstance(a, int) or isinstance(a, float):
                args[i] = np.array([a])[:,None]
        args = tuple(args)
        for key, val in kwargs.items():
            if isinstance(val, int) or isinstance(val, float):
                kwargs[key] = np.array([val])[:,None]
        return func(*args, **kwargs)
    return wrapper

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


def verify_dims(*dargs):
    # Makes sure matrices get set at the right dimensions
    def wrapper(func):
        def wrapped_func(*args,**kwargs):
            
            # Dynamically get dims if strings
            dims = [ getattr(args[0],'_'+d) if isinstance(d,str) else d for d in dargs]

            if isinstance(args[1],np.ndarray):
                arg_dims = args[1].shape
            elif (isinstance(args[1],int) or isinstance(args[1],float)):
                arg_dims = [1]
            else:
                ValueError

            # Update zero dims
            for i,d in enumerate(dims):
                if d == 0:
                    dims[i] = arg_dims[i]
                    if isinstance(dargs[i],str):
                        setattr(args[0],'_'+dargs[i],arg_dims[i]) 

            # Verifying the dims
            if args[1] is None:
                return func(*args, **kwargs)
            elif isinstance(args[1],np.ndarray) and \
                (len(args[1].shape) == len(dims) and \
                    all([ d1 == d2 for d1,d2 in zip(args[1].shape,dims)])):
                return func(*args, **kwargs)
            elif (isinstance(args[1], int) or isinstance(args[1], float)) and len(dims) == 1 and dims[0] == 1:
                return func(*args, **kwargs)
            else:
                raise ValueError
            
            

        return wrapped_func
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


