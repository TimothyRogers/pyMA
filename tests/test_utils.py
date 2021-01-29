'''
Test Utility Functions
'''

import pytest

from pyma import utils
import numpy as np



def test_is_square():

    A = np.random.rand(2, 2)
    B = np.random.rand(2, 3)
    @utils.is_square
    def dummy(self,x):
        return x
    
    assert((dummy(None,A)==A).all)
    with pytest.raises(ValueError):
        dummy(None,B)

def test_is_symmetric():

    A = np.array([[2,1],[1,3]])
    B = np.array([[2,1],[2,3]])
    @utils.is_symmetric
    def dummy(self,x):
        return x
    
    assert((dummy(None,A)==A).all)
    with pytest.raises(ValueError):
        dummy(None,B)

def test_is_pos_def():

    A = np.array([[2,1],[1,3]])
    B = np.array([[-1,1],[1,3]])
    @utils.is_positive_definite
    def dummy(self,x):
        return x
    
    assert((dummy(None,A)==A).all)
    with pytest.raises(ValueError):
        dummy(None,B)

def test_num_to_np():
    
    @utils.num_to_np
    def dummy(self,a,b=None):
        return a, b

    a, b = dummy(None,1,b=2)

    assert(len(a.shape) == 2 and a.shape[0] == 1 and a.shape[1] == 1)
    assert(len(b.shape) == 2 and b.shape[0] == 1 and b.shape[1] == 1)

