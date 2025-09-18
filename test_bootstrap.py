import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, r_squared

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    
def test_bootstrap_ci():
    """Test bootstrap_sample"""
    # Alpha too low (0)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        bootstrap_ci([1,2,3], alpha=0)

    # Alpha too high (1)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        bootstrap_ci([1,2,3], alpha=1)

    with pytest(TypeError, match = 'arrays or lists'):
        bootstrap_ci(1, alpha = 0.05)

def test_R_squared():
    """test R_squared"""
    X=np.ones((2,3))
    y=np.ones(2)
    with pytest(ValueError, match = 'X.shape[0]'):
        r_squared(X,y)
    with pytest(TypeError, match = 'arrays or lists'):
        r_squared(1,'abc')