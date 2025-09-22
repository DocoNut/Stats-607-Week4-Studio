import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared

def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""

def test_bootstrap_sample_valid():
    """Check bootstrap_sample works on a simple dataset."""
    rng = np.random.default_rng(123)
    n = 100
    X = rng.normal(size=(n, 1))
    y = 3*X[:, 0] + rng.normal(size=n)
    X_design = np.c_[np.ones(n), X]

    def slope_stat(X, y):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta[1]

    boot_stats = bootstrap_sample(X_design, y, slope_stat, n_bootstrap=100, random_state=42)
    assert boot_stats.shape == (100,)
    assert boot_stats.std() > 0


def test_bootstrap_sample_invalid_shapes():
    """Check bootstrap_sample raises ValueError for mismatched X and y."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(50, 2))  # 50 samples
    y = rng.normal(size=60)       # 60 samples, mismatch!

    def dummy_stat(X, y):
        return 0.0

    with pytest.raises(ValueError, match="must match length of y"):
        bootstrap_sample(X, y, dummy_stat, n_bootstrap=10)
    with pytest.raises(TypeError, match="must be numpy ndarray"):
        bootstrap_sample(X, 0, dummy_stat, n_bootstrap=10)
    
def test_bootstrap_ci():
    """Test bootstrap_sample"""
    # Alpha too low (0)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        bootstrap_ci(np.array([1,2,3]), alpha=0)

    # Alpha too high (1)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        bootstrap_ci(np.array([1,2,3]), alpha=1)

    with pytest.raises(TypeError, match = 'bootstrap_stats must be numpy ndarray'):
        bootstrap_ci(1, alpha = 0.05)

def test_R_squared():
    """test R_squared"""
    X=np.ones((3,3))
    y=np.ones(2)
    with pytest.raises(ValueError, match = 'must match length of y'):
        R_squared(X,y)
    with pytest.raises(TypeError, match = 'must be numpy ndarray'):
        R_squared(1,y)

def integrated_test():
    test_bootstrap_ci()
    test_bootstrap_sample_invalid_shapes()
    test_R_squared()
    print("All the functions work!")

if __name__ == '__main__':
    integrated_test()