
import numpy as np
"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""


def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    ....
    """
    if (not isinstance(X, np.ndarray)) or (not isinstance(y, np.ndarray)):
        raise TypeError("X and y must be numpy ndarray")
    
    n = len(y)
    if X.shape[0] != n:
        raise ValueError("the number of rows in X must match length of y")
    
    bootstrap_statistic=np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        
        # empty bootstrap X,y
        X_b = np. zeros_like(X)
        y_b = np.zeros_like(y)

        # fill in these X, y
        np.random.seed(0)
        for j in range(n):
            X_b[j,:] = X[np.random.rand(n),:]
            y[j] = y[np.random.rand(n)]

        # compute bootstrap statistic
        bootstrap_statistic[i] = compute_stat(X_b,y_b)
    
    return bootstrap_statistic
        

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI
    
    ....
    """

    if(not isinstance(bootstrap_stats, np.ndarray) or (not isinstance(alpha, (int,float)))):
        raise TypeError("bootstrap_stats must be numpy ndarray and alpha must be float")

    if ((alpha < 0) or (alpha > 1)):
        raise ValueError("alpha must be between 0 and 1")
        
    lower = np.percentile(bootstrap_stats, 100 * (alpha / 2))
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return lower, upper

def R_squared(X, y):
    """
    Calculate R-squared from multiple linear regression.

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)

    Returns
    -------
    float
        R-squared value (between 0 and 1) from OLS
    
    Raises
    ------
    ValueError
        If X.shape[0] != len(y)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if (not isinstance(X, np.ndarray)) or (not isinstance(y, np.ndarray)):
        raise TypeError("X and y must be numpy ndarray")

    if X.shape[0] != len(y):
        raise ValueError("Number of rows in X must match length of y")

    # OLS estimate: beta = (X'X)^(-1) X'y
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)

    # Predictions
    y_hat = X @ beta

    # Residual sum of squares (RSS) and total sum of squares (TSS)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    return 1 - ss_res / ss_tot
