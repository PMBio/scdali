"""Stats utility functions."""

from scdali.utils.matop import atleast_2d_column
import numpy as np
from scipy.special import digamma
from scipy.special import polygamma
trigamma = lambda x: polygamma(1, x)


def logit(x):
    """The logit link function."""
    return np.log(x / (1 - x))


def logistic(x):
    """The logistic (inverse logit) function."""
    return 1 / (1 + np.exp(-x))


def apply_fdr_bh(pv, ignore_nan=True):
    """Corrects p-values with Benjamini-Hochberg procedure.

    Args:
        pv: 1-d array or list of pvalues to correct.
        ignore_nan: Boolean indicating if nan values should be ignored.
            Defaults to True.

    Returns:
        BH-corrected p-values.
    """
    from statsmodels.sandbox.stats.multicomp import multipletests
    pv = np.asarray(pv).copy().squeeze()
    if len(pv.shape) > 1:
        raise ValueError('Pleases pass 1-d array or list or p-values')

    pvals_adj = np.full_like(pv, np.nan)
    not_nan =  ~np.isnan(pv) if ignore_nan else range(pvals_adj.size)
    pvals_adj[not_nan] = multipletests(
            pv[not_nan], method="fdr_bh")[1]
    return pvals_adj


def compute_quantile_diff(a, q):
    """Computes difference between q and 1-q quantiles."""
    q1 = np.quantile(a=a, q=q, axis=0)
    q2 = np.quantile(a=a, q=1-q, axis=0)
    return q2 - q1


def freeman_tukey(a, d):
    """Variance stabilizing transform for binomial data."""
    r = np.arcsin(np.sqrt(a/(d+1))) + np.arcsin(np.sqrt((a+1)/(d+1)))
    return r/2


def compute_expected_sample_variance(cov):
    """Computes the expected sample variance for a multivariate normal variable.

    Args:
        cov: Covariance matrix.

    Returns:
        Expected sample variance.
   """
    n = cov.shape[0]
    x = np.trace(cov) - cov.mean(0)[:, np.newaxis].sum()
    return x / (n - 1)


def fit_polya(data, alpha_init=None, tol=1e-6, maxiter=1000):
    """Fits parameters of a Polya distribution

    Uses fixed-point iteration as proposed by Thomas Minka:
    https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf

    Args:
        data: N by K array of counts, where N is the number of observations and
            K is the number of items / categories.
        alpha_init: Initialization for alpha. If None, uses fit_polya_precision
            to find the optimal precision when fixing the mean to the first
            sample moment. This is recommended, as fit_polya_precision can
            detect the lack of overdispersion (alpha infinity).
        tol: Optimization terminates when parameter estimates
            change by less than tol between iterations.
        maxiter: Maximum number of iterations.

    Returns:
        Estimated alpha parameter and number of iterations.
    """
    data = np.asarray(data, float)

    N, K = data.shape
    n = data.sum(1, keepdims=True)

    p = data / n
    if np.array_equal(p, p.astype(bool)):
        return np.zeros(K), 0

    if alpha_init is None:
        # find optimal precision given the empirical mean
        # to initialize alpha
        m, s = reparameterize_polya_alpha(match_polya_moments(data))
        # precision is optimized using Newton-Raphson, which
        # tends to converge very fast
        s, _ = fit_polya_precision(data=data, m=m, s_init=s)
        if np.isinf(s):
            # probably dealing with binomial distribution
            return np.ones_like(m) * np.inf
        alpha = reparameterize_polya_ms(m, s)
    else:
        alpha = np.asarray(alpha_init, float).ravel()
    for niter in range(maxiter):
        # fixed-point iteration
        s = alpha.sum()
        enum = digamma(data + alpha).sum(0) - N * digamma(alpha)
        denom = digamma(n + s).sum() - N * digamma(s)

        alpha_old = alpha
        alpha = alpha_old * enum / denom

        if np.abs(alpha_old - alpha).max() < tol:
            break
    return alpha, niter


def fit_polya_precision(data, m, s_init=None, tol=1e-6, maxiter=1000):
    """Fits precision parameters of a Polya distribution

    Uses Newton-Raphson as proposed by Thomas Minka:
    https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf

    Args:
        data: N by K array of counts, where N is the number of observations and
            K is the number of items / categories.
        m: Distribution mean.
        s_init: Initialization for the precision parameter s.
        tol: Optimization terminates when parameter estimates
            change by less than tol between iterations.
        maxiter: Maximum number of iterations.

    Returns:
        Estimated precision parameter and number of iterations.
    """
    data = np.asarray(data, float)
    m = np.atleast_2d(m)

    N, K = data.shape
    n = data.sum(1, keepdims=True)

    p = data / n
    if np.array_equal(p, p.astype(bool)):
        return 0, 0

    if s_init is None:
        s = match_polya_moments(data).sum()
    else:
        s = float(s_init)
    a_limit = (n/6 * (n-1) * (2*n - 1)).sum()
    a_limit += (data * (data - 1) * (2*data - 1) / (6 * m**2)).sum()
    for niter in range(maxiter):
        sm = m * s
        sum_data_sm = data + sm
        sum_n_s = n + s

        # first derivative
        f1 = N*digamma(s) - digamma(sum_n_s).sum()
        f1 += (m*(digamma(sum_data_sm) - digamma(sm))).sum()

        # second deriative
        m_squared = m**2
        f2 = N*trigamma(s) - trigamma(sum_n_s).sum()
        f2 += (m_squared*(trigamma(sum_data_sm) - trigamma(sm))).sum()

        s_old = s

        if f1 > 0:
            a = -s**2 * f2
            c = f1 - a/s
            s = -a / c
            if c  >= 0:
                # probably dealing with binomial distribution
                return np.inf, niter
            # s = s / (1 + f1 / (s*f2))
        else:
            if np.abs(s*f2 + 2*f1) > 1e-15:
                s = -a_limit / (f1 - a_limit/s)
            else:
                s = s - f1 / (f2 + 3*f1/s)

        if np.abs(s_old - s) < tol:
            break
    return s, niter


def match_polya_moments(data):
    """Matches moments to compute parameters of a Polya distribution

    Used for initializing second-order and fixed-point methods. From:
    https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf

    Args:
        data: N by K array of counts, where N is the number of
            observations and K is the number of items / categories.

    Returns:
        Estimated alpha parameter.
    """
    data = np.asarray(data, float)
    p = data / data.sum(1, keepdims=True)
    p_mean = p.mean(0)
    p0_squared_mean = (p[:, 0]**2).mean()
    s = p_mean[0] - p0_squared_mean
    s = s / (p0_squared_mean - p_mean[0]**2)
    return s * p_mean


def reparameterize_polya_alpha(alpha):
    """Reparameterizes Polya distribution

    Args:
        alpha: Vector of positive values.

    Returns:
        Mean and precision of the distribution
    """
    s = alpha.sum()
    return alpha/s, s


def reparameterize_polya_ms(m, s):
    """Reparameterizes Polya distribution

    Args:
        m: Mean of the distribution.
        s: Precision of the distribution.

    Returns:
        Alpha vector.
    """
    return s * m


def fit_bb_regression(a, d, X, theta=None, maxiter=100, tol=1e-5):
    """Fits Beta-Binomial regression model.
    
    Uses iteratively reweighted least squares / Fisher scoring.

    Args:
        a: Vector successes.
        d: Vector of trials.
        X: Design matrix.
        theta: Dispersion parameter. If None, estimate alternatingly.
        maxiter: Maximum number of iterations
        tol: Break if mean absolute change in estimated parameters is below tol.

    Returns:
        Regression coefficients, estimated dispersion parameter and number of 
        iterations.
    """
    from numpy_sugar.linalg import rsolve

    a = atleast_2d_column(a)
    d = atleast_2d_column(d)
    X = atleast_2d_column(X)

    y = a / d

    fit_precision = theta is None
    if fit_precision:
        data = np.hstack([a, d-a])
    
    beta = rsolve(X.T @ X, X.T @ y)
    for i in range(maxiter):
        print(i)

        eta = X @ beta
        mu = logistic(eta)

        if fit_precision:
            m = np.hstack([mu, 1-mu])
            maxiter = min(10**(i+1), 1000)
            (s, niter) = fit_polya_precision(data=data, m=m, maxiter=maxiter)
            print('Polya precision niter: %d' % niter)
            theta = 1/s

        gprime = 1 / ((1 - mu) * mu)
        z = eta + gprime * (y - mu)

        W = d * mu * (1 - mu) * (theta + 1)
        W = W / (d * theta + 1)

        XW = (W * X).T
        beta_new = rsolve(XW @ X, XW @ z)

        if np.abs(beta - beta_new).mean() < tol:
            break

        beta = beta_new
    return beta, theta, i

