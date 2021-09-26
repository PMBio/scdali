"""Beta-Binomial generalized linear model."""


import numpy as np
from scipy.stats import chi2, betabinom, binom

from scdali.models.core import DaliModule
from scdali.utils.stats import logistic
from scdali.utils.stats import fit_bb_glm
from scdali.utils.stats import compute_bb_nll


class BetaBinomialGLM(DaliModule):
    """Test for allelic imbalance in single cells."""


    def __init__(self,
        a,
        d,
        E,
        X=None,
        binomial=False):
        """Creates the model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Environment / cell-state matrix.
            X: Covariate matrix. Please note that X must not be used to add an
                intercept term (column of ones). An intercept is always
                added automatically.
            binomial (bool): Use a Binomial likelihood (True) instead of a Beta-
                Binomial model (False).
        """
        super().__init__(a=a, d=d, E=E, X=X)

        self.binomial = binomial

        # add intercept
        ones = np.ones_like(self.a)
        if self.X is not None:
            self.X = np.hstack([self.X, ones])
        else:
            self.X = ones

        self.beta0 = None
        self.theta0 = None
        self.niter0 = 0

        self.beta1 = None
        self.theta1 = None
        self.niter1 = 0


    def fit(self, maxiter=100, tol=1e-5):
        """Fits the null and alternative models."""

        # fit Beta-Binomial GLM
        theta = 0 if self.binomial else None

        X = self.X
        self.beta0, self.theta0, self.niter0 = fit_bb_glm(
            a=self.a, d=self.d, X=X, theta=theta,
            maxiter=maxiter, tol=tol)
        mu0 = logistic(X @ self.beta0)
        self.nll0 = compute_bb_nll(
            a=self.a, d=self.d, mu=mu0, theta=self.theta0)

        X = np.hstack([self.E, self.X])
        self.beta1, self.theta1, self.niter1 = fit_bb_glm(
            a=self.a, d=self.d, X=X, theta=theta,
            maxiter=maxiter, tol=tol)
        mu1 = logistic(X @ self.beta1)
        self.nll1 = compute_bb_nll(
            a=self.a, d=self.d, mu=mu1, theta=self.theta1)


    def test(self):
        """LRT for cell-state-specific effects.

        Returns:
            P-value.
        """
        pval = chi2.sf(2*(self.nll0 - self.nll1), df=self.k)
        return pval


