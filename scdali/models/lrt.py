"""Likelihood ratio test for the mean of a Beta-Binomial distribution."""


import numpy as np
from scipy.stats import chi2, betabinom, binom

from scdali.models.core import DaliModule
from scdali.utils.stats import reparameterize_polya_ms
from scdali.utils.stats import fit_polya, fit_polya_precision


class BetaBinomLRT(DaliModule):
    """Beta-Binomial LRT test for the mean."""


    def __init__(self, a, d, base_rate=0.5, binomial=False):
        """Creates the model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            base_rate (float): Mean for null model.
            binomial (bool): Use a Binomial likelihood (True) instead of a Beta-
                Binomial model (False).
        """
        E = np.ones_like(a)
        super().__init__(a, d, E)
        self.r = self.a / self.d
        self.m0 = base_rate
        self.binomial = binomial


    def fit(self):
        """Fits the null and alternative models."""

        if not self.binomial:
            # attempt Beta-Binomial parameter estimation
            data = np.hstack([self.a, self.d-self.a])
            self.alpha1, self.niter1 = fit_polya(data=data)
            if np.isinf(self.alpha1).all() or (self.alpha1 == 0).all():
                self.binomial = True
                msg = 'Overdispersion estimate out of bounds.'
                msg += ' Reverting to Binomial LRT.'
                print(msg)

        # fit alternative model / estimate negative log-likelihoods
        if not self.binomial:
            # fit overdispersion parameter
            m0 = np.asarray([self.m0, 1-self.m0])
            s0, self.niter0 = fit_polya_precision(data, m=m0)
            self.alpha0 = reparameterize_polya_ms(m0, s0)

            self.nll0 = -betabinom(
                n=self.d,
                a=self.alpha0[0],
                b=self.alpha0[1]).logpmf(self.a).sum()
            self.nll1 = -betabinom(
                n=self.d,
                a=self.alpha1[0],
                b=self.alpha1[1]).logpmf(self.a).sum()
        else:
            # no overdispersion estimate desired / possible, estimate mean
            self.m1 = self.a.sum() / self.d.sum()
            self.nll0 = -binom(
                n=self.d,
                p=self.m0).logpmf(self.a).sum()
            self.nll1 = -binom(
                n=self.d,
                p=self.m1).logpmf(self.a).sum()


    def test(self):
        """Performs a likelihood-ratio test."""
        pval = chi2.sf(2*(self.nll0 - self.nll1), df=1)
        return pval


    def get_estimated_mean(self):
        """Returns mean under the alternative."""
        if self.binomial:
            return self.m1
        else:
            m1 = self.alpha1 / self.alpha1.sum()
            return m1[0]

