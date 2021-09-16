"""One-vs.-all t-test for differences between cluster means."""


from functools import partial

import numpy as np

from scdali.models.core import DaliModule
from scdali.models.scipy_cluster import ScipyClusterTest
from scdali.models.lm import OLS


class OneVsAllTest(DaliModule):
    """Apply a test separately for each cell-state dimension.

    For example, if E represents clustered data (one-hot encoding or cluster
    labels) and dali_module is a suitable test comparing two populations, this
    model will perform a one-vs-all comparison for each cluster. P-values are
    combined using Bonferroni correction.
    """

    def __init__(self, a, d, E, dali_module, **init_kwargs):
        """Creates model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Environment / cell-state matrix.
            dali_module: A DaliModule model that allows testing for
                cell-state-specific allelic imbalance. Has to implement a
                .test() method returning a p-value.
            init_kwargs: Keyword arguments to be passed to the constructor
                of the given dali_module.
        """
        super().__init__(a=a, d=d, E=E)
        self.pvalues = list()
        self.dali_module = dali_module
        self.init_kwargs = init_kwargs


    def fit(self):
        for i in range(self.k):
            sub_model = self.dali_module(
                a=a, d=d, E=E[:, i, np.newaxis],
                **self.init_kwargs)
            sub_model.fit()
            self.pvalues.append(sub_model.test())


    def test(self, comine_pvals=True):
        """Test for cell-state-specific effects.

        Args:
            combine_pvals: If True, returns the minimum of the
                Bonferroni-corrected p-values. Else returns raw p-values
                for each cell-state dimension.

        Returns:
            P-value if combine_pvals or list of P-values for each cell-state
            dimension.
        """
        if combine_pvals:
            pvals = [p for p in self.pvalues if not np.isnan(p)]
            pvals_bonferroni = multipletests(pvals, method='bonferroni')[1]
            return pvals_bonferroni.min()
        else:
            return self.pvalues


class TTestOVA(OneVsAllTest):
    """One-vs-all t-test for differences between clusters."""

    def __init__(
            self,
            a, d, E,
            apply_freeman_tukey=True):
        super().__init__(
            a=a, d=d, E=E,
            dali_module=partial(ScipyClusterTest, model='ttest_ind'),
            apply_freeman_tukey=apply_freeman_tukey)


class OLSOVA(OneVsAllTest):
    """One-vs-all OLS LRT for cell-state-specific effects."""

    def __init__(
            self,
            a, d, E,
            X=None,
            apply_freeman_tukey=True):
        super().__init__(
            a=a, d=d, E=E,
            X=X, dali_module=OLS,
            apply_freeman_tukey=apply_freeman_tukey)


