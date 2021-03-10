"""One-vs.-all t-test for differences between cluster means."""


import numpy as np
from scipy.stats import chi2, ttest_ind

from statsmodels.stats.multitest import multipletests

from scdali.utils.matop import aggregate_rows, preprocess_clusters
from scdali.utils.stats import freeman_tukey
from scdali.models.core import DaliModule


class ClusterTTest(DaliModule):
    """One-vs-all t-test for heterogeneous allelic imbalance.

    Implements a one-vs-all t-test to detect differences in mean allelic rates
    between clusters.
    """

    def __init__(self, a, d, E, apply_freeman_tukey=True):
        """Creates model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Cluster labels for each cell.
            apply_freeman_tukey: Use the Freeman-Tukey variance stabilizing
                transform to compute rates.
        """
        cluster_ids, self.cluster_order = preprocess_clusters(E)
        super().__init__(a, d, cluster_ids)
        self.apply_freeman_tukey = apply_freeman_tukey

        if self.apply_freeman_tukey:
            self.r = freeman_tukey(self.a, self.d)
        else:
            self.r = self.a / self.d


    def fit(self):
        pass


    def _test_cluster(self, c):
        """Performs test for one cluster."""
        ids = (self.E == c).flatten()
        if ids.sum() == 0:
            return np.nan
        d = aggregate_rows(self.d > 0, ids, fun='sum')
        if (d < 2).any():
            return np.nan
        else:
            # t-test
            _, pval = ttest_ind(self.r[ids], self.r[~ids], equal_var = True)
            pval = pval.item()
        return pval


    def test(self, aggregate_pvals=True):
        """Tests mean of each cluster vs the remaining cells.

        Args:
            aggregate_pvals: If True, returns the minimum of the
            Bonferroni-corrected p-values. Else returns raw p-values
            for each cluster.

        Returns:
            P-value if aggregate_pvals or dict with pvals and cluster order.
        """
        pvals = list()
        for c in range(len(self.cluster_order)):
            pval = self._test_cluster(c)
            pvals.append(pval)
        if aggregate_pvals:
            pvals = [p for p in pvals if not np.isnan(p)]
            pvals_bonferroni = multipletests(pvals, method='bonferroni')[1]
            return pvals_bonferroni.min()
        else:
            return {'pvals': pvals, 'cluster_order': self.cluster_order}

