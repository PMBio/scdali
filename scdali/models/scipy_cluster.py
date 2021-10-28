"""ANOVA test for differences between clusters."""


import numpy as np
import scipy.stats

from scdali.utils.matop import preprocess_clusters, aggregate_rows
from scdali.utils.stats import freeman_tukey
from scdali.models.core import DaliModule


MODELS = ['ttest_ind', 'f_oneway', 'kruskal']
# MODELS = ['ttest_ind', 'f_oneway', 'kruskal', 'alexandergovern']

MIN_COUNTS_PER_CLUSTER = 2


class ScipyClusterTest(DaliModule):
    """Wrapper for selected tests from scipy.stats to compare multiple clusters.

    Accepted tests are specified in MODELS. Refer to the scipy.stats API
    reference for detailed info on test implementations and assumptions.
    """

    def __init__(
            self,
            a, d, E,
            apply_freeman_tukey=True,
            model='f_oneway',
            **model_kwargs):
        """Creates model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Cluster labels for each cell.
            apply_freeman_tukey: Use the Freeman-Tukey variance stabilizing
                transform to compute rates.
            model: Test to perform. Options are specified in MODELS. Defaults
                to f_oneway, the basic one-way ANOVA test.
            model_kwargs: Additional keyword arguments to be passed to the
                scipy.stats model.
        """
        cluster_ids, self.cluster_order = preprocess_clusters(E)
        super().__init__(a, d, cluster_ids)
        self.apply_freeman_tukey = apply_freeman_tukey

        if self.apply_freeman_tukey:
            self.r = freeman_tukey(self.a, self.d)
        else:
            self.r = self.a / self.d

        self.n_clusters = int(self.E.max() + 1)
        if self.n_clusters < 2:
            raise ValueError('Tests require at least two clusters.')
        if self.n_clusters > 2 and model == 'ttest_ind':
            raise ValueError('ttest_ind requires exactly two clusters')

        if model not in MODELS:
            raise ValueError('Unrecognized model %s. '
                'Choices are %s.' % (model, ', '.join(MODELS)))
        self.model = getattr(scipy.stats, model)


    def fit(self):
        pass


    def test(self):
        """Tests for differences in population statistics between clusters.

        Returns:
            P-value.
        """
        E = self.E.flatten()
        d = self.d
        r = self.r

        d_cluster = aggregate_rows(d > 0, E, fun='sum')
        to_keep = np.where(d_cluster >= MIN_COUNTS_PER_CLUSTER)[0]
        if to_keep.size < 2: # need at least 2 clusters to compare
            print('Warning: Insufficient counts per cluster.')
            return np.nan

        if self.n_clusters - to_keep.size > 0:
            # some clusters were dropped, reindex clusters
            ids = np.in1d(E, to_keep)
            E, _ = preprocess_clusters(E[ids])
            d = d[ids, :]
            r = r[ids, :]

        n_clusters = int(E.max() + 1)
        samples = [r[E == i, :] for i in range(n_clusters)]
        _, pvalue = self.model(*samples)
        return pvalue.item()


