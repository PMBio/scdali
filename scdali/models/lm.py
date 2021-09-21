"""Fixed-effect test for differences between clusters."""


import numpy as np
import statsmodels.api as sm

from scdali.utils.stats import freeman_tukey
from scdali.models.core import DaliModule


class OLS(DaliModule):
    """Perform linear regression to test for cell-state-specific effects.

    Wrapper around statsmodels.api.OLS. Refer to the statsmodels API
    reference for detailed info on implementation and assumptions.
    """

    def __init__(self, a, d, E, X=None, apply_freeman_tukey=True):
        """Creates model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Environment / cell-state matrix.
            X: Covariate matrix. Please note that X must not be used to add an
                intercept term (column of ones). An intercept is always
                added automatically.
            apply_freeman_tukey: Use the Freeman-Tukey variance stabilizing
                transform to compute rates.
        """
        super().__init__(a=a, d=d, E=E, X=X)
        self.apply_freeman_tukey = apply_freeman_tukey

        if self.apply_freeman_tukey:
            self.r = freeman_tukey(self.a, self.d)
        else:
            self.r = self.a / self.d


    def fit(self):
        X = sm.add_constant(self.X) if self.X is None else np.ones_like(self.r)
        self.fit_null = sm.OLS(self.r, np.hstack([self.E, X])).fit()
        self.fit_alt = sm.OLS(self.r, X).fit()


    def test(self):
        """Test for cell-state-specific effects.

        Returns:
            P-value.
        """
        _, pvalue, _ = self.fit_alt.compare_lr_test(self.fit_null)
        return pvalue


