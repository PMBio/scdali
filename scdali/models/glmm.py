"""Dali model."""


import numpy as np
from scipy.stats import chi2
from chiscore import davies_pvalue, optimal_davies_pvalue, liu_sf

from scdali.models.core import DaliModule
from scdali.utils.stats import logit, reparameterize_polya_alpha
from scdali.utils.stats import fit_polya, fit_polya_precision


class DaliJoint(DaliModule):
    """Test for allelic imbalance in single cells."""


    def __init__(self,
        a,
        d,
        E,
        base_rate=.5,
        test_cell_state_only=False,
        rhos=None,
        binomial=False):
        """Creates the model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Environment / cell-state matrix. Note that for the joint test,
                (i.e. test_cel_state_only=False) E should by normalized by the
                expected sample variance to make the value of rho more
                interpretable.
            base_rate: Base/null mean rate (ignored if test_cell_state_only)
            test_cell_state_only: Do not test for global allelic imbalance. In
                this case the base allelic rate is treated as a fixed effect and
                optimized during .fit().
            rhos: Grid for optimizing rho. Ignored if test_cell_state_only.
            binomial (bool): Use a Binomial likelihood (True) instead of a Beta-
                Binomial model (False).
        """
        super().__init__(a, d, E)
        self.r = self.a / self.d

        if base_rate <= 0 or base_rate >= 1:
            raise ValueError('base_rate has to be between 0 and 1.')

        self.test_cell_state_only = test_cell_state_only
        self.binomial = binomial

        if test_cell_state_only:
            # consider only cell-state kernel
            rhos = [0]
        else:
            self.m0 = base_rate

        if rhos is None:
            self.rhos = np.linspace(start=0, stop=1, num=10)
        else:
            self.rhos = rhos

        self.niter = 0


    def fit(self):
        """Fits the null model."""
        s = np.inf
        if not self.binomial:
            # attempt Beta-Binomial parameter estimation
            data = np.hstack([self.a, self.d-self.a])
            # message for the case of failure
            msg = 'Overdispersion estimate out of bounds.'
            msg += ' Reverting to Binomial likelihood.'
            if self.test_cell_state_only:
                # fit both mean and overdispersion
                alpha, self.niter = fit_polya(data)
                if np.isinf(alpha).all() or (alpha == 0).all():
                    self.binomial = True
                    print(msg)
                else:
                    m, s = reparameterize_polya_alpha(alpha)
                    self.m0 = m[0]
            else:
                # only fit overdispersion parameter
                m = np.asarray([self.m0, 1-self.m0])
                s, self.niter = fit_polya_precision(data, m=m)
                if np.isinf(s) or (s == 0):
                    s = np.inf
                    self.binomial = True
                    print(msg)

        if self.binomial and self.test_cell_state_only:
            # no overdispersion estimate desired / possible, estimate mean
            self.m0 = self.a.sum() / self.d.sum()

        # compute overdispersion parameter, note 1/np.inf = 0
        self.theta0 = 1/s

        # compute the inverse noise covariance
        # for a Binomial likelihood (theta0=0) V_inv simply corresponds to the
        # variance, because the logit is the canonical link function
        self.V_inv = self.d * self.m0 * (1 - self.m0) * (self.theta0 + 1)
        self.V_inv = self.V_inv / (self.d * self.theta0 + 1)

        # compute the derivative of the logit link function at the mean
        self.gprime = 1 / ((1-self.m0) * self.m0)

        # working vector
        self.y = self.gprime * (self.r - self.m0)
        if self.test_cell_state_only:
            # m0 was estimated
            self.y += logit(self.m0)


    def test(self, return_rho=False):
        """Tests for allelic imbalance.

        Args:
            return_rho: If True, return optimal rho.

        Returns:
            P-value and optimal rho if return_rho is True.
        """
        # compute score statistic for each rho
        Q_rho = self._compute_score()

        # compute parameters of the score distribution
        Fs, null_lambdas = self._compute_score_dist_parameters()

        # approximate score distribution for each rho
        if len(self.rhos) == 1:
            # approximate null distribution using Davies method
            pvalue = davies_pval(Q_rho[0], Fs[0])
            if return_rho:
                return pvalue, self.rhos[0]
            else:
                return pvalue

        # approximate their distributions using Liu's method:
        approx_out = self._approximate_score_dist(Q_rho, null_lambdas)

        if approx_out[:, 0].min() < 4e-14:
            # beyond Liu method's precision, use Davies + Bonferroni
            pvalues = [davies_pval(Q_rho[i], Fs[i]) for i in range(len(self.rhos))]
            pvalues = np.asarray(pvalues)
            min_idx = pvalues.argmin()
            pvalue = pvalues[min_idx] * len(self.rhos)
            if return_rho:
                return pvalue, self.rhos[min_idx]
            else:
                return pvalue


        # the smallest p-value will be the combined test statistic for all rhos
        T = approx_out[:, 0].min()


        optimal_rho = self.rhos[approx_out[:, 0].argmin()]

        # compute elements of the null distribution for T
        qmin = self._compute_qmin(approx_out)
        null_params = self._compute_null_parameters()

        # compute final p-value
        # return 2 * qmin, *null_params, self.rhos, T
        pvalue = optimal_davies_pvalue(2 * qmin, *null_params, self.rhos, T)

        # resort to Bonferroni in case of numerical issues
        # TODO find more robust estimation
        if pvalue <= 0:
            pvalue = T * len(self.rhos)

        if return_rho:
            return pvalue, optimal_rho
        return pvalue


    def _compute_null_parameters(self):
        """Computes quantities required for the null distribution."""
        Pg = self._P(np.ones_like(self.y))
        m = Pg.sum()
        PE = self._P(self.E)
        gPE = PE.sum(0, keepdims=True)
        EPE = self.E.T @ PE

        tau_top = gPE @ gPE.T
        tau_rho = np.empty(len(self.rhos))
        for i in range(len(self.rhos)):
            tau_rho[i] = self.rhos[i] * m + (1 - self.rhos[i]) / m * tau_top

        phi_F2 = gPE.T @ gPE / m
        phi_F = EPE - phi_F2

        lambda_phi = np.linalg.eigvalsh(phi_F)
        variance_eta = 4 * np.trace(phi_F @ phi_F2)

        # compute mean, variance and kurtosis, see e.g.:
        muQ = np.sum(lambda_phi)
        # the variance of a chi-square distribution with 1 dof is 2
        c2 = np.sum(lambda_phi ** 2)
        varQ = c2 * 2 + variance_eta
        # the kurtosis of a chi-square distribution with 1 dof is 12
        kurQ = 12 * np.sum(lambda_phi ** 4) / (c2 ** 2)
        # degrees of freedom
        dof = 12 / kurQ
        return muQ, varQ, kurQ, lambda_phi, variance_eta, dof, tau_rho


    def _compute_score(self):
        """Computes score statistic for each rho.

        Let 𝙺 be the optimal covariance matrix under the null hypothesis.
        For a given ρ, the score-based test statistic is given by

            𝑄ᵨ = ½𝐲ᵀ𝙿ᵨ(∂𝙺ᵨ)𝙿ᵨ𝐲,

        where

            ∂𝙺ᵨ = ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ.
        """
        Q = np.zeros(len(self.rhos))
        Py = self._P(self.y)
        l = Py.sum() ** 2
        yPE = Py.T @ self.E
        r = yPE @ yPE.T
        for i, rho in enumerate(self.rhos):
            Q[i] = (rho * l + (1 - rho) * r) / 2
        return Q


    def _compute_score_dist_parameters(self):
        """Computes parameters for the distribution of the score statistics.

       The score-based test statistic follows a weighted sum of random variables:

            𝑄 ∼ ∑ᵢ𝜆ᵢχ²(1),

        where 𝜆ᵢ are the non-zero eigenvalues of ½√𝙿(∂𝙺)√𝙿 and

            ∂𝙺ᵨ = ρ𝟏𝟏ᵀ + (1-ρ)𝙴𝙴ᵀ.

        By using SVD decomposition, one can show that the non-zero eigenvalues
        of 𝚇𝚇ᵀ are equal to the non-zero eigenvalues of 𝚇ᵀ𝚇. Therefore, 𝜆ᵢ are
        the non-zero eigenvalues of

            ½[√ρ𝟏 (1-ρ)𝙴̃]𝙿[√ρ𝟏 √(1-ρ)𝙴̃]ᵀ.

        Args:
            return_F: Whether to return F (True) or its eigenvalues for each rho.
        """
        Pg = self._P(np.ones_like(self.y))
        gPg = Pg.sum()
        PE = self._P(self.E)
        EPE = self.E.T @ PE
        gPE = PE.sum(0, keepdims=True)

        F = np.empty((self.k + 1, self.k + 1))
        eigenvals = []
        Fs = []
        for i in range(len(self.rhos)):
            rho = self.rhos[i]

            F[0, 0] = rho * gPg
            F[0, 1:] = np.sqrt(rho) * np.sqrt(1 - rho) * gPE
            F[1:, 0] = F[0, 1:]
            F[1:, 1:] = (1 - rho) * EPE

            Fs.append(F / 2)
            eigenvals.append(np.linalg.eigvalsh(F) / 2)
        return Fs, eigenvals


    def _P(self, X):
        """Project out fixed effects."""
        VinvX = self.V_inv * X
        if self.test_cell_state_only:
            PX = self.V_inv @ VinvX.sum(0, keepdims=True)
            PX = VinvX - PX / self.V_inv.sum()
            return PX
        else:
            return VinvX


    def _approximate_score_dist(self, Qs, lambdas):
        """Computes Pr(𝑄 > q) for 𝑄 ∼ ∑ᵢ𝜆ᵢχ²(1).

        Pr(𝑄 > q) is the p-value for the score statistic.

        Args:
            Qs: 𝑄ᵨ statistic.
            lambdas : 𝜆ᵢ from the null distribution for each ρ.
        """
        return np.stack(
            [_mod_liu(Q, lam) for Q, lam in zip(Qs, lambdas)],
            axis=0)


    def _compute_qmin(self, approx_out):
        """Computes the (1-T)-percentile of the null distribution for each rho.

        T is the minimum p-value across all rhos. Uses Liu approximation.
        """
        T = approx_out[:, 0].min()
        qmin = np.zeros(len(self.rhos))
        percentile = 1 - T
        for i in range(len(self.rhos)):
            mu_q, sigma_q, dof = approx_out[i, 1:]
            q = chi2.ppf(percentile, dof)
            qmin[i] = (q - dof) / (2 * dof) ** 0.5 * sigma_q + mu_q
        return qmin


def _mod_liu(values, weights):
    """Approximates survival function for mixtures of chi-square variables.

    Wrapper around liu_sf which assumes each variable is centered with one
    degree of freedom.

    Args:
        values: Points at which the surival function will be evaluated.
        weights: Mixture weights.

    Returns:
        Approximated survival function applied to values as well as
        matched mean and variance.
    """
    (pv, dof_x, _, info) = liu_sf(
        t=values, lambs=weights,
        dofs=[1] * len(weights),
        deltas=[0] * len(weights),
        kurtosis=True)
    return (pv, info["mu_q"], info["sigma_q"], dof_x)


def davies_pval(Q, F):
    """Wrapper around davies_pvalue that catches AssertionError."""
    try:
        pval = davies_pvalue(Q, F)
    except AssertionError:
        print('Warning - Davies pvalue assertion error: zero p-value')
        pval = 0
    return pval


class DaliHet(DaliJoint):
    """DaliJoint wrapper: Test for heterogeneous allelic imbalance."""


    def __init__(self,
        a,
        d,
        E,
        binomial=False):
        """Creates the model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Environment / cell-state matrix.
            binomial (bool): Use a Binomial likelihood (True) instead of a Beta-
                Binomial model (False).
        """
        super().__init__(
            a=a, d=d, E=E,
            test_cell_state_only=True,
            binomial=binomial)


class DaliHom(DaliJoint):
    """DaliJoint wrapper: Test for homogeneous allelic imbalance."""


    def __init__(self,
        a,
        d,
        base_rate=.5,
        binomial=False):
        """Creates the model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            base_rate: Base/null mean rate (ignored if test_cell_state_only)
            binomial (bool): Use a Binomial likelihood (True) instead of a Beta-
                Binomial model (False).
        """
        super().__init__(
            a=a, d=d, E=np.ones((a.shape[0], 1)),
            base_rate=base_rate,
            test_cell_state_only=False,
            rhos=[1.],
            binomial=binomial)

