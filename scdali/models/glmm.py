"""Dali model."""


from scdali.utils.matop import atleast_2d_column
import numpy as np
from scipy.stats import chi2
from scipy.linalg import cho_factor, cho_solve
from chiscore import davies_pvalue, optimal_davies_pvalue, liu_sf

from scdali.models.core import DaliModule
from scdali.utils.stats import logit, logistic 
from scdali.utils.stats import fit_polya_precision, fit_bb_glm


JITTER = 1e-7

class DaliJoint(DaliModule):
    """Test for allelic imbalance in single cells."""


    def __init__(self,
        a,
        d,
        E,
        X=None,
        base_rate=.5,
        test_cell_state_only=False,
        rhos=None,
        binomial=False):
        """Creates the model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Environment / cell-state matrix. For the joint test, (i.e.
                test_cell_state_only=False) it is advised to normalize E by the
                expected sample variance to make the value of rho more
                interpretable.
            X: Covariate matrix. Please note that X must not be used to add an
                intercept term (column of ones). An intercept is automatically
                added if test_cell_state_only=True. Otherwise it is part of the 
                alternative model.
            base_rate: Base/null mean rate (ignored if test_cell_state_only)
            test_cell_state_only: If true, test only for heterogeneous
                imbalance. 
            rhos: Grid for optimizing rho. Ignored if test_cell_state_only.
            binomial (bool): Use a Binomial likelihood (True) instead of a Beta-
                Binomial model (False).
        """
        super().__init__(a=a, d=d, E=E, X=X)
        self.r = self.a / self.d
        if base_rate <= 0 or base_rate >= 1:
            raise ValueError('base_rate has to be between 0 and 1.')

        self.test_cell_state_only = test_cell_state_only
        self.binomial = binomial
        self.base_rate = base_rate

        if test_cell_state_only:
            # consider only heterogeneous kernel
            rhos = [0]

            # add intercept
            ones = np.ones_like(self.r)
            if self.X is not None:
                self.X = np.hstack([self.X, ones])
            else:
                self.X = ones

        if rhos is None:
            self.rhos = np.linspace(start=0, stop=1, num=10)
        else:
            self.rhos = rhos

        self.beta0 = None
        self.theta0 = None
        self.niter = 0


    def fit(self, maxiter=100, tol=1e-5):
        """Fits the null model."""
        # base rate on the logit scale
        offset = 0 if self.test_cell_state_only else logit(self.base_rate)

        if self.X is not None:
            # fit Beta-Binomial GLM
            theta = 0 if self.binomial else None 
            self.beta0, self.theta0, self.niter = fit_bb_glm(
                a=self.a, d=self.d, X=self.X, offset=offset, theta=theta,
                maxiter=maxiter, tol=tol)

            eta0 = self.X @ self.beta0 + offset
            mu0 = logistic(eta0)
        else:
            # joint test without covariates, only fit dispersion parameter
            if self.binomial:
                self.theta0 = 0
            else:
                m = np.asarray([self.base_rate, 1 - self.base_rate])
                data = np.hstack([self.a, self.d-self.a])
                s, self.niter = fit_polya_precision(data, m=m)

                self.theta0 = np.inf if s == 0 else 1/s

            eta0 = offset
            mu0 = self.base_rate

        d = self.d
        theta0 = self.theta0
        if np.isinf(self.theta0):
            # Bernoulli model, substitute limit
            d = (d > 0).astype(float)
            theta0 = 0

        # compute the inverse noise covariance
        # for a Binomial likelihood (theta0=0) W0 simply corresponds to the
        # variance, because the logit is the canonical link function
        self.W0 = d * mu0 * (1 - mu0) * (theta0 + 1)
        self.W0 = self.W0 / (d * theta0 + 1)

        # compute the derivative of the logit link function at the mean
        self.gprime = 1 / ((1 - mu0) * mu0)

        # working vector
        self.y = eta0 + self.gprime * (self.r - mu0) - offset

        if self.X is not None:
            self.Lh = cho_factor(
                (self.W0 * self.X).T @ self.X + JITTER * np.eye(self.X.shape[1]))


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


    def _P(self, v):
        """Project out fixed effects."""
        W0v = self.W0 * v
        if self.X is not None:
            return W0v - (self.W0 * self.X) @ cho_solve(self.Lh, self.X.T @ W0v)
        else:
            return W0v


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
        X=None,
        binomial=False):
        """Creates the model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Environment / cell-state matrix.
            X: Covariate matrix. Please note that X must not be used to add an
                intercept term (column of ones). An intercept is automatically
                added. 
            binomial (bool): Use a Binomial likelihood (True) instead of a Beta-
                Binomial model (False).
        """
        super().__init__(
            a=a, d=d, E=E, X=X,
            test_cell_state_only=True,
            binomial=binomial)


class DaliHom(DaliJoint):
    """DaliJoint wrapper: Test for homogeneous allelic imbalance."""


    def __init__(self,
        a,
        d,
        X=None,
        base_rate=.5,
        binomial=False):
        """Creates the model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            X: Covariate matrix. Please note that X must not be used to add an
                intercept term (column of ones) as it is part of the 
                alternative model.
            base_rate: Base/null mean rate. 
            binomial (bool): Use a Binomial likelihood (True) instead of a Beta-
                Binomial model (False).
        """
        super().__init__(
            a=a, d=d, E=np.ones((a.shape[0], 1)), X=X,
            base_rate=base_rate,
            test_cell_state_only=False,
            rhos=[1.],
            binomial=binomial)

