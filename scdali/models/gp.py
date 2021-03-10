"""Gaussian process implementation based on gpflow."""


import numpy as np

import gpflow
from gpflow.utilities import print_summary, to_default_float, set_trainable

import tensorflow as tf
import tensorflow_probability as tfp

from scdali.models.core import DaliModule
from scdali.utils.stats import freeman_tukey, compute_expected_sample_variance
from scdali.utils.matop import atleast_2d_column


class SparseGP(DaliModule):
    """Sparse GP model for modelling allelic imbalance in single cells.

    A simple wrapper class around the gpflow SGPR model.
    """

    def __init__(
            self, a, d, E,
            kernel='Linear',
            num_inducing=300,
            kernel_params=None,
            variance_prior=False,
            length_scale_prior=True,
            apply_freeman_tukey=False):
        """Creates model.

        Args
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Environment / cell-state matrix.
            kernel: String representing a formula of gpflow.kernels,
                e.g. Linear + RBF.
            num_inducing: Number of inducing points.
            kernel_params: List of dict with arguments for kernel creation.
            variance_prior: Boolean indicating whether to use Gamma prior for
                variance components.
            variance_prior: Boolean indicating whether to use Inverse-Gamma
                prior for kernel lengthscales.
            apply_freeman_tukey: Use the Freeman-Tukey variance stabilizing
                transform to compute rates.
        """
        super().__init__(a, d, E)
        self.apply_freeman_tukey = apply_freeman_tukey

        if self.apply_freeman_tukey:
            self.r = freeman_tukey(self.a, self.d)
        else:
            self.r = self.a / self.d

        if num_inducing > self.n:
            num_inducing = self.n

        self.variance_prior = variance_prior
        self.length_scale_prior = length_scale_prior
        self.num_inducing = num_inducing
        self.kernel = kernel
        self.kernel_split = parse_str_formula(kernel)
        if kernel_params is None:
            kernel_params = [dict() for k in self.kernel_split]
        self.kernel_params = kernel_params

        self.model = self._init_model()


    def _init_model(self):
        """Creates a gpflow SGPR model."""
        Z = self._init_inducing_points()
        mean_function=gpflow.mean_functions.Constant()
        # constrain to be between 0 and 1
        mean_function.c = gpflow.Parameter(
            .5, transform=tfp.bijectors.Sigmoid())
        return gpflow.models.SGPR(
            data=(self.E, self.r),
            kernel=self._create_kernel(),
            inducing_variable=Z,
            mean_function=mean_function,
            num_latent_gps=1)


    def _create_kernel(self):
        """Creates a kernel from list of strings stored in _kernel_split."""
        k = None
        for i, prod_kern in enumerate(self.kernel_split):
            sub_k = None
            for j, kern in enumerate(prod_kern):
                new_k = getattr(
                    gpflow.kernels,
                    kern)( **self.kernel_params[i + j])
                if hasattr(new_k, 'lengthscales') and self.length_scale_prior:
                    new_k.lengthscales.prior = tfp.distributions.InverseGamma(
                        to_default_float(1),
                        to_default_float(1))
                if j == 0:
                    sub_k = new_k
                    if self.variance_prior:
                        new_k.variance.prior = tfp.distributions.Gamma(
                            to_default_float(1),
                            to_default_float(1))
                else:
                    set_trainable(new_k.variance, False)
                    sub_k *= new_k
            if i == 0:
                k = sub_k
            else:
                k += sub_k
        return k


    def _init_inducing_points(self):
        """Samples at random to initialize the inducing point locations."""
        return self.E[np.random.choice(range(self.n), self.num_inducing), :]


    def fit(self, maxiter=250):
        """Fits the model."""
        opt = gpflow.optimizers.Scipy()
        try:
            opt.minimize(
                self.model.training_loss,
                self.model.trainable_variables,
                options={'maxiter': maxiter})
        except tf.errors.InvalidArgumentError:
            print('Warning: Optimization terminated, check model parameters!!')


    def compute_elbo(self):
        """Evalutes the ELBO, a lower bound on the marginal log likelihood."""
        return self.model.elbo().numpy()


    def compute_posterior(self, E=None, full_cov=False):
        """Computes the mean and variances of the posterior over latent rates."""
        E = self.E if E is None else atleast_2d_column(E)
        mu, covar = self.model.predict_f(E, full_cov=full_cov)
        if full_cov:
            covar = covar[0, :, :]
        return mu.numpy(), covar.numpy()


    def compute_explained_variance(self):
        """Normalizes each variance component by the kernel variance."""
        variances = list()
        if isinstance(self.model.kernel, gpflow.kernels.base.ReducingCombination):
            for kernel in self.model.kernel.kernels:
                variances.append(compute_expected_sample_variance(kernel(self.E).numpy()))
        else:
            variances.append(compute_expected_sample_variance(self.model.kernel(self.E).numpy()))

        likelihood_kernel = self.model.likelihood.variance.numpy() * np.eye(self.n)
        variances.append(compute_expected_sample_variance(likelihood_kernel))
        return variances


    def get_prior_mean(self):
        """Returns the estimated prior mean."""
        return self.model.mean_function.c.numpy()


    def print_summary(self):
        """Prints a model summary."""
        print_summary(self.model)


def parse_str_formula(formula):
    """Turns formula of strings with + and * into list of lists with variable names."""
    return [[k.strip() for k in k_prod.split('*')] for k_prod in formula.split('+') ]


