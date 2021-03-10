"""Tests for beta-binomial estimation."""


import numpy as np
from scipy.stats import betabinom

from scdali.utils.stats import fit_polya
from scdali.utils.stats import fit_polya_precision
from scdali.utils.stats import match_polya_moments


EXAMPLE_DATA_BINOMIAL = np.asarray([
    [0, 2],
    [1, 1],
    [1, 1],
    [2, 0]
])

EXAMPLE_DATA_BINARY = np.asarray([
    [0, 1],
    [1, 0]
])

EXAMPLE_DATA_alpha = np.asarray([1, 1.5])
EXAMPLE_DATA_BETA = np.random.default_rng(123).beta(
    a=EXAMPLE_DATA_alpha[0],
    b=EXAMPLE_DATA_alpha[1],
    size=1000)
EXAMPLE_DATA_BETABINOMIAL = np.random.default_rng(123).binomial(
    n=5,
    p=EXAMPLE_DATA_BETA,
    size=1000)
EXAMPLE_DATA_BETABINOMIAL = np.stack([
    EXAMPLE_DATA_BETABINOMIAL, 
    5 - EXAMPLE_DATA_BETABINOMIAL]).T
EXAMPLE_DATA_NUMERIC_s = 2.4633351335133513
EXAMPLE_DATA_NUMERIC_alpha = np.asarray([
    0.9629679679679679,
    1.5336836836836838
])


def test_fit_polya():
    # binary data, zero has to be global optimum
    alpha, _ = fit_polya(EXAMPLE_DATA_BINARY)
    np.testing.assert_allclose(alpha, np.zeros(2))

    # binomial data, optimum at infinity
    alpha, _ = fit_polya(EXAMPLE_DATA_BINOMIAL)
    np.testing.assert_equal(alpha, np.inf * np.ones(2))


    # beta-binomial data
    alpha, _ = fit_polya(EXAMPLE_DATA_BETABINOMIAL)
    np.testing.assert_allclose(alpha, EXAMPLE_DATA_NUMERIC_alpha, rtol=1e-4)


def test_fit_polya_precision():
    # binary data, zero has to be global optimum
    m = np.asarray([.5, .5])
    s, _ = fit_polya_precision(EXAMPLE_DATA_BINARY, m=m)
    np.testing.assert_allclose(s, 0)

    # binomial data, optimum at infinity
    m = np.asarray([.5, .5])
    s, _ = fit_polya_precision(EXAMPLE_DATA_BINOMIAL, m=m)
    np.testing.assert_equal(s, np.inf * np.ones(2))

    # beta-binomial data
    m = EXAMPLE_DATA_alpha / EXAMPLE_DATA_alpha.sum()
    s, _ = fit_polya_precision(EXAMPLE_DATA_BETABINOMIAL, m=m)
    np.testing.assert_allclose(s, EXAMPLE_DATA_NUMERIC_s, rtol=1e-6)
