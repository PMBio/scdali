"""Functions to run scDALI models."""

from functools import partial

import numpy as np

from scdali.models import DaliJoint
from scdali.models import DaliHet
from scdali.models import DaliHom

from scdali.utils.run_model import run_model, create_method_callback
from scdali.utils.parallel import process_parallel
from scdali.utils.matop import atleast_2d_column


MODELS = {
    'scDALI-Joint': DaliJoint,
    'scDALI-Het': DaliHet,
    'scDALI-Hom': DaliHom,
}


def run_tests(
        A, D,
        model,
        X=None,
        cell_state=None,
        return_rho=False,
        base_rate=None,
        n_cores=1):
    """Run scDALI hypothesis tests for each region.

    A, D are assumed to be n-by-d, where n is the number of cells and d the
    number of regions to model.

    Args:
        A: Alternative counts for each cell and region.
        D: Total counts for each cell and region.
        model: String indicating the model to run. Options are
            'scDALI-Joint' - a Beta-Binomial variance component score test to test
                for either heterogeneous or homogeneous allelic imbalance.
                Requires cell_state and base_rate.
            'scDALI-Het' - a Beta-Binomial variance component score test to test
                for heterogeneous allelic imbalance.
                Requires cell_state.
            'scDALI-Hom' - a Beta-Binomial variance component score test to test
                for homogeneous allelic imbalance.
                Requires base_rate.
        X: Optional design matrix.
        cell_state: Numerical vector or matrix of cell states, e.g. clusters or
            coordinates in a low-dimensional cell-state space.
        return_rho: When model is scDALI-Joint, this flag indicates whether to
            return rho, the fraction of allelic variation explained by global
            imbalance.  cell_state: Matrix of cell states, e.g. clusters or coordinates
            in a low-dimensional cell-state space.
        base_rate: Null allelic rate.
        n_cores: Number of cores to use.

    Returns:
        p-values for each region.
    """
    D = atleast_2d_column(D)
    A = atleast_2d_column(A)

    if X is not None:
        X = atleast_2d_column(X)

    if A.shape != D.shape:
        raise ValueError('A and D need to be of the same shape.')

    try:
        m = MODELS[model]
    except KeyError:
        msg = ('Model not recognized. Choices are '
            + ', '.join(MODELS.keys()) + '.')
        raise ValueError(msg)

    if model in ['scDALI-Joint', 'scDALI-Het'] and cell_state is None:
        raise ValueError('%s requires cell_state to be specified' % model)
    if model in ['scDALI-Joint', 'scDALI-Hom'] and base_rate is None:
        raise ValueError('%s requires base_rate to be specified' % model)

    init_kwargs = {}
    fit_kwargs = {}
    if model in ['scDALI-Joint', 'scDALI-Hom']:
        init_kwargs['base_rate'] = base_rate
    if model in ['scDALI-Joint', 'scDALI-Het']:
        init_kwargs['E'] = cell_state
    init_kwargs['X'] = X


    n_cores = min(n_cores, D.shape[1])
    print('[scdali] Processing %d regions on %d core(s) ... ' % (D.shape[1], n_cores), flush=True)

    callbacks = []
    if model == 'scDALI-Joint':
        callbacks.append(create_method_callback('test', return_rho=return_rho))
    else:
        callbacks.append(create_method_callback('test'))

    show_progress = False if n_cores > 1 else True
    f = partial(
            run_model,
            m,
            init_kwargs=init_kwargs,
            fit_kwargs=fit_kwargs,
            callbacks=callbacks,
            show_progress=show_progress)
    results = process_parallel(
            f,
            mat_dict={'A':A, 'D':D},
            n_cores=n_cores)

    out = dict()
    if model == 'scDALI-Joint' and return_rho:
        out['pvalues'] = np.asarray([r[0][0] for r in results]).T
        out['rhos'] = np.asarray([r[0][1] for r in results]).T
    else:
        out['pvalues'] = np.asarray([r[0] for r in results]).T
    return out

