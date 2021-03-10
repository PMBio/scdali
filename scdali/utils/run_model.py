"""Run model for all columns in matrix."""


import numpy as np
from tqdm import trange

from scdali.utils.matop import atleast_2d_column


def run_model(
        model,
        A, D,
        init_kwargs={},
        fit_kwargs={},
        callbacks=list(),
        show_progress=True,
        verbose=True):
    """Fit a DaliModule for each column of A and D.

    For each column in D with non-zero counts, this function creates a model
    and calls model.fit() and executes additional callbacks.

    Args
        model: Model to run on each region.
        A: cell-by-region matrix of counts for the alternative allele.
        D: cell-by-region matrix of counts for both alleles (total counts).
        init_kwargs: Additional keyword arguments for the model initialization,
            e.g. cell-state variables.
        fit_kwargs: Additional Keyword arguments for model.fit().
        callbacks: List of callbacks to be executed after fitting the model.
            A callback is a function operating on DaliModule objects. For
            example, to run additional functions or extract fitted parameters.
        show_progress: Show progressbar.
        verbose: Be verbose.

    Returns:
        List of callback return values for each region.
    """
    A = atleast_2d_column(A)
    D = atleast_2d_column(D)
    n_regions = D.shape[1]

    results = list()
    pb = trange(n_regions) if show_progress else range(n_regions)
    for i in pb:
        if D[:, i].sum() == 0:
            if verbose:
                print("Warning: Zero column in D, appending None")
            results.append([None for cb in callbacks])
            continue

        # create model for i-th column
        mod = model(A[:, i], D[:, i], **init_kwargs)

        # fit model
        mod.fit(**fit_kwargs)

        # run callbacks on fitted model
        region_results = list()
        for cb in callbacks:
            region_results.append(cb(mod))
        results.append(region_results)
    return results


# =======================================
# callbacks to query model or run methods
# =======================================

def create_method_callback(method_name, **kwargs):
    """Creates a callback function which executes model method.

    Args:
        method_name: The name (str) of the method to execute.
        kwargs: Optional keyword arguments.

    Returns:
        Callback to be used with run_model. For example, to call the 'test'
        method of a DaliModule:

        cb = create_method_callback('test')
    """
    def callback_method(model):
        return getattr(model, method_name)(**kwargs)
    return callback_method

