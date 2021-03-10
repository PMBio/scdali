"""Run function in parallel."""


import os
import time

import numpy as np

import joblib
from joblib import Parallel, delayed

import contextlib
from tqdm import tqdm


def process_parallel(
        fun_sequential,
        mat_dict,
        n_cores=1,
        verbose=0,
        show_progress=True):
    """Runs function in parallel on matrices using joblib.

    Each matrix in mat_dict has to have the same number of columns. This
    function divides the columns into n_cores blocks to process sequentially.
    This can be faster than using joblib to process each column.

    Args:
        fun_sequential (function): Function that processes submatrices
            sequentially. Should return list of results.
        mat_dict (dict): Dictionary of matrices.
        n_cores (int, optional): Number of cores. Defaults to 1.
        verbose: If nonzero, be verbose.
        show_progress: Show progress bar.

    Returns:
        List of results.
    """

    n_cols = list(mat_dict.values())[0].shape[1]
    for mat in mat_dict.values():
        if mat.shape[1] != n_cols:
            raise ValueError(
                'All values in mat_dict have to '
                'have the same number of columns!')

    n_cores = min(n_cores, n_cols)

    if verbose > 0:
        print('Using %d core(s) ... ' % n_cores, end='')
        start = time.time()

    if n_cores==1:
        # don't invoke joblib
        results = fun_sequential(**mat_dict)
    else:
        blocks = np.linspace(0, n_cols, n_cores + 1).astype(int)
        if show_progress:
            with tqdm_joblib(tqdm(total=n_cores)) as progress_bar:
                results = Parallel(verbose=verbose, n_jobs=n_cores)(delayed(fun_sequential)(
                        **{key: mat[:, blocks[i]:blocks[i+1]] for (key, mat) in mat_dict.items()}
                    ) for i in range(n_cores))
        else:
            results = Parallel(verbose=verbose, n_jobs=n_cores)(delayed(fun_sequential)(
                    **{key: mat[:, blocks[i]:blocks[i+1]] for (key, mat) in mat_dict.items()}
                ) for i in range(n_cores))

        results = [res for sublist in results for res in sublist]
    if verbose > 0:
        print('done. Elapsed time: %.2f seconds' % (time.time() - start))
    return results


# tqdm support for joblib
# https://stackoverflow.com/a/58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

