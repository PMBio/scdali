"""Matrix helper operations."""

import numpy as np


def atleast_2d_column(ary):
    """
    Same as np.atleast_2d except that it returns
    column instead of row vectors.
    """
    ary = np.asanyarray(ary)
    if ary.ndim == 0:
        result = ary.reshape(1, 1)
    elif ary.ndim == 1:
        result = ary[:, np.newaxis]
    else:
        result = ary
    return result


def aggregate_rows(x, groups, fun='mean'):
    """Aggregates rows in matrix by groups.

    Args:
        x (ndarray): Array to aggregate.
        groups (1d ndarray): Groups, represented by integers between
        0 and n_groups.
    """
    from scipy.sparse import isspmatrix

    if isspmatrix(x):
        x = x.tocsr()

    n_groups = int(np.max(groups) + 1)
    x_agg = np.zeros([n_groups, x.shape[1]])

    for group in range(n_groups):
        if fun == 'mean':
            x_agg[group, :] = x[groups == group, :].mean(0)
        elif fun == 'std':
            x_agg[group, :] = x[groups == group, :].std(0)
        elif fun == 'sum':
            x_agg[group, :] = x[groups == group, :].sum(0)
    return x_agg


def preprocess_clusters(x):
    """Computes array of cluster indices with contiguous labels.

    If x is a two-dimensional array it is assumed to contain one-hot-encoded
    cluster identities. If x is one-dimensional, this function identifies
    unique entries with contiguous integers.

    Args:
        x: Cluster assignments.

    Returns:
        A tuple of integer labels and a list of the original cluster labels in the new order.
    """
    x = np.asarray(x).squeeze()
    dims = len(x.shape)
    if dims > 2:
        raise ValueError('x has to be one- or two-dimensinal.')
    if dims == 2:
        if ((x != 0).sum(1) != 1).any():
            msg = ('Input is 2-dimensinal but does not'
                'contain valid one-hot encoding.')
            raise ValueError(msg)
        else:
            x = np.where(x)[1]
    cluster_order = np.unique(x).tolist()
    cluster_dict = {j: i for i, j in enumerate(cluster_order)}
    cluster_ids = np.asarray([cluster_dict[e] for e in x])
    return cluster_ids, cluster_order


def normalize_cov_matrix(E, norm_type="linear_covariance"):
    """Normalises covariance matrix.

    Args:
        E: Cell-state matrix.
        norm_type (str): Type of normalization. Options are:
            'linear_covariance' - the environment matrix is normalized in such
                a way that the outer product EE^T has mean of diagonal of ones.
            'weighted_covariance' - the environment matrix is normalized in
                such a way that the outer product EE^T has diagonal of ones.
            'correlation', the environment is normalized in such a way that the
                outer product EE^T is a correlation matrix (with a diagonal of
                ones).

    Returns
        E ((n, k) ndarray): Normalized cell-state matrix.
    """
    std = E.std(0)
    E = E[:, std > 0]
    E -= E.mean(0)
    E /= E.std(0)
    if norm_type == "linear_covariance":
        E *= np.sqrt(E.shape[0] / np.sum(E ** 2))
    elif norm_type == "weighted_covariance":
        E /= ((E ** 2).sum(1) ** 0.5)[:, np.newaxis]
    elif norm_type == "correlation":
        E -= E.mean(1)[:, np.newaxis]
        E /= ((E ** 2).sum(1) ** 0.5)[:, np.newaxis]
    return E

