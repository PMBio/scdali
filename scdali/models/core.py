""" Abstract base class for Dali modules."""


from abc import ABC, abstractmethod
import numpy as np

from scdali.utils.matop import atleast_2d_column


class DaliModule(ABC):
    """Abstract base class for single-cell allelic imbalance model."""


    def __init__(self, a, d, E=None, X=None):
        """Creates DaliModule.

       Stores a, d, E and X as 2-dimensional numpy arrays, keeping only entries
       with nonzero value in d (i.e. non-zero total counts).

        Args:
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Optional environment / cell-state matrix.
            X: Optional design matrix.
        """
        if a.shape[0] != d.shape[0]:
            msg = ('Dimension mismatch: a and d need'
                'to have the same number of entries!')
            raise ValueError(msg)
        if (E is not None) and (a.shape[0] != E.shape[0]):
            msg = ('Dimension mismatch: First dimension of E'
                'has to equal the number of elements in a and d!')
            raise ValueError(msg)
        if (X is not None) and (a.shape[0] != X.shape[0]):
            msg = ('Dimension mismatch: First dimension of X'
                'has to equal the number of elements in a and d!')
            raise ValueError(msg)
        if d.sum() == 0:
            raise ValueError('All counts are zero!')

        self.d = atleast_2d_column(d).astype(float)
        self.idsnonzero = (self.d > 0).flatten()

        self.d = self.d[self.idsnonzero, :]
        self.a = atleast_2d_column(a)[self.idsnonzero, :].astype(float)

        if E is not None:
            self.E = atleast_2d_column(E)[self.idsnonzero, :].astype(float)
            self.k = self.E.shape[1]
        else:
            self.E = None
            self.k = 0

        if X is not None:
            X = atleast_2d_column(X)[self.idsnonzero, :].astype(float)
            non_constant = ~(X[0, :, np.newaxis] == X.T).T.all(0)
            if non_constant.sum() < X.shape[1]:
                print('Warning: Removing constant columns from X.')
                if non_constant.sum() == 0:
                    X = None
                else:
                    X = atleast_2d_column(X[:, non_constant])
            self.X = X
        else:
            self.X = None

        self.n = self.a.shape[0]


    @abstractmethod
    def fit(self):
        """Fit model. Has to be implemented by all subclasses."""
        return

