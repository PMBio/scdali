""" Abstract base class for Dali modules."""


from abc import ABC, abstractmethod
import numpy as np

from scdali.utils.matop import atleast_2d_column


class DaliModule(ABC):
    """Abstract base class for single-cell allelic imbalance model."""


    def __init__(self, a, d, E):
        """Creates DaliModule.

       Stores a, d and E as 2-dimensional numpy arrays, keeping only entries
       with nonzero value in d (i.e. non-zero total counts).

        Args:
            a: Counts for the alternative allele in each cell.
            d: Total counts for both alleles in each cell.
            E: Environment / cell-state matrix.
        """
        if a.shape[0] != d.shape[0]:
            msg = ('Dimension mismatch: a and d need'
                'to have the same number of entries!')
            raise ValueError(msg)
        if a.shape[0] != E.shape[0]:
            msg = ('Dimension mismatch: First dimension of E'
                'has to equal the number of elements in a and d!')
            raise ValueError(msg)
        if d.sum() == 0:
            raise ValueError('All counts are zero!')

        self.d = atleast_2d_column(d).astype(float)
        idsnonzero = (self.d > 0).flatten()

        self.d = self.d[idsnonzero, :]
        self.a = atleast_2d_column(a)[idsnonzero, :].astype(float)
        self.E = atleast_2d_column(E)[idsnonzero, :].astype(float)
        self.k = self.E.shape[1]
        self.n = self.a.shape[0]


    @abstractmethod
    def fit(self):
        """Fit model. Has to be implemented by all subclasses."""
        return

