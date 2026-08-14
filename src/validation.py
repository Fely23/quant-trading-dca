import numpy as np
import pandas as pd
from typing import Generator, Tuple, List

class PurgedTimeSeriesSplit:
    """
    Purged and Embargoed Time Series Cross-Validator.
    
    Parameters:
        n_splits (int): Number of splits.
        horizon (int): Prediction horizon (H) in periods. Used to purge train indices before test.
        embargo_pct (float): Percentage of data to embargo after test set.
        embargo_td (int): Fixed number of periods to embargo after test set (overrides embargo_pct if set).
    """
    def __init__(self, n_splits: int = 5, horizon: int = 5, embargo_pct: float = 0.01, embargo_td: int = None):
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo_pct = embargo_pct
        self.embargo_td = embargo_td
        
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates indices for train and test splits.
        
        Parameters:
            X (pd.DataFrame): Features dataframe.
            y (pd.Series): Target series.
            
        Yields:
            Tuple[np.ndarray, np.ndarray]: Train and test indices.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate size of each test fold
        test_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Strict walk-forward split: every training observation precedes its
            # test fold. This makes the evaluation representative of information
            # that would have been available at the time of the prediction.
            start_test = (i + 1) * test_size
            end_test = start_test + test_size
            if i == self.n_splits - 1:
                end_test = n_samples
                
            test_indices = indices[start_test:end_test]
            
            # Purge observations whose forward-return label overlaps the first
            # test observation. No post-test rows are used in walk-forward mode,
            # so an embargo after the test set is not applicable here.
            purge_start = max(0, start_test - self.horizon + 1)
            train_indices = indices[:purge_start]
            
            yield train_indices, test_indices
            
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
