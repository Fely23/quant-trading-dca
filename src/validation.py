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
        
        # Determine fixed embargo size
        if self.embargo_td is not None:
            embargo_size = self.embargo_td
        else:
            embargo_size = int(n_samples * self.embargo_pct)
            
        for i in range(self.n_splits):
            # In split i, test fold starts after (i + 1) train/test blocks
            # Standard Walk-Forward split:
            # Train: 0 to start_test
            # Test: start_test to end_test
            start_test = (i + 1) * test_size
            end_test = start_test + test_size
            if i == self.n_splits - 1:
                end_test = n_samples
                
            test_indices = indices[start_test:end_test]
            
            # Train indices before test
            train_indices_before = indices[0:start_test]
            # Purge the end of the train set before the test set starts
            # We must remove any training indices in [start_test - horizon + 1, start_test]
            purge_start = max(0, start_test - self.horizon + 1)
            train_indices_before = train_indices_before[train_indices_before < purge_start]
            
            # Train indices after test (in K-fold validation style, if we used future data, we would include them)
            # In TimeSeriesSplit, we only use past data.
            # But let's check if we want to allow future training (like in Purged K-Fold).
            # To be general and follow Marcos Lopez de Prado's Purged K-Fold, we can split the indices into K blocks.
            # In block i, test is block i, and train is all other blocks (before and after), but purged.
            # Let's implement BOTH: standard Walk-Forward Purged, and Purged K-Fold.
            # A Purged K-Fold is much more sample efficient and allows testing multiple paths.
            # Let's write the Purged K-Fold splits!
            # For each fold i, the test set is block i.
            # The train set is all other blocks, but we purge:
            # - [start_test - horizon + 1, end_test + embargo_size]
            
            train_indices = np.setdiff1d(indices, test_indices)
            
            # Apply purging and embargo: remove any training index in [start_test - horizon + 1, end_test + embargo_size]
            purge_embargo_start = max(0, start_test - self.horizon + 1)
            purge_embargo_end = min(n_samples, end_test + embargo_size)
            
            leakage_indices = np.arange(purge_embargo_start, purge_embargo_end)
            train_indices = np.setdiff1d(train_indices, leakage_indices)
            
            yield train_indices, test_indices
            
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
