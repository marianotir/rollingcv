import numpy as np
from sklearn.model_selection import BaseCrossValidator

class RollingWindowSplit(BaseCrossValidator):
    """
    Rolling window time series cross-validation.

    Parameters:
        n_splits (int): Number of folds to generate.
        window_size (int or float): Training window size.
            - int = absolute number of samples
            - float = fraction of total samples (0 < float <= 1)
        horizon (int or float): Test window size.
            - int = absolute number of samples
            - float = fraction of total samples (0 < float <= 1)
        gap (int): Number of samples to skip between training and test sets to avoid leakage.
    """

    def __init__(self, n_splits=5, window_size=0.6, horizon=0.1, gap=0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if isinstance(window_size, float) and not (0 < window_size <= 1):
            raise ValueError("If float, window_size must be between 0 and 1.")
        if isinstance(horizon, float) and not (0 < horizon <= 1):
            raise ValueError("If float, horizon must be between 0 and 1.")
        if window_size < 0 or horizon < 0 or gap < 0:
            raise ValueError("window_size, horizon, and gap must be non-negative.")

        self.n_splits = n_splits
        self.window_size = window_size
        self.horizon = horizon
        self.gap = gap

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        window_size = int(self.window_size * n_samples) if isinstance(self.window_size, float) else self.window_size
        horizon = int(self.horizon * n_samples) if isinstance(self.horizon, float) else self.horizon

        if window_size <= 0 or horizon <= 0:
            raise ValueError("window_size and horizon must be greater than 0.")

        total_required = window_size + self.gap + horizon + (self.n_splits - 1)
        if n_samples < total_required:
            raise ValueError("Not enough data to create the requested number of splits.")

        max_start = n_samples - window_size - self.gap - horizon
        step = max_start // (self.n_splits - 1)

        for i in range(self.n_splits):
            start = i * step
            train_idx = np.arange(start, start + window_size)
            test_start = start + window_size + self.gap
            test_idx = np.arange(test_start, test_start + horizon)
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def __repr__(self):
        return (f"RollingWindowSplit(n_splits={self.n_splits}, "
                f"window_size={self.window_size}, "
                f"horizon={self.horizon}, "
                f"gap={self.gap})")

    def preview(self, X, width=80, style='default', train_char='=', test_char='-'):
        """
        Print a console preview of the rolling splits.

        Parameters:
            X (array-like): Input data.
            width (int): Width of the console bar.
            style (str): 'default' (text summary) or 'bar' (ascii visual).
            train_char (str): Character for train segment in bar view.
            test_char (str): Character for test segment in bar view.
        """
        n = len(X)
        try:
            splits = list(self.split(X))
        except ValueError as e:
            print("\nRollingWindowSplit error:", e)
            print("Hint: Try reducing n_splits, window_size, or horizon.\n")
            return

        if style == 'default':
            print(f"\nRollingWindowSplit: {self.n_splits} folds\n")
            for i, (train_idx, test_idx) in enumerate(splits):
                print(f"Fold {i + 1}:")
                print(f"  Train: {train_idx[0]} -> {train_idx[-1]}  (len={len(train_idx)})")
                print(f"  Test : {test_idx[0]} -> {test_idx[-1]}  (len={len(test_idx)})")
                print()
        elif style == 'bar':
            print(f"\nRollingWindowSplit Visual Preview (width={width}):\n")
            for i, (train_idx, test_idx) in enumerate(splits):
                line = [' '] * width
                train_start = int((train_idx[0] / n) * width)
                train_end = int((train_idx[-1] / n) * width)
                test_start = int((test_idx[0] / n) * width)
                test_end = int((test_idx[-1] / n) * width)

                for j in range(train_start, min(train_end + 1, width)):
                    line[j] = train_char
                for j in range(test_start, min(test_end + 1, width)):
                    line[j] = test_char

                print(f"Fold {i + 1:>2}: {''.join(line)}")
            print()
        else:
            raise ValueError("Invalid style. Use 'default' or 'bar'.")
