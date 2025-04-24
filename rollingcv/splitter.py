from sklearn.model_selection import BaseCrossValidator
import numpy as np

class RollingWindowSplit(BaseCrossValidator):
    def __init__(self, n_splits=5, window_size=0.6, horizon=0.1, gap=0):
        self.n_splits = n_splits
        self.window_size = window_size
        self.horizon = horizon
        self.gap = gap
        self._validate_params()

    def _validate_params(self):
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        for name, val in [("window_size", self.window_size), ("horizon", self.horizon), ("gap", self.gap)]:
            if not isinstance(val, (int, float)):
                raise ValueError(f"{name} must be int or float")
            if isinstance(val, float) and not (0 < val < 1):
                raise ValueError(f"{name} as float must be between 0 and 1")
            if isinstance(val, int) and val < 0:
                raise ValueError(f"{name} as int must be non-negative")

    def split(self, X, y=None, groups=None):
        n = len(X)
        window = int(self.window_size * n) if isinstance(self.window_size, float) else self.window_size
        horizon = int(self.horizon * n) if isinstance(self.horizon, float) else self.horizon

        total_needed = window + self.gap + horizon + (self.n_splits - 1)
        if n < total_needed:
            raise ValueError("Not enough data for the requested number of splits.")

        max_start = n - window - self.gap - horizon
        step = max_start // (self.n_splits - 1)

        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + window
            test_start = train_end + self.gap
            test_end = test_start + horizon
            yield np.arange(train_start, train_end), np.arange(test_start, test_end)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def __repr__(self):
        return (f"RollingWindowSplit(n_splits={self.n_splits}, "
                f"window_size={self.window_size}, horizon={self.horizon}, gap={self.gap})")

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
            print(f"[Error] Cannot preview: {e}")
            return

        if style == 'default':
            print(f"\nRollingWindowSplit: {self.n_splits} folds\n")
            for i, (train_idx, test_idx) in enumerate(splits):
                print(f"Fold {i + 1}:")
                print(f"  Train: {train_idx[0]} → {train_idx[-1]}  (len={len(train_idx)})")
                print(f"  Test : {test_idx[0]} → {test_idx[-1]}  (len={len(test_idx)})\n")

        elif style == 'bar':
            print(f"\nRollingWindowSplit Visual Preview (width={width}):\n")
            def scale(x): return min(round((x / n) * width), width - 1)

            label_width = len(f"Fold {len(splits)}: ")

            for i, (train_idx, test_idx) in enumerate(splits):
                line = [' '] * width
                train_start = train_idx[0]
                train_end = train_idx[-1]
                test_start = test_idx[0]
                test_end = test_idx[-1]

                for j in range(scale(train_start), scale(train_end + 1)):
                    if j < width:
                        line[j] = train_char
                for j in range(scale(test_start), scale(test_end + 1)):
                    if j < width:
                        line[j] = test_char

                label = f"Fold {i + 1}:".rjust(label_width)
                print(f"{label} {''.join(line)}")
            print()
        else:
            raise ValueError("Invalid style. Use 'default' or 'bar'.")
