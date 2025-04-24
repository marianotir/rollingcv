import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from rollingcv import RollingWindowSplit

# -----------------------
# ✅ Functionality Tests
# -----------------------

def test_rolling_window_split_sizes():
    data = np.arange(1000)
    rws = RollingWindowSplit(n_splits=5, window_size=100, horizon=10)

    for i, (train_idx, test_idx) in enumerate(rws.split(data)):
        assert len(train_idx) == 100, f"Fold {i+1}: Expected train length 100, got {len(train_idx)}"
        assert len(test_idx) == 10, f"Fold {i+1}: Expected test length 10, got {len(test_idx)}"

def test_rolling_window_split_count():
    data = np.arange(1000)
    rws = RollingWindowSplit(n_splits=5, window_size=100, horizon=10)

    splits = list(rws.split(data))
    assert len(splits) == 5, f"Expected 5 splits, got {len(splits)}"

# -----------------------
# 🚨 Error Handling Tests
# -----------------------

def test_invalid_n_splits():
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        RollingWindowSplit(n_splits=1, window_size=100, horizon=10)

def test_negative_values():
    with pytest.raises(ValueError, match="must be non-negative"):
        RollingWindowSplit(n_splits=5, window_size=-10, horizon=10)

    with pytest.raises(ValueError, match="must be non-negative"):
        RollingWindowSplit(n_splits=5, window_size=100, horizon=-1)

    with pytest.raises(ValueError, match="must be non-negative"):
        RollingWindowSplit(n_splits=5, window_size=100, horizon=10, gap=-5)

def test_float_out_of_bounds():
    with pytest.raises(ValueError, match="window_size must be between 0 and 1"):
        RollingWindowSplit(n_splits=5, window_size=1.5, horizon=0.1)

    with pytest.raises(ValueError, match="horizon must be between 0 and 1"):
        RollingWindowSplit(n_splits=5, window_size=0.5, horizon=1.2)

def test_insufficient_data():
    data = np.arange(10)
    rws = RollingWindowSplit(n_splits=5, window_size=6, horizon=3, gap=2)
    with pytest.raises(ValueError, match="Not enough data"):
        list(rws.split(data))
