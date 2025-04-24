
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from rollingcv import RollingWindowSplit

# Simulated time series
data = np.arange(1000)

# Create the cross-validator
rws = RollingWindowSplit(n_splits=50, window_size=0.6, horizon=0.1, gap=5)

# Show a simple text summary
rws.preview(data, style='default')

# Show a console bar visualization
rws.preview(data, style='bar', width=80)
