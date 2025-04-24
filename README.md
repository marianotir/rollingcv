<p align="center">
  <img src="assets/logo.png" alt="RollingCV Logo" width="200"/>
</p>

 RollingWindowSplit

**A smarter, production-ready alternative to scikit-learn's TimeSeriesSplit.**

Perform rolling window cross-validation with fixed-size training sets and forecast horizons. Fully compatible with scikit-learn pipelines and supports percent-based sizes, configurable gaps, and clean fold visualization — even without matplotlib.

---

## 🚀 Why RollingWindowSplit?

When you use `TimeSeriesSplit`, each fold has a larger training set than the previous. That means:

- You're training **different models** on **different data sizes** in each fold  
- You're not properly evaluating the model you plan to deploy  
- You're not reproducing the same model `k` times  

With `RollingWindowSplit`, you:

- ✅ Fix the training window size (just like in production)  
- ✅ Fix the forecast horizon (true out-of-sample evaluation)  
- ✅ Optionally add a gap to avoid data leakage  
- ✅ Run folds that actually replicate your deployment logic  

---

## 📦 Installation

Install via pip:

```
pip install rollingcv
```

Or install locally for development:

```
pip install -e .
```

---

## 💡 Example

```python
import numpy as np
from rollingcv import RollingWindowSplit

data = np.arange(1000)

rws = RollingWindowSplit(n_splits=5, window_size=0.6, horizon=0.1, gap=5)

# Show a text preview of the folds
rws.preview(data, style='default')

# Show a bar-style fold preview in console
rws.preview(data, style='bar')
```

---

### 🔍 Console Preview (Bar Style)

```
RollingWindowSplit Visual Preview (width=80):

Fold  1: ====================-----                    
Fold  2:   ====================-----                  
Fold  3:     ====================-----                
Fold  4:       ====================-----              
Fold  5:         ====================-----            
```

---

## 🧠 Key Features

- ✅ Fixed or percent-based window and horizon sizes  
- ✅ Optional `gap` to simulate production delays  
- ✅ Clean `__repr__` for logging/debugging  
- ✅ Compatible with scikit-learn pipelines  
- ✅ Console previews — no matplotlib required  

---

## 🛡️ Error Handling

RollingWindowSplit protects against common pitfalls:

- `n_splits` must be at least 2  
- `window_size`, `horizon`, and `gap` must be non-negative  
- Float sizes must be between 0 and 1  
- Raises a friendly message if there's not enough data to split  

---

## 🧪 Testing

Run the built-in unit tests with:

```
pytest tests/
```

---

## 📄 License

MIT License. Use it, modify it, love it.

---

## ✨ Contribute

Pull requests welcome! Fork it, play with it, and if you improve it — share it back!

---

## 🙌 Credits

Created with care by [marianotir](https://github.com/marianotir)