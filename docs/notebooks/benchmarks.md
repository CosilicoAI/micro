# Benchmarks

This notebook compares `micro` against other synthesis methods.

## Methods Compared

| Method | Description | Library |
|--------|-------------|---------|
| **micro** | Normalizing flows | This package |
| **CT-GAN** | Conditional Tabular GAN | SDV |
| **TVAE** | Tabular VAE | SDV |
| **Copula** | Gaussian Copula | SDV |

## Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from benchmarks.compare import run_benchmark, results_to_dataframe
```

## Create Test Data

```python
np.random.seed(42)
n_train = 5000
n_test = 1000

# Demographics
age = np.random.randint(18, 80, n_train)
education = np.random.choice([1, 2, 3, 4], n_train)

# Income sources (zero-inflated)
base = np.random.lognormal(10, 1, n_train)
wages = base * (1 + 0.01 * (age - 18)) * (1 + 0.2 * education)
wages[np.random.random(n_train) < 0.08] = 0

capital_gains = np.where(
    base > np.percentile(base, 70),
    np.random.lognormal(9, 2, n_train),
    0
)

training_data = pd.DataFrame({
    "age": age,
    "education": education,
    "wages": wages,
    "capital_gains": capital_gains,
})

# Test conditions
test_conditions = pd.DataFrame({
    "age": np.random.randint(18, 80, n_test),
    "education": np.random.choice([1, 2, 3, 4], n_test),
})
```

## Run Benchmarks

```python
results = run_benchmark(
    train_data=training_data,
    test_conditions=test_conditions,
    target_vars=["wages", "capital_gains"],
    condition_vars=["age", "education"],
    methods=["micro", "ctgan", "tvae", "copula"],
    epochs=100,
)

df = results_to_dataframe(results)
df
```

## Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# KS statistic (lower is better)
ax = axes[0, 0]
df.plot.bar(x="method", y="mean_ks", ax=ax, legend=False)
ax.set_title("Marginal Fidelity (KS Statistic)")
ax.set_ylabel("KS Statistic (lower is better)")

# Correlation error (lower is better)
ax = axes[0, 1]
df.plot.bar(x="method", y="correlation_error", ax=ax, legend=False, color="orange")
ax.set_title("Joint Fidelity (Correlation Error)")
ax.set_ylabel("Error (lower is better)")

# Zero fraction error (lower is better)
ax = axes[1, 0]
df.plot.bar(x="method", y="mean_zero_error", ax=ax, legend=False, color="green")
ax.set_title("Zero Fraction Error")
ax.set_ylabel("Error (lower is better)")

# Training time
ax = axes[1, 1]
df.plot.bar(x="method", y="train_time", ax=ax, legend=False, color="red")
ax.set_title("Training Time")
ax.set_ylabel("Seconds")

plt.tight_layout()
plt.show()
```

## Key Findings

1. **Marginal Fidelity**: `micro` achieves competitive KS statistics
2. **Zero Handling**: `micro` excels at preserving zero fractions
3. **Correlations**: Normalizing flows preserve joint distributions well
4. **Speed**: Training time comparable to TVAE, faster than CT-GAN
