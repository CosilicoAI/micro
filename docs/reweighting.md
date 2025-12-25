# Reweighting Module

The `Reweighter` class implements sparse optimization for calibrating microdata to population targets.

## Overview

Reweighting finds optimal weights for synthetic microdata records to match official population statistics (margins/targets) while using the minimal number of records.

### Mathematical Formulation

The reweighting problem is formulated as:

```
minimize    ||w||_p
subject to  A @ w = b
            w >= 0
```

where:
- `w`: weight vector (decision variables)
- `A`: constraint matrix (indicator matrix for margins)
- `b`: target vector (population totals)
- `p`: sparsity norm (0, 1, or 2)

### Key Features

- **Multiple sparsity objectives**: L0, L1, L2 norms
- **Geographic hierarchies**: State, county, tract level targeting
- **Multiple backends**: scipy (default), cvxpy (optional)
- **Efficient sparse solutions**: L0 uses iterative reweighted L1 (IRL1)

## Installation

The reweighter requires `scipy` (included in base dependencies). For additional optimization capabilities, install `cvxpy`:

```bash
pip install microplex[cvxpy]
```

## Basic Usage

```python
from microplex import Reweighter
import pandas as pd

# Load synthetic microdata
data = pd.DataFrame({
    "state": ["CA", "CA", "NY", "NY", "TX", "TX"],
    "age_group": ["young", "old", "young", "old", "young", "old"],
    "income": [50000, 60000, 55000, 65000, 48000, 58000],
    "weight": [1, 1, 1, 1, 1, 1],  # Initial uniform weights
})

# Define population targets
targets = {
    "state": {"CA": 100, "NY": 50, "TX": 50},
    "age_group": {"young": 120, "old": 80},
}

# Fit and transform
reweighter = Reweighter(sparsity="l1")
weighted_data = reweighter.fit_transform(data, targets)

print(weighted_data["weight"])
```

## API Reference

### Reweighter

```python
class Reweighter:
    def __init__(
        self,
        backend: Literal["scipy", "cvxpy"] = "scipy",
        sparsity: Literal["l0", "l1", "l2"] = "l1",
        tol: float = 1e-4,
        max_iter: int = 1000,
    )
```

**Parameters:**
- `backend`: Optimization backend ("scipy" or "cvxpy")
- `sparsity`: Sparsity objective ("l0", "l1", or "l2")
  - `l0`: Minimize number of non-zero weights (sparsest)
  - `l1`: Minimize sum of weights (sparse)
  - `l2`: Minimize squared weights (dense)
- `tol`: Convergence tolerance
- `max_iter`: Maximum optimization iterations

### Methods

#### fit

```python
def fit(
    self,
    data: pd.DataFrame,
    targets: Dict[str, Dict[str, float]],
    weight_col: str = "weight",
) -> "Reweighter"
```

Fit weights to match population targets.

**Parameters:**
- `data`: DataFrame with microdata records
- `targets`: Nested dict `{margin_var: {category: count}}`
- `weight_col`: Name of weight column (optional)

**Returns:** `self`

**Example:**
```python
targets = {
    "state": {"CA": 1000, "NY": 500},
    "age_group": {"18-64": 1000, "65+": 500},
}
reweighter.fit(data, targets)
```

#### transform

```python
def transform(
    self,
    data: pd.DataFrame,
    weight_col: str = "weight",
    drop_zeros: bool = False,
) -> pd.DataFrame
```

Apply fitted weights to data.

**Parameters:**
- `data`: DataFrame to reweight
- `weight_col`: Weight column name
- `drop_zeros`: If True, remove zero-weight records

**Returns:** DataFrame with updated weights

#### fit_transform

```python
def fit_transform(
    self,
    data: pd.DataFrame,
    targets: Dict[str, Dict[str, float]],
    weight_col: str = "weight",
    drop_zeros: bool = False,
) -> pd.DataFrame
```

Fit and transform in one call (convenience method).

#### get_sparsity_stats

```python
def get_sparsity_stats() -> Dict[str, Union[int, float]]
```

Get statistics about fitted weights.

**Returns:** Dictionary with:
- `n_records`: Total records
- `n_nonzero`: Records with positive weight
- `sparsity`: Fraction of zero weights
- `max_weight`: Maximum weight value
- `total_weight`: Sum of all weights

## Sparsity Objectives

### L0 (Minimize Count)

**Objective:** `min ||w||_0 = min (number of non-zero weights)`

**Use case:** Extreme sparsity - use fewest possible records

**Algorithm:** Iterative Reweighted L1 (IRL1)

```python
reweighter = Reweighter(sparsity="l0")
```

**Properties:**
- Produces sparsest solutions
- Non-convex (uses approximation)
- Good for computational efficiency

### L1 (Minimize Sum)

**Objective:** `min ||w||_1 = min sum(w_i)`

**Use case:** Sparse solutions with computational guarantees

**Algorithm:** Linear programming (scipy.optimize.linprog)

```python
reweighter = Reweighter(sparsity="l1")
```

**Properties:**
- Convex optimization (globally optimal)
- Naturally sparse solutions
- Fast and reliable

### L2 (Minimize Squares)

**Objective:** `min ||w||_2^2 = min sum(w_i^2)`

**Use case:** Smooth weight distributions

**Algorithm:** Quadratic programming (scipy.optimize.minimize)

```python
reweighter = Reweighter(sparsity="l2")
```

**Properties:**
- Convex optimization
- Dense solutions (most records used)
- Penalizes large weights

## Advanced Examples

### Geographic Hierarchy

```python
# State and county level targets
targets = {
    "state": {
        "CA": 39_500_000,
        "NY": 19_500_000,
    },
    "county": {
        "Los Angeles": 10_000_000,
        "Orange": 3_100_000,
        "San Diego": 3_300_000,
        "New York": 1_600_000,
        "Kings": 2_600_000,
    },
}

reweighter = Reweighter(sparsity="l1")
weighted = reweighter.fit_transform(data, targets)
```

### Multiple Margin Variables

```python
# Match multiple demographic margins
targets = {
    "state": {"CA": 1000, "NY": 500, "TX": 500},
    "age_group": {"0-17": 400, "18-64": 1200, "65+": 400},
    "sex": {"M": 1000, "F": 1000},
}

reweighter = Reweighter(sparsity="l0")
weighted = reweighter.fit_transform(data, targets)

# Check how many records used
stats = reweighter.get_sparsity_stats()
print(f"Used {stats['n_nonzero']} records")
```

### Comparing Sparsity Methods

```python
import matplotlib.pyplot as plt

sparsities = []
for method in ["l0", "l1", "l2"]:
    rw = Reweighter(sparsity=method)
    result = rw.fit_transform(data, targets)
    stats = rw.get_sparsity_stats()
    sparsities.append({
        "method": method.upper(),
        "n_nonzero": stats["n_nonzero"],
        "max_weight": stats["max_weight"],
    })

df = pd.DataFrame(sparsities)
print(df)
```

### Drop Zero-Weight Records

```python
# Remove records with zero weight to reduce dataset size
weighted = reweighter.fit_transform(data, targets, drop_zeros=True)

print(f"Original records: {len(data)}")
print(f"Retained records: {len(weighted)}")
```

## Integration with Synthesizer

Combine synthesis and reweighting for end-to-end microdata creation:

```python
from microplex import Synthesizer, Reweighter

# Step 1: Synthesize microdata
synth = Synthesizer(
    target_vars=["income"],
    condition_vars=["age", "education", "state"],
)
synth.fit(training_data, epochs=100)
synthetic = synth.generate(demographics, n=10000)

# Step 2: Reweight to population targets
targets = {
    "state": {"CA": 4000, "NY": 3000, "TX": 3000},
}
reweighter = Reweighter(sparsity="l0")
calibrated = reweighter.fit_transform(synthetic, targets)

# Step 3: Analyze
stats = reweighter.get_sparsity_stats()
print(f"Final dataset: {stats['n_nonzero']} weighted records")
```

## Performance Considerations

### Computational Complexity

- **L1/L2**: Polynomial time (efficient for large problems)
- **L0**: Iterative approximation (may be slower)

### Problem Size

- **Small** (<10k records, <10 margins): All methods work well
- **Medium** (10k-100k records, 10-50 margins): L1 recommended
- **Large** (>100k records, >50 margins): L1 with sparse backends

### Tips for Large Datasets

1. Use L1 for speed and reliability
2. Consider cvxpy backend for complex constraints
3. Pre-filter data to relevant categories
4. Use hierarchical reweighting (state → county → tract)

## Optimization Backends

### scipy (default)

**Pros:**
- No additional dependencies
- Fast for L1/L2
- Stable and well-tested

**Cons:**
- L0 uses approximation
- Limited to standard problem forms

### cvxpy (optional)

**Pros:**
- More flexible problem formulations
- Multiple solver options (ECOS, SCS, etc.)
- Better handling of complex constraints

**Cons:**
- Requires additional installation
- Can be slower for simple problems

**Installation:**
```bash
pip install cvxpy
```

**Usage:**
```python
reweighter = Reweighter(backend="cvxpy", sparsity="l1")
```

## Error Handling

### Common Errors

**ValueError: Data contains categories not in targets**
```python
# Solution: Ensure all data categories have targets
targets = {
    "state": {"CA": 100, "NY": 50, "TX": 50, "FL": 25}
}
```

**ValueError: Reweighter not fitted**
```python
# Solution: Call fit() before transform()
reweighter.fit(data, targets)
result = reweighter.transform(data)
```

**ValueError: Data length doesn't match fitted length**
```python
# Solution: Use same data for fit and transform
reweighter.fit(data, targets)
result = reweighter.transform(data)  # Same data
```

## Validation

Check that weights match targets:

```python
weighted = reweighter.fit_transform(data, targets)

# Verify state targets
for state, target in targets["state"].items():
    actual = weighted[weighted["state"] == state]["weight"].sum()
    error = abs(actual - target) / target * 100
    print(f"{state}: {actual:.0f} (target: {target}, error: {error:.2f}%)")
```

## References

- **Iterative Reweighted L1**: Candès et al. (2008) "Enhancing Sparsity by Reweighted ℓ1 Minimization"
- **Survey Calibration**: Deville & Särndal (1992) "Calibration Estimators in Survey Sampling"
- **Sparse Optimization**: Boyd & Vandenberghe (2004) "Convex Optimization"
