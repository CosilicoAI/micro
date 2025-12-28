# Sparse Calibration for Microdata Synthesis

**Abstract**: Survey calibration adjusts sample weights to match known population totals. When synthesizing microdata from multiple sources, we face a different problem: selecting a minimal subset of synthetic records that, when weighted, reproduce thousands of hierarchical targets. We review classical calibration methods (IPF, chi-square, entropy), discuss why they are insufficient for large-scale synthetic data calibration, and present a cross-category L0 selection approach that achieves exact sparsity while preserving joint distributions.

## 1. Introduction

Microdata synthesis requires calibrating synthetic populations to match official statistics at multiple geographic levels. A typical application might require matching:

- ~50 federal targets (total income, employment, benefits by program)
- ~2,500 state-level targets (50 states × 50 variables)
- ~150,000 county-level targets (3,000 counties × 50 variables)
- Millions of tract-level targets

This scale fundamentally changes the calibration problem. Classical survey calibration assumes:

1. A single sample with design weights to preserve
2. A modest number of non-conflicting targets
3. Exact target satisfaction as the goal

None of these assumptions hold for synthetic data calibration at scale.

## 2. Classical Calibration Methods

### 2.1 Iterative Proportional Fitting (IPF)

IPF, also known as raking, iteratively adjusts weights to match marginal totals:

```
for each margin m:
    adjustment = target[m] / current_weighted_sum[m]
    weights[in_margin_m] *= adjustment
```

**Properties:**
- Simple and fast
- Converges for consistent targets
- No explicit objective function
- Stops when targets are hit (or fails)

**Limitation:** IPF provides no mechanism to balance conflicting targets. With thousands of hierarchical targets, some will inevitably conflict (e.g., state totals may not exactly equal sum of county totals due to sampling error in the targets themselves).

### 2.2 Chi-Square Distance Minimization

Chi-square calibration minimizes deviation from initial weights:

$$\min_w \sum_i \frac{(w_i - d_i)^2}{d_i} \quad \text{subject to} \quad Aw = b$$

where $d_i$ are design weights and $b$ are targets.

**Properties:**
- Preserves design weight structure
- Quadratic programming (efficient)
- Well-defined objective for tradeoffs

**Limitation:** Assumes meaningful initial weights. When synthesizing from multiple sources (CPS + ACS + IRS + SSA), there is no single "design weight" to preserve.

### 2.3 Entropy Balancing

Entropy calibration minimizes Kullback-Leibler divergence:

$$\min_w \sum_i w_i \log\frac{w_i}{d_i} \quad \text{subject to} \quad Aw = b$$

**Properties:**
- Maximum entropy principle
- Preserves initial distribution structure
- Smooth, exponential tilting

**Limitation:** Same as chi-square - requires meaningful initial weights.

## 3. The Multi-Source Synthesis Problem

When building synthetic populations from multiple data sources, we face fundamentally different constraints:

### 3.1 No Meaningful Initial Weights

Consider synthesizing a population using:
- Demographics from Census
- Income from CPS
- Tax variables from IRS SOI
- Benefits from SSA administrative data
- Housing from ACS

Each source has its own sampling design and weights. There is no single "initial weight" to preserve - we are creating a new population, not reweighting an existing sample.

### 3.2 Thousands of Potentially Conflicting Targets

With hierarchical geography (nation → state → county → tract), targets at different levels may conflict:

```
# Published statistics have measurement error
sum(county_populations) ≠ state_population  # off by ~0.1-1%
sum(tract_incomes) ≠ county_income          # off by ~1-5%
```

IPF cannot handle this - it assumes exact target satisfaction. We need a method that finds the best tradeoff across all targets.

### 3.3 Sparsity as a First-Class Goal

For computational efficiency, we want the smallest subset of synthetic records that adequately represents the population. This is an L0 (cardinality) constraint:

$$\min \|w\|_0 \quad \text{subject to} \quad Aw \approx b, \quad w \geq 0$$

## 4. Gradient Descent Calibration

One approach is to use gradient descent with a unified loss function:

$$\mathcal{L} = \sum_{j=1}^{M} \left( \frac{\sum_i w_i x_{ij} - b_j}{b_j + \epsilon} \right)^2 + \lambda \cdot \text{penalty}(w)$$

where:
- $M$ is the number of targets (potentially thousands)
- $x_{ij}$ is the value of target variable $j$ for record $i$
- $b_j$ is target $j$
- The normalization by $b_j + \epsilon$ makes targets of different magnitudes comparable

### 4.1 Advantages

1. **Automatic tradeoffs**: When targets conflict, gradient descent finds the best compromise
2. **No tolerance setting**: One loss function, no per-target tolerances to tune
3. **Differentiable sparsity**: Can use L0 relaxations (Hard Concrete distribution) for end-to-end optimization

### 4.2 Disadvantages

1. **Slow**: Requires 1000+ epochs for convergence
2. **Approximate zeros**: Differentiable L0 relaxations produce near-zero, not exact zero weights
3. **Hyperparameter sensitivity**: Learning rate, temperature, regularization strength all matter

## 5. Cross-Category L0 Selection

We propose an alternative that achieves exact sparsity while preserving calibration accuracy.

### 5.1 Key Insight: Cross-Category Structure

For categorical constraints (state, age group, income bracket), records belong to discrete cross-categories:

```
Record 1: (state=CA, age=25-34, income=50k-75k)
Record 2: (state=CA, age=25-34, income=75k-100k)
Record 3: (state=NY, age=35-44, income=50k-75k)
...
```

When selecting a subset for sparsity, we must preserve the joint distribution across all categorical dimensions, not just the marginals.

### 5.2 Algorithm

```python
def sparse_calibrate(data, targets, target_sparsity):
    # Step 1: Identify cross-categories
    # Each record belongs to exactly one (state × age × income × ...) cell
    cross_cats = group_by_all_categorical_constraints(data)

    # Step 2: Select proportionally from each cross-category
    k = int(len(data) * (1 - target_sparsity))
    selected = []
    for cell in cross_cats:
        n_keep = max(1, int(len(cell) * k / len(data)))
        selected.extend(random_sample(cell, n_keep))

    # Step 3: Calibrate selected records via IPF
    weights = ipf_calibrate(data[selected], targets)

    return weights
```

### 5.3 Properties

1. **Exact sparsity**: Achieves exactly the target sparsity level
2. **Preserved joint distribution**: Cross-category selection ensures the selected subset has the correct joint distribution for all categorical constraints
3. **Fast**: O(n) selection + O(iterations × constraints) IPF
4. **Exact zeros**: Non-selected records have exactly zero weight

### 5.4 Limitations

1. **Categorical constraints only**: The cross-category approach requires discrete categories
2. **Continuous targets are secondary**: After selection, continuous targets may have higher error
3. **IPF limitations apply**: Still assumes non-conflicting targets within the selected subset

## 6. When to Use Each Approach

| Scenario | Recommended Method |
|----------|-------------------|
| Simple calibration (5-10 margins) | IPF or Chi-square |
| Preserve design weights | Chi-square or Entropy |
| Thousands of hierarchical targets | Gradient Descent |
| Need exact sparsity + categorical accuracy | Cross-Category L0 |
| Conflicting targets that need tradeoffs | Gradient Descent |
| Fast iteration, simple constraints | Cross-Category L0 |

## 7. Open Questions

1. **Hybrid approaches**: Can we combine cross-category selection with gradient descent for final calibration?

2. **Continuous target handling**: How should we modify cross-category selection to better handle continuous targets (total income, total benefits)?

3. **Hierarchical consistency**: When targets are hierarchical (state = sum of counties), how do we propagate consistency constraints?

4. **Uncertainty quantification**: How do we quantify uncertainty in calibrated estimates when sparsity discards information?

5. **Adaptive sparsity**: Can sparsity vary by geography (more records for larger counties)?

## 8. Conclusion

Large-scale microdata synthesis requires rethinking classical calibration. IPF's "hit target or fail" approach cannot handle thousands of potentially conflicting targets. Chi-square and entropy methods assume meaningful initial weights that don't exist for multi-source synthesis.

Gradient descent with unified loss functions provides automatic tradeoffs but is slow and produces approximate sparsity. Cross-category L0 selection achieves exact sparsity with categorical accuracy but struggles with continuous targets.

The optimal approach likely combines elements of both: cross-category selection for efficient L0 sparsity, followed by gradient-based refinement for continuous targets and conflicting constraints.

## References

1. Deville, J.C. & Särndal, C.E. (1992). Calibration Estimators in Survey Sampling. *Journal of the American Statistical Association*, 87(418), 376-382.

2. Hainmueller, J. (2012). Entropy Balancing for Causal Effects: A Multivariate Reweighting Method to Produce Balanced Samples in Observational Studies. *Political Analysis*, 20(1), 25-46.

3. Louizos, C., Welling, M., & Kingma, D.P. (2017). Learning Sparse Neural Networks through L0 Regularization. *arXiv:1712.01312*.

4. Candès, E.J., Wakin, M.B., & Boyd, S.P. (2008). Enhancing Sparsity by Reweighted ℓ1 Minimization. *Journal of Fourier Analysis and Applications*, 14(5), 877-905.

5. Ireland, C.T. & Kullback, S. (1968). Contingency Tables with Given Marginals. *Biometrika*, 55(1), 179-188.
