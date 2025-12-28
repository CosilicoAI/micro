# Sparse Calibration for Microdata Synthesis

**Abstract**: Survey calibration adjusts sample weights to match known population totals. When synthesizing microdata from multiple sources, we face a different problem: selecting a minimal subset of synthetic records that, when weighted, reproduce thousands of hierarchical targets. We review classical calibration methods (IPF, chi-square, entropy), their theoretical foundations and limitations, discuss sparse optimization approaches from compressed sensing and machine learning, examine PolicyEngine's Hard Concrete L0 implementation as a case study, and present a cross-category L0 selection approach that achieves exact sparsity while preserving joint distributions.

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

#### 2.1.1 Historical Development

IPF, also known as raking or biproportional fitting, has been independently discovered multiple times across different fields:

- **Kruithof (1937)**: Developed the "double factor method" for telephone traffic analysis
- **Deming & Stephan (1940)**: Introduced IPF for the 1940 U.S. Census to reconcile sample cross-tabulations with known marginal totals
- **Sinkhorn (1964)**: Proved convergence for positive matrices in the context of doubly stochastic matrices
- **Bishop, Fienberg & Holland (1975)**: Provided comprehensive treatment in *Discrete Multivariate Analysis*, establishing IPF as the standard for log-linear models

The algorithm iteratively adjusts weights to match marginal totals:

```
for each margin m:
    adjustment = target[m] / current_weighted_sum[m]
    weights[in_margin_m] *= adjustment
```

#### 2.1.2 Convergence Theory

Key theoretical contributions establish convergence conditions:

- **Sinkhorn & Knopp (1967)**: Proved convergence for nonnegative matrices with "total support" (containing at least one positive diagonal)
- **Csiszar (1975)**: Established the information-theoretic foundation, showing IPF minimizes Kullback-Leibler divergence. Provided necessary and sufficient conditions for convergence with zero entries
- **Fienberg (1970)**: Gave geometric proof using differential geometry, interpreting contingency tables as manifolds
- **Ruschendorf (1995)**: Extended convergence theory to continuous bivariate densities

#### 2.1.3 Properties and Limitations

**Properties:**
- Simple and fast
- Converges for consistent targets
- No explicit objective function
- Stops when targets are hit (or fails)

**Limitations:**
- **No tradeoff mechanism**: IPF provides no way to balance conflicting targets. With thousands of hierarchical targets, some will inevitably conflict
- **Categorical variables only**: Fundamentally operates on discrete contingency tables
- **Empty cell problem**: Zero cells can cause convergence failure
- **Fractional weights**: Produces non-integer weights requiring "integerization" for agent-based microsimulation
- **Slow convergence**: Linear convergence in worst case (Fienberg 1970)

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

**Theoretical Foundation:**
- **Ireland & Kullback (1968)**: Formulated IPF as minimum discrimination information estimation
- **Hainmueller (2012)**: Developed entropy balancing for causal inference, achieving exact covariate balance by reweighting

**Properties:**
- Maximum entropy principle
- Guarantees non-negative weights (unlike GREG)
- Smooth, exponential tilting
- Doubly robust when combined with outcome regression (Zhao & Percival 2017)

**Limitation:** Same as chi-square - requires meaningful initial weights. May fail to converge in sparse scenarios or when positivity is violated.

### 2.4 Generalized Regression Estimation (GREG)

The GREG estimator provides a unified framework for calibration:

- **Deville & Särndal (1992)**: Showed that raking is a special case of minimizing distance between adjusted and original weights subject to calibration equations
- **Särndal, Swensson & Wretman (1992)**: Established the model-assisted paradigm where estimators are asymptotically unbiased regardless of model specification

GREG extends calibration to continuous auxiliary variables but shares limitations regarding initial weights.

## 3. Sparse Optimization Background

### 3.1 The L0 Minimization Problem

Finding the sparsest solution to an underdetermined system is NP-hard:

$$\min \|w\|_0 \quad \text{subject to} \quad Aw = b$$

where $\|w\|_0$ counts non-zero entries. Natarajan (1995) proved this computational intractability, motivating convex relaxations.

### 3.2 Compressed Sensing Foundations

Two seminal papers established that sparse solutions can be recovered efficiently:

- **Candes, Romberg & Tao (2006)**: Proved that sparse signals can be exactly recovered from incomplete measurements via L1 minimization with high probability, provided the measurement matrix satisfies the Restricted Isometry Property (RIP)
- **Donoho (2006)**: Demonstrated that signals with sparse representations can be recovered from far fewer samples than Nyquist-Shannon requires

These theoretical guarantees justify using L1 relaxation for sparse calibration:

$$\min \|w\|_1 \quad \text{subject to} \quad Aw = b$$

### 3.3 LASSO and Extensions

- **Tibshirani (1996)**: Introduced LASSO, which naturally produces exact zeros via L1 penalty
- **Zou & Hastie (2005)**: Developed Elastic Net combining L1 and L2 penalties for grouping correlated variables
- **Candes, Wakin & Boyd (2008)**: Proposed iteratively reweighted L1 (IRL1) to better approximate L0

### 3.4 Optimization Algorithms

Modern algorithms efficiently solve L1-penalized problems:

- **ADMM** (Boyd et al. 2011): Decomposes problems with L1 penalties plus constraints
- **ISTA/FISTA** (Beck & Teboulle 2009): Proximal gradient methods with soft-thresholding
- **Best Subset Selection** (Bertsimas et al. 2016): Mixed-integer optimization can solve exact L0 problems for moderate sizes

## 4. Statistical Matching and Data Fusion

### 4.1 The Identification Problem

When combining data sources that don't jointly observe all variables, the full joint distribution cannot be uniquely identified. D'Orazio, Di Zio & Scanu (2006) provide the foundational treatment in *Statistical Matching: Theory and Practice*.

### 4.2 Conditional Independence Assumption

Statistical matching typically assumes:

$$P(Y, Z | X) = P(Y | X) \cdot P(Z | X)$$

This simplifies matching but may bias correlations toward zero. Singh et al. (1993) showed auxiliary information can substitute for this assumption.

### 4.3 Fréchet Bounds

Without additional assumptions, Fréchet-Hoeffding bounds provide sharp limits on joint distributions given marginals:

$$\max\{0, P(A) + P(B) - 1\} \leq P(A,B) \leq \min\{P(A), P(B)\}$$

### 4.4 Uncertainty Quantification

- **Moriarity & Scheuren (2001, 2003)**: Established framework for assessing uncertainty in statistical matching
- **Rässler (2002)**: Proposed multiple imputation with informative priors to overcome conditional independence
- **Edwards & Tanton (2016)**: Addressed uncertainty estimation in spatial microsimulation

## 5. Small Area Estimation

### 5.1 Fay-Herriot Model

The Fay-Herriot (1979) area-level model is fundamental for borrowing strength across sparse areas:

$$\hat{\theta}_i = \gamma_i \hat{\theta}_i^{direct} + (1-\gamma_i) \hat{\theta}_i^{synthetic}$$

where the shrinkage factor $\gamma_i$ depends on relative variance of direct vs. synthetic estimators.

### 5.2 Hierarchical Bayesian Methods

- **Molina, Nandram & Rao (2014)**: Hierarchical Bayes for complex nonlinear parameters like poverty indices
- Provides both point estimates and proper uncertainty quantification, unlike frequentist EBLUP

### 5.3 Modern Synthetic Population Generation

Recent large-scale implementations:

- **Nature Scientific Data (2025)**: U.S. national dataset generating 120M+ household synthetic population from ACS
- **Nature Scientific Data (2022)**: UK SIPHER dataset combining Census with Understanding Society survey
- **World Bank REaLTabFormer**: Transformer architecture for hierarchical synthetic population generation

## 6. The Multi-Source Synthesis Problem

When building synthetic populations from multiple data sources, we face fundamentally different constraints:

### 6.1 No Meaningful Initial Weights

Consider synthesizing a population using:
- Demographics from Census
- Income from CPS
- Tax variables from IRS SOI
- Benefits from SSA administrative data
- Housing from ACS

Each source has its own sampling design and weights. There is no single "initial weight" to preserve - we are creating a new population, not reweighting an existing sample.

### 6.2 Thousands of Potentially Conflicting Targets

With hierarchical geography (nation → state → county → tract), targets at different levels may conflict:

```
# Published statistics have measurement error
sum(county_populations) ≠ state_population  # off by ~0.1-1%
sum(tract_incomes) ≠ county_income          # off by ~1-5%
```

IPF cannot handle this - it assumes exact target satisfaction. We need a method that finds the best tradeoff across all targets.

### 6.3 Sparsity as a First-Class Goal

For computational efficiency, we want the smallest subset of synthetic records that adequately represents the population. This is an L0 (cardinality) constraint:

$$\min \|w\|_0 \quad \text{subject to} \quad Aw \approx b, \quad w \geq 0$$

## 7. PolicyEngine's Hard Concrete L0 Approach

PolicyEngine has implemented a production-grade L0 regularization system across multiple repositories. This section documents their methodology as a case study.

### 7.1 Hard Concrete Distribution

Based on Louizos, Welling & Kingma (2017), the Hard Concrete distribution provides a differentiable approximation of discrete L0 regularization.

For each weight, a gate $z_i \in [0,1]$ is learned via:

**Sampling (Training):**
$$s = \sigma\left(\frac{\log u - \log(1-u) + \alpha}{\tau}\right)$$
$$\bar{s} = s(\zeta - \gamma) + \gamma$$
$$z = \min(1, \max(0, \bar{s}))$$

where $u \sim U(\epsilon, 1-\epsilon)$, $\alpha$ are learnable logits, $\tau$ is temperature, and $\gamma=-0.1, \zeta=1.1$ are stretch parameters encouraging exact zeros.

**L0 Penalty:**
$$\mathbb{E}[\|z\|_0] = \sum_i \sigma\left(\alpha_i - \tau \log(-\gamma/\zeta)\right)$$

### 7.2 Implementation Architecture

PolicyEngine's implementation spans several repositories:

**L0 Package** (`l0-python`):
- `distributions.py`: HardConcrete distribution
- `gates.py`: SampleGate, FeatureGate, HybridGate for selection
- `calibration.py`: SparseCalibrationWeights combining gates with positive weights

**MicroCalibrate** (`microcalibrate`):
- Two-phase optimization: dense reweighting then L0 regularization
- Adam optimizer with log-space weight parameterization
- Relative squared error loss with normalization

**PolicyEngine-US-Data** (`policyengine-us-data`):
- 2,813+ calibration targets from IRS, Census, CBO, healthcare sources
- Group-wise loss averaging for balanced contribution across target types
- Geographic stratification for congressional district analysis

### 7.3 Loss Function

The core loss function uses relative squared error:

$$\mathcal{L} = \frac{1}{K} \sum_{k=1}^{K} \left( \frac{f_k(w \odot g) - t_k}{t_k + 1} \right)^2 + \lambda \mathbb{E}[\|g\|_0]$$

where:
- $w = \exp(\log w)$ ensures positivity
- $g \sim \text{HardConcrete}$ are learned gates
- The $+1$ offset prevents division by zero
- Normalization makes targets of different magnitudes comparable

### 7.4 Two-Phase Optimization

**Phase 1 (Dense):**
- Optimize weights using Adam with learning rate ~0.001
- Optional dropout regularization
- ~2,000 epochs

**Phase 2 (Sparse):**
- Add Hard Concrete gates with L0 penalty
- Higher learning rate (~0.2) for gate parameters
- ~4,000 epochs (doubled due to increased difficulty)

### 7.5 Design Rationale

PolicyEngine chose gradient descent over IPF for several reasons:

1. **Automatic tradeoffs**: When targets conflict, the loss function finds the best compromise
2. **No tolerance setting**: One objective, no per-target tolerances to tune
3. **End-to-end sparsity**: L0 gates are jointly optimized with weights
4. **Scalability**: Handles 2,800+ targets efficiently with sparse matrices

Key hyperparameters (from production use):
- `l0_lambda = 2.6e-7`: Regularization strength
- `temperature = 0.25`: Low for hard gate decisions
- `init_mean = 0.999`: Start with most weights active

## 8. Gradient Descent Calibration

### 8.1 Unified Loss Formulation

One approach is to use gradient descent with a unified loss function:

$$\mathcal{L} = \sum_{j=1}^{M} \left( \frac{\sum_i w_i x_{ij} - b_j}{b_j + \epsilon} \right)^2 + \lambda \cdot \text{penalty}(w)$$

where:
- $M$ is the number of targets (potentially thousands)
- $x_{ij}$ is the value of target variable $j$ for record $i$
- $b_j$ is target $j$
- The normalization by $b_j + \epsilon$ makes targets of different magnitudes comparable

### 8.2 Advantages

1. **Automatic tradeoffs**: When targets conflict, gradient descent finds the best compromise
2. **No tolerance setting**: One loss function, no per-target tolerances to tune
3. **Differentiable sparsity**: Can use L0 relaxations (Hard Concrete distribution) for end-to-end optimization

### 8.3 Disadvantages

1. **Slow**: Requires 1000+ epochs for convergence
2. **Approximate zeros**: Differentiable L0 relaxations produce near-zero, not exact zero weights
3. **Hyperparameter sensitivity**: Learning rate, temperature, regularization strength all matter

## 9. Cross-Category L0 Selection

We propose an alternative that achieves exact sparsity while preserving calibration accuracy.

### 9.1 Key Insight: Cross-Category Structure

For categorical constraints (state, age group, income bracket), records belong to discrete cross-categories:

```
Record 1: (state=CA, age=25-34, income=50k-75k)
Record 2: (state=CA, age=25-34, income=75k-100k)
Record 3: (state=NY, age=35-44, income=50k-75k)
...
```

When selecting a subset for sparsity, we must preserve the joint distribution across all categorical dimensions, not just the marginals.

### 9.2 Algorithm

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

### 9.3 Properties

1. **Exact sparsity**: Achieves exactly the target sparsity level
2. **Preserved joint distribution**: Cross-category selection ensures the selected subset has the correct joint distribution for all categorical constraints
3. **Fast**: O(n) selection + O(iterations × constraints) IPF
4. **Exact zeros**: Non-selected records have exactly zero weight

### 9.4 Limitations

1. **Categorical constraints only**: The cross-category approach requires discrete categories
2. **Continuous targets are secondary**: After selection, continuous targets may have higher error
3. **IPF limitations apply**: Still assumes non-conflicting targets within the selected subset

## 10. Imputation Methods for Synthesis

Beyond calibration, microdata synthesis often requires imputation for variables not jointly observed.

### 10.1 Random Forest Imputation

RF-based methods show strong performance for tabular data:
- Handles mixed types, interactions, nonlinearity
- ~30% improvement over conventional methods (benchmarks)
- PolicyEngine uses Quantile Regression Forests to preserve conditional distributions

### 10.2 Copula-Based Methods

Copulas model marginal distributions and dependence structure separately:
- **Gaussian copula** (arXiv 2019): Handles continuous, ordinal, categorical jointly
- Relaxes normality assumptions for multivariate mixed data

### 10.3 Hot Deck Imputation

Despite ML advances, nearest neighbor hot deck remains competitive:
- Preserves observed value distributions
- No parametric assumptions
- StatMatch R package implements k-NN and probabilistic methods

## 11. When to Use Each Approach

| Scenario | Recommended Method |
|----------|-------------------|
| Simple calibration (5-10 margins) | IPF or Chi-square |
| Preserve design weights | Chi-square or Entropy |
| Thousands of hierarchical targets | Gradient Descent with L0 |
| Need exact sparsity + categorical accuracy | Cross-Category L0 |
| Conflicting targets that need tradeoffs | Gradient Descent |
| Fast iteration, simple constraints | Cross-Category L0 |
| Borrowing strength across sparse areas | Fay-Herriot / Hierarchical Bayes |
| Multi-source data fusion | Statistical matching + calibration |

## 12. Open Questions

1. **Hybrid approaches**: Can we combine cross-category selection with gradient descent for final calibration?

2. **Continuous target handling**: How should we modify cross-category selection to better handle continuous targets (total income, total benefits)?

3. **Hierarchical consistency**: When targets are hierarchical (state = sum of counties), how do we propagate consistency constraints?

4. **Uncertainty quantification**: How do we quantify uncertainty in calibrated estimates when sparsity discards information? Multiple imputation and Fréchet bounds provide partial answers.

5. **Adaptive sparsity**: Can sparsity vary by geography (more records for larger counties)?

6. **Temperature scheduling**: What annealing schedule optimizes the tradeoff between exploration and exploitation in Hard Concrete gates?

7. **Best subset feasibility**: For what problem sizes can exact L0 optimization (Bertsimas et al. 2016) replace approximate methods?

## 13. Conclusion

Large-scale microdata synthesis requires rethinking classical calibration. IPF's "hit target or fail" approach cannot handle thousands of potentially conflicting targets. Chi-square and entropy methods assume meaningful initial weights that don't exist for multi-source synthesis.

Gradient descent with unified loss functions provides automatic tradeoffs but is slow and produces approximate sparsity. PolicyEngine's Hard Concrete L0 implementation demonstrates that production-scale calibration with 2,800+ targets is feasible, though requiring careful hyperparameter tuning and two-phase optimization.

Cross-category L0 selection achieves exact sparsity with categorical accuracy but struggles with continuous targets. The optimal approach likely combines elements of multiple methods: cross-category selection for efficient L0 sparsity, followed by gradient-based refinement for continuous targets and conflicting constraints, with hierarchical Bayesian methods for uncertainty quantification.

## References

### Classical Calibration

1. Deming, W.E. & Stephan, F.F. (1940). On a least squares adjustment of a sampled frequency table when the expected marginal totals are known. *Annals of Mathematical Statistics*, 11, 427-444.

2. Deville, J.C. & Särndal, C.E. (1992). Calibration Estimators in Survey Sampling. *Journal of the American Statistical Association*, 87(418), 376-382.

3. Hainmueller, J. (2012). Entropy Balancing for Causal Effects: A Multivariate Reweighting Method to Produce Balanced Samples in Observational Studies. *Political Analysis*, 20(1), 25-46.

4. Ireland, C.T. & Kullback, S. (1968). Contingency Tables with Given Marginals. *Biometrika*, 55(1), 179-188.

5. Särndal, C.E., Swensson, B. & Wretman, J. (1992). *Model Assisted Survey Sampling*. Springer.

### Convergence Theory

6. Sinkhorn, R. (1964). A relationship between arbitrary positive matrices and doubly stochastic matrices. *Annals of Mathematical Statistics*, 35, 876-879.

7. Sinkhorn, R. & Knopp, P. (1967). Concerning nonnegative matrices and doubly stochastic matrices. *Pacific Journal of Mathematics*, 21(2), 343-348.

8. Csiszar, I. (1975). I-Divergence Geometry of Probability Distributions and Minimization Problems. *Annals of Probability*, 3(1), 146-158.

9. Fienberg, S.E. (1970). An Iterative Procedure for Estimation in Contingency Tables. *Annals of Mathematical Statistics*, 41(3), 907-917.

10. Bishop, Y.M.M., Fienberg, S.E. & Holland, P.W. (1975). *Discrete Multivariate Analysis: Theory and Practice*. MIT Press.

### Sparse Optimization

11. Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society B*, 58, 267-288.

12. Candes, E.J., Romberg, J. & Tao, T. (2006). Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information. *IEEE Transactions on Information Theory*, 52(2), 489-509.

13. Donoho, D.L. (2006). Compressed Sensing. *IEEE Transactions on Information Theory*, 52(4), 1289-1306.

14. Candes, E.J., Wakin, M.B. & Boyd, S.P. (2008). Enhancing Sparsity by Reweighted ℓ1 Minimization. *Journal of Fourier Analysis and Applications*, 14(5), 877-905.

15. Boyd, S., Parikh, N., Chu, E., Peleato, B. & Eckstein, J. (2011). Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. *Foundations and Trends in Machine Learning*, 3(1), 1-122.

16. Bertsimas, D., King, A. & Mazumder, R. (2016). Best Subset Selection via a Modern Optimization Lens. *Annals of Statistics*, 44(2), 813-852.

### L0 Regularization

17. Louizos, C., Welling, M. & Kingma, D.P. (2017). Learning Sparse Neural Networks through L0 Regularization. *arXiv:1712.01312*.

### Statistical Matching

18. D'Orazio, M., Di Zio, M. & Scanu, M. (2006). *Statistical Matching: Theory and Practice*. Wiley.

19. Rässler, S. (2002). *Statistical Matching: A Frequentist Theory, Practical Applications, and Alternative Bayesian Approaches*. Springer.

20. Moriarity, C. & Scheuren, F. (2001). Statistical Matching: A Paradigm for Assessing the Uncertainty in the Procedure. *Journal of Official Statistics*, 17, 407-422.

### Small Area Estimation

21. Fay, R.E. & Herriot, R.A. (1979). Estimates of Income for Small Places: An Application of James-Stein Procedures to Census Data. *Journal of the American Statistical Association*, 74, 269-277.

22. Molina, I., Nandram, B. & Rao, J.N.K. (2014). Small area estimation of general parameters with application to poverty indicators: A hierarchical Bayes approach. *Annals of Applied Statistics*, 8(2), 852-885.

23. Edwards, K.L. & Tanton, R. (2016). Estimating Uncertainty in Spatial Microsimulation Approaches to Small Area Estimation. *Computers, Environment and Urban Systems*.

### Imputation

24. Andridge, R.R. & Little, R.J.A. (2010). A Review of Hot Deck Imputation for Survey Non-response. *International Statistical Review*, 78(1), 40-64.

25. Rubin, D.B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.
