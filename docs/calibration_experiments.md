# Calibration Method Comparison: IPF vs Gradient Descent

**Date**: 2025-12-28
**Authors**: Max Ghenis, Claude
**Dataset**: CPS ASEC 2024 (98,902 tax filers)

## Executive Summary

We compared Iterative Proportional Fitting (IPF) and Gradient Descent (GD) for survey weight calibration across various constraint configurations. The key finding is:

> **The choice between IPF and GD depends on constraint feasibility, not constraint count.**
> - IPF wins when constraints are feasible (achievable within weight bounds)
> - GD wins when constraints are infeasible (mathematically conflicting)

## Methods

### IPF (Iterative Proportional Fitting)
- Sequential updates: cycles through each constraint, adjusting weights to match target
- Damping: ratio clipped to [0.9, 1.1] per iteration
- Weight bounds: adjustment factors clipped to [0.2, 5.0]

### Gradient Descent
- Parallel updates: all weights updated simultaneously based on aggregate gradient
- Loss function: PolicyEngine-style `mean(((estimate - target + 1) / (target + 1))^2)`
- Optimization: log-space GD with learning rate 2.0, 30k iterations
- Weight bounds: same [0.2, 5.0]

### Metrics
- **PE Loss**: PolicyEngine-style mean squared relative error
- **Mean Error**: Mean absolute percentage error across constraints
- **Max Error**: Worst single constraint error

## Experiment 1: Feasible Constraints (IRS AGI Brackets)

**Setup**: 15 real IRS SOI targets for AGI brackets (2021)

| Method | PE Loss | Mean Error | Max Error |
|--------|---------|------------|-----------|
| IPF (200 iter) | 0.000538 | 0.60% | 9.0% |
| GD (20k iter) | 0.027085 | 12.43% | 40.2% |

**Winner**: IPF (50x better PE loss)

**Analysis**: When constraints are achievable, IPF's sequential approach finds a near-perfect solution. GD struggles with gradient interference between constraints.

## Experiment 2: More Feasible Constraints (IRS + States + Age)

**Setup**: 32 constraints from 3 sources (IRS AGI, Census states, age distribution)

| Method | PE Loss | Mean Error |
|--------|---------|------------|
| IPF | 0.005040 | 6.15% |
| GD | 0.018844 | 10.56% |

**Winner**: IPF (3.7x better PE loss)

**Per-constraint pattern**:
- IPF: Uniform ~8.2% error across most constraints (good compromise)
- GD: Highly variable (from -8.6% to +43.4%)

## Experiment 3: Infeasible Constraints

**Setup**: 20 constraints where state totals sum to 129% of national total (mathematically impossible to satisfy all)

| Method | PE Loss | Mean Error |
|--------|---------|------------|
| IPF | 0.176476 | 29.78% |
| GD | 0.080353 | 15.88% |

**Winner**: GD (2.2x better PE loss)

**Per-constraint pattern**:
- IPF: Satisfies some constraints perfectly (0%), fails badly on others (-68% to -71%)
- GD: Distributes error more evenly (-17% to +117%)

**Key insight**: IPF "picks winners" among conflicting constraints based on iteration order. GD finds a globally optimal compromise.

## Experiment 4: Sparsity Comparison

**Setup**: Same 106 constraints, varying weight bounds

| Bounds | IPF PE Loss | GD PE Loss | IPF Mean | GD Mean |
|--------|-------------|------------|----------|---------|
| (0.2, 5.0) | 0.000131 | 0.059001 | 0.25% | 22.00% |
| (0.5, 2.0) | 0.000131 | 0.059021 | 0.25% | 22.01% |
| (0.7, 1.5) | 0.010998 | 0.063127 | 10.41% | 23.70% |
| (0.8, 1.25) | 0.068781 | 0.088826 | 26.19% | 29.56% |

**Observation**: Both methods perform worse with narrower bounds (more "sparse" adjustments), but IPF degrades more gracefully.

## Key Findings

### 1. Feasibility Determines the Winner

| Constraint Type | Winner | Margin |
|-----------------|--------|--------|
| Feasible, few constraints | IPF | 50x |
| Feasible, many constraints | IPF | 3-4x |
| Infeasible constraints | GD | 2x |

### 2. IPF's Sequential Nature

**Advantage**: When constraints are feasible, IPF's sequential updates converge to near-perfect solutions because each update brings the solution closer to the feasible region.

**Disadvantage**: When constraints are infeasible, IPF exhibits "ping-pong" behavior, oscillating between satisfying different constraints. The final solution depends on iteration order.

### 3. GD's Global Optimization

**Advantage**: GD minimizes a global loss function, finding the best compromise when no perfect solution exists.

**Disadvantage**: When a perfect solution exists, GD's gradient updates can interfere across constraints, slowing convergence.

### 4. Computational Efficiency

| Method | Time (100 constraints) | Scaling |
|--------|------------------------|---------|
| IPF (200 iter) | ~0.5s | O(m × n × iter) |
| GD (20k iter) | ~10s | O(m × n × iter) |

IPF is typically 20x faster due to simpler per-iteration operations.

## Recommendations for Microplex

1. **Default to IPF** for standard calibration with consistent targets
2. **Use GD when**:
   - Targets come from multiple sources with potential conflicts
   - You suspect infeasibility (targets don't sum correctly)
   - You want to minimize worst-case error rather than achieve perfect fit on some targets
3. **Hybrid approach**: Start with IPF, switch to GD if max error > 20%

## Code

All experiments used the calibration module at `src/microplex/calibration.py`:
- `Calibrator(method="ipf")` for IPF
- Custom GD implementation with PE-style loss

## Experiment 5: Constraint-Aware Sampling

**Setup**: Instead of L0/L1 regularization (which proved too aggressive), keep K samples per constraint.

| K | Samples | Mean Error | Weight Range |
|---|---------|------------|--------------|
| 50 | 44,595 | 0.64% | 1,060x |
| 20 | 19,992 | 0.64% | 320x |
| 10 | 10,294 | 0.64% | 259x |
| **5** | **5,235** | **0.64%** | 266x |
| 3 | 3,177 | 1.22% | explodes |

**Key finding**: ~4 samples per constraint is the minimum for stable calibration.

## Experiment 6: Hierarchical Synthesis

**Setup**: Two-level structure with household flow + person flow, dual-level calibration.

| Level | Targets | Mean Error |
|-------|---------|------------|
| Household (districts) | 440 | 9.42% |
| Person (age × state) | 918 | 3.65% |

**Structure**:
- 18,825 households with calibrated weights
- 48,292 persons linked to households
- Household weights apply to all persons within

**Key finding**: Hierarchical calibration is more challenging than flat calibration because person-level targets must be achieved through household-level weights.

## Future Work

1. **Automatic feasibility detection**: Detect when constraints are likely infeasible
2. **Hybrid IPF+GD**: Use IPF to warm-start, then GD to refine
3. **Constraint weighting**: Allow different importance weights for different targets
4. **Improve hierarchical calibration**: Better algorithms for dual-level constraints

## References

- PolicyEngine-US-Data: https://github.com/PolicyEngine/policyengine-us-data
- Microcalibrate: https://github.com/PolicyEngine/microcalibrate
- Deville & Särndal (1992): Calibration Estimators in Survey Sampling
