---
kernelspec:
  name: python3
  display_name: Python 3
---

# Microplex: Multi-source synthetic microdata via zero-inflated conditional models

**Max Ghenis**

max@cosilico.ai | Cosilico

```{code-cell} python
:tags: [remove-cell]

from paper_results import r
```

## Abstract

Government surveys observe different slices of the same population: the CPS captures employment and income, the SIPP tracks program participation, and the PSID follows families longitudinally. No single survey observes all variables for all people. We present microplex, a framework for learning joint distributions from these partial views and generating synthetic populations with complete variable coverage. We compare six synthesis methods — quantile regression forests (QRF), quantile deep neural networks (QDNN), and masked autoregressive flows (MAF), each with and without zero-inflation handling — using PRDC metrics {cite:p}`naeem2020reliable` evaluated against holdouts from each source survey. Zero-inflated MAF achieves the highest mean coverage ({eval}`r.zi_maf.coverage_pct`), but the key finding is architectural: zero-inflation handling lifts MAF coverage by {eval}`r.zi_maf_vs_maf_lift` and QDNN by {eval}`r.zi_qdnn_vs_qdnn_lift`, while barely affecting QRF ({eval}`r.zi_qrf_vs_qrf_lift`). We then calibrate synthetic populations against administrative targets (IRS income statistics, benefit program totals, Census geography) using sparse optimization, and evaluate downstream reweighting loss. Code and data are available at [github.com/CosilicoAI/microplex](https://github.com/CosilicoAI/microplex).

## Introduction

### The multi-source microdata problem

Policy microsimulation requires detailed individual records spanning demographics, income, taxes, transfers, wealth, and health. No single survey covers all domains. The Current Population Survey (CPS) Annual Social and Economic Supplement {cite:p}`flood2020integrated` captures {eval}`f"{r.n_cps:,}"` persons with employment and income variables. The Survey of Income and Program Participation (SIPP) adds program participation detail for {eval}`f"{r.n_sipp:,}"` persons. The Panel Study of Income Dynamics (PSID) provides longitudinal structure for {eval}`f"{r.n_psid:,}"` persons. Administrative sources (IRS Statistics of Income, SSA earnings records) cover entire populations but with narrower variable sets.

Current approaches to combining these sources — sequential imputation, statistical matching, or record linkage — suffer from well-documented limitations. Sequential chaining (e.g., imputing CPS variables onto ACS, then PUF variables onto CPS) loses joint distributional structure at each step {cite:p}`meinfelder2011simulation`. Statistical matching preserves marginals but distorts correlations {cite:p}`little1993statistical`. Record linkage requires common identifiers rarely available across surveys.

### Contribution

We make three contributions:

1. **Multi-source synthesis framework.** We formalize the problem of learning $P(\text{all variables})$ from surveys that each observe different subsets of variables, where shared demographic variables provide the conditioning bridge between sources.

2. **Zero-inflation as architectural choice.** We show that zero-inflation handling — a two-stage model that separately predicts whether a variable is zero vs. its positive-value distribution — is the single most impactful design decision for survey microdata synthesis, more important than the choice of base model (forest vs. neural network vs. normalizing flow).

3. **Cross-source holdout evaluation.** We evaluate synthetic data quality using PRDC metrics {cite:p}`naeem2020reliable` computed against holdouts from each source survey separately, providing a multi-dimensional view of distributional fidelity that goes beyond single-survey benchmarks.

## Methods

### Problem formulation

Let $\mathcal{S} = \{S_1, \ldots, S_K\}$ be $K$ surveys, each observing a subset of variables $V_k \subset V$ for $n_k$ records drawn from the same population. A set of shared variables $V_{\text{shared}} = \bigcap_k V_k$ appears in all surveys (e.g., age, sex). For each non-shared variable $v \in V_k \setminus V_{\text{shared}}$, we learn $P(v \mid V_{\text{shared}})$ from survey $S_k$.

To generate synthetic records, we:
1. Sample shared variables from the pooled empirical distribution
2. For each non-shared variable, sample from its learned conditional distribution
3. Calibrate weights against administrative targets

### Zero-inflation handling

Economic variables exhibit mass-at-zero: many people have zero values for income sources, benefit receipts, or tax credits. For a variable $y$ with zero fraction $\pi_0 \geq \theta$ (we use $\theta = 0.1$), the zero-inflated model decomposes generation into:

$$
y \sim \begin{cases} 0 & \text{with probability } \hat{\pi}_0(x) \\ g(x) & \text{with probability } 1 - \hat{\pi}_0(x) \end{cases}
$$

where $\hat{\pi}_0(x)$ is a random forest classifier predicting zero vs. non-zero, and $g(x)$ is the base model (QRF, QDNN, or MAF) trained only on positive values {cite:p}`lambert1992zero`.

### Base models

We compare three model families, each with and without zero-inflation:

**Quantile regression forest (QRF).** Following {cite:t}`meinshausen2006quantile`, we fit a random forest that learns the full conditional distribution $P(y \mid x)$ by retaining quantile information from training observations in leaf nodes. At generation time, we sample a random quantile $\tau \sim \text{Uniform}(0.1, 0.9)$ and return the corresponding predicted quantile.

**Quantile deep neural network (QDNN).** A multi-layer perceptron trained with pinball loss {cite:p}`koenker2001quantile` to predict quantiles $\hat{q}_\tau(x)$ for $\tau \in \{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95\}$. At generation time, we sample a random quantile index and return the corresponding prediction.

**Masked autoregressive flow (MAF).** A normalizing flow {cite:p}`papamakarios2017masked` that learns the full conditional density $p(y \mid x)$ via invertible transformations. We apply log transformation to positive values before standardization and train with maximum likelihood.

### Evaluation: PRDC metrics

We evaluate using Precision, Recall, Density, and Coverage (PRDC) {cite:p}`naeem2020reliable`, originally developed for evaluating generative image models and adapted here for tabular survey data.

For each source survey $S_k$:
1. Split into 80% train / 20% holdout
2. Train all methods on the training portions
3. Generate synthetic records
4. Compute PRDC on columns present in $S_k$

**Coverage** — our primary metric — measures the fraction of real holdout points that have at least one synthetic neighbor within their $k$-th nearest-neighbor radius (we use $k={eval}`r.k`$). The radius is adaptive and per-point: dense regions of the real manifold have smaller radii, sparse regions larger. All distances are computed in standardized Euclidean space.

### Calibration via sparse reweighting

After synthesis, we calibrate the microdata against administrative targets using constrained optimization:

$$
\min_w \|w\|_p \quad \text{subject to} \quad Aw = b
$$

where $w$ is a weight vector, $A$ is a constraint matrix encoding target definitions (geographic populations, income aggregates, benefit totals), and $b$ is the target vector from official sources (Census, IRS SOI, CBO). We solve with $L_1$ or $L_2$ penalties using scipy {cite:p}`deville1992calibration`.

## Data

We use three public-use surveys stacked into a common format ({eval}`f"{r.n_total:,}"` total records):

| Source | Records | Variables | Domain |
|--------|--------:|----------:|--------|
| SIPP | {eval}`f"{r.n_sipp:,}"` | 9 | Program participation, income |
| CPS ASEC | {eval}`f"{r.n_cps:,}"` | 10 | Employment, income, demographics |
| PSID | {eval}`f"{r.n_psid:,}"` | 15 | Longitudinal income, wealth |

Shared conditioning variables across all sources: age and sex. Source-specific variables include income components, benefit receipts, employment status, and household characteristics.

## Results

### Synthesis: method comparison

```{code-cell} python
:tags: [remove-input]

import json
import pandas as pd
from pathlib import Path

with open(Path("..") / "benchmarks" / "results" / "benchmark_full.json") as f:
    data = json.load(f)

rows = []
for name, m in data["methods"].items():
    rows.append({
        "Method": name,
        "Coverage": f"{m['mean_coverage']:.1%}",
        "Precision": f"{m['mean_precision']:.1%}",
        "Density": f"{m['mean_density']:.2f}",
        "Time (s)": f"{m['elapsed_seconds']:.0f}",
    })
df = pd.DataFrame(rows).sort_values("Coverage", ascending=False)
df.index = range(1, len(df) + 1)
df
```

{eval}`r.best_method` achieves the highest mean coverage at {eval}`r.best_coverage_pct`, followed by ZI-QRF at {eval}`r.zi_qrf.coverage_pct`. The ranking reveals a clear pattern: zero-inflation handling provides substantial gains for neural methods but minimal benefit for tree-based methods.

### The zero-inflation effect

The most striking result is the differential impact of zero-inflation across model families:

| Base model | Without ZI | With ZI | Lift |
|-----------|-----------|---------|------|
| MAF | {eval}`r.maf.coverage_pct` | {eval}`r.zi_maf.coverage_pct` | +{eval}`r.zi_maf_vs_maf_lift` |
| QDNN | {eval}`r.qdnn.coverage_pct` | {eval}`r.zi_qdnn.coverage_pct` | +{eval}`r.zi_qdnn_vs_qdnn_lift` |
| QRF | {eval}`r.qrf.coverage_pct` | {eval}`r.zi_qrf.coverage_pct` | +{eval}`r.zi_qrf_vs_qrf_lift` |

MAF without zero-inflation ({eval}`r.maf.coverage_pct` coverage) performs worse than even plain QRF ({eval}`r.qrf.coverage_pct`). Adding zero-inflation lifts MAF to the top of the ranking ({eval}`r.zi_maf.coverage_pct`). This suggests that normalizing flows struggle to model the discontinuity between zero and positive values as a single distribution, but excel at modeling the smooth positive-value conditional once freed from this burden.

QRF is naturally robust to zero-inflation because quantile forests can represent mixed distributions — leaf nodes containing both zero and positive training observations produce quantile predictions that implicitly capture the zero mass.

### Per-source coverage

```{code-cell} python
:tags: [remove-input]

rows = []
for name, m in data["methods"].items():
    for s in m["sources"]:
        rows.append({
            "Method": name,
            "Source": s["source"].upper(),
            "Coverage": s["coverage"],
            "Precision": s["precision"],
        })
df = pd.DataFrame(rows)
pivot = df.pivot_table(index="Method", columns="Source", values="Coverage")
pivot = pivot.sort_values("CPS", ascending=False)
pivot.style.format("{:.1%}").background_gradient(cmap="RdYlGn", vmin=0, vmax=1)
```

SIPP coverage is consistently high across methods (87-95% for ZI variants), CPS moderate (33-46%), and PSID near zero. The PSID result reflects a fundamental limitation: with only 2 shared conditioning variables (age, sex) and 15 PSID-specific columns, the model cannot learn the 15-dimensional joint structure from demographics alone. This motivates expanding the shared variable set — our P1 variable expansion adds tax filing structure, employment, and disability indicators, which should substantially improve cross-source coverage.

### Speed-accuracy tradeoff

ZI-QRF offers a compelling practical tradeoff: {eval}`r.zi_qrf.coverage_pct` coverage in {eval}`r.zi_qrf.time_str`, compared to ZI-MAF's {eval}`r.zi_maf.coverage_pct` in {eval}`r.zi_maf.time_str` — {eval}`r.zi_speedup_over_maf` slower for 1.6 percentage points of additional coverage. For production pipelines requiring frequent regeneration, ZI-QRF may be preferred despite slightly lower coverage.

### Reweighting calibration

After synthesis, we calibrate weights against administrative targets spanning income (IRS SOI), benefits (CBO program totals), and geography (Census state populations). Evaluation compares weighted aggregates from the calibrated microdata against official target values, reporting relative error as a percentage.

Calibration targets include state-level population counts, total income by source (wages, dividends, Social Security), benefit program spending (SNAP, SSI, TANF), and tax aggregates. The reweighting engine uses $L_1$-penalized optimization to find sparse weight adjustments that minimize distance to all targets simultaneously.

## Discussion

### Zero-inflation is the key architectural choice

Our central finding is that zero-inflation handling matters more than model selection. A simple two-stage decomposition — classifier for zero vs. non-zero, followed by a conditional model on positive values only — transforms underperforming neural methods into competitive or best-performing ones. This has practical implications: researchers choosing a synthesis method should first implement zero-inflation handling, then select a base model based on speed and accuracy requirements.

The mechanism is clear for economic survey variables. Income sources (wages, dividends, transfers) are zero for large population fractions. Without ZI, a normalizing flow or neural network must simultaneously model: (a) the probability of being a recipient, and (b) the distribution of amounts conditional on receipt. These are fundamentally different modeling tasks — one is a classification boundary, the other a continuous density estimation — and conflating them degrades both.

### Limitations

**Shared variable bottleneck.** With only age and sex as shared conditioning variables, the model cannot capture the rich covariance structure that depends on education, occupation, geography, and other demographics. The 0% PSID coverage demonstrates this limitation. Expanding shared variables is the highest-priority improvement.

**Independence assumption.** Non-shared variables are generated independently conditional on shared variables. This preserves the marginal $P(v \mid V_{\text{shared}})$ for each variable but may distort cross-variable correlations (e.g., the correlation between SIPP-specific and CPS-specific variables). A full joint model over all variables — either via the unified latent space approach or conditional dependency chains — would address this.

**Household structure.** Current synthesis operates at the person level. Realistic microdata requires consistent household structure: spouses should have compatible incomes, dependents should be children, tax unit filing status should match household composition. We plan to add hierarchical synthesis and relationship pointers (spouse_person_id, parent_person_id) in future work.

**Deep generative baselines.** We exclude CTGAN {cite:p}`xu2019modeling` and TVAE {cite:p}`xu2019tvae` from the current benchmark due to dependency constraints. Adding these baselines, along with recent diffusion-based methods like Forest Flow {cite:p}`jolicoeurmartineau2024generating`, would strengthen the comparison.

### Future work

1. **Expanded shared variables** — add employment, education, filing status, disability status as shared conditioning features
2. **Hierarchical synthesis** — top-down generation preserving household → tax unit → person structure
3. **Relationship pointers** — spouse_person_id, parent_person_id for intra-household consistency
4. **Custom tax unit formation** — replacing Census-assigned units with improved methodology {cite:p}`gale2022simulating`
5. **Longitudinal extension** — leveraging PSID panel structure for synthetic lifecycle trajectories

## References

```{bibliography}
:style: unsrt
```
