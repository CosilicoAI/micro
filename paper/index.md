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

Government surveys observe different slices of the same population: the Current Population Survey (CPS) captures employment and income, the Survey of Income and Program Participation (SIPP) tracks program participation, and the Panel Study of Income Dynamics (PSID) follows families longitudinally. No single survey observes all variables for all people. We present microplex, a framework for learning per-variable conditional distributions $P(v \mid V_{\text{shared}})$ from multiple surveys and generating synthetic records with complete variable coverage. Because each variable is modeled conditionally on shared demographics (age, sex), the resulting synthetic data preserves within-source marginals but does not learn cross-source correlations — a limitation we discuss. We compare six synthesis methods — quantile regression forests (QRF), quantile deep neural networks (QDNN), and masked autoregressive flows (MAF), each with and without zero-inflation (ZI) handling — using Precision, Recall, Density, and Coverage (PRDC) metrics {cite:p}`naeem2020reliable` evaluated against holdouts from each source survey. ZI-MAF achieves the highest SIPP coverage ({eval}`r.zi_maf.sipp_pct`) and CPS coverage ({eval}`r.zi_maf.cps_pct`), but the key finding is architectural: zero-inflation handling lifts MAF coverage by {eval}`r.zi_maf_vs_maf_lift` and QDNN by {eval}`r.zi_qdnn_vs_qdnn_lift`, while barely affecting QRF ({eval}`r.zi_qrf_vs_qrf_lift`). We also compare five calibration methods for reweighting synthetic populations, finding that entropy balancing achieves the lowest mean relative error ({eval}`r.rw_entropy.mean_error_pct`). Code is available at [github.com/CosilicoAI/microplex](https://github.com/CosilicoAI/microplex).

## Introduction

### The multi-source microdata problem

Policy microsimulation requires detailed individual records spanning demographics, income, taxes, transfers, wealth, and health. No single survey covers all domains. The Current Population Survey (CPS) Annual Social and Economic Supplement {cite:p}`flood2020integrated` captures {eval}`f"{r.n_cps:,}"` persons with employment and income variables. The Survey of Income and Program Participation (SIPP) adds program participation detail for {eval}`f"{r.n_sipp:,}"` persons. The Panel Study of Income Dynamics (PSID) provides longitudinal structure for {eval}`f"{r.n_psid:,}"` persons. Administrative sources (IRS Statistics of Income, SSA earnings records) cover entire populations but with narrower variable sets.

Current approaches to combining these sources — sequential imputation, statistical matching, or record linkage — suffer from well-documented limitations. Synthetic data approaches {cite:p}`rubin1993statistical,drechsler2011synthetic` and multiple imputation {cite:p}`raghunathan2003multiple` address disclosure concerns but typically operate on single surveys. Sequential chaining (e.g., imputing CPS variables onto ACS, then PUF variables onto CPS) loses joint distributional structure at each step {cite:p}`meinfelder2011simulation`. Statistical matching preserves marginals but distorts correlations {cite:p}`dorazio2006statistical`. Record linkage requires common identifiers rarely available across surveys.

### Contribution

We make three contributions:

1. **Multi-source conditional synthesis framework.** We formalize the problem of learning per-variable conditionals $P(v \mid V_{\text{shared}})$ from surveys that each observe different subsets of variables. This approach generates records with complete variable coverage, though it assumes conditional independence across sources given shared variables — a strong assumption whose implications we evaluate.

2. **Zero-inflation as architectural choice.** We show that zero-inflation handling — a two-stage model that separately predicts whether a variable is zero vs. its positive-value distribution — provides large coverage gains for neural methods (MAF: +{eval}`r.zi_maf_vs_maf_lift`; QDNN: +{eval}`r.zi_qdnn_vs_qdnn_lift`) while barely affecting tree-based methods (QRF: +{eval}`r.zi_qrf_vs_qrf_lift`), suggesting it is more impactful than the choice of base model for economic survey data with mass-at-zero variables.

3. **Cross-source holdout evaluation.** We evaluate synthetic data quality using PRDC metrics {cite:p}`naeem2020reliable` computed against holdouts from each source survey separately, revealing that coverage varies dramatically across sources (SIPP: {eval}`r.zi_maf.sipp_pct`; CPS: {eval}`r.zi_maf.cps_pct`; PSID: {eval}`r.zi_maf.psid_pct` for ZI-MAF) — a pattern obscured by aggregate metrics.

## Methods

### Problem formulation

Let $\mathcal{S} = \{S_1, \ldots, S_K\}$ be $K$ surveys, each observing a subset of variables $V_k \subset V$ for $n_k$ records drawn from the same population. A set of shared variables $V_{\text{shared}} = \bigcap_k V_k$ appears in all surveys (e.g., age, sex). For each non-shared variable $v \in V_k \setminus V_{\text{shared}}$, we learn $P(v \mid V_{\text{shared}})$ from survey $S_k$.

This factorization implies a conditional independence assumption: non-shared variables from different sources are independent given $V_{\text{shared}}$. The resulting synthetic joint distribution is $P(V) = P(V_{\text{shared}}) \prod_{k} \prod_{v \in V_k \setminus V_{\text{shared}}} P(v \mid V_{\text{shared}})$. This preserves within-source marginal conditionals but does not capture cross-source correlations (e.g., between SIPP program participation and CPS income components). The quality of this approximation depends on the richness of the shared variable set — with only demographic variables, it is coarse; with employment, education, and filing status added, it would improve substantially.

To generate synthetic records, we:
1. Sample shared variables from the pooled empirical distribution (with small Gaussian perturbation, $\sigma=0.1$, to smooth the discrete sample)
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

### Calibration via reweighting

After synthesis, we calibrate the microdata against administrative targets by adjusting record weights. We compare five methods spanning two families:

**Calibration methods** solve for weights that match both categorical marginals and continuous targets simultaneously:

- **Iterative proportional fitting (IPF)** {cite:p}`deming1940least`. The classical raking algorithm that alternately adjusts weights to match each marginal target. Converges when all marginal constraints are satisfied simultaneously.
- **Entropy balancing** {cite:p}`hainmueller2012entropy`. Minimizes the Kullback-Leibler divergence from the original weights subject to target constraints: $\min_w \sum_i w_i \log(w_i / w_i^0)$ s.t. $Aw = b$.
- **SparseCalibrator** {cite:p}`deville1992calibration`. Selects a sparse subset of records via cross-category proportional sampling, then calibrates the selected subset using iterative proportional fitting to match both categorical and continuous targets.

**Sparse optimization methods** minimize the weight norm subject to categorical constraints only:

- **$L_1$-sparse** and **$L_0$-sparse**. Solve $\min_w \|w\|_p$ s.t. $Aw = b$ for subset selection rather than population calibration.

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
    source_map = {s["source"]: s for s in m["sources"]}
    rows.append({
        "Method": name,
        "SIPP cov.": f"{source_map.get('sipp', {}).get('coverage', 0):.1%}",
        "CPS cov.": f"{source_map.get('cps', {}).get('coverage', 0):.1%}",
        "PSID cov.": f"{source_map.get('psid', {}).get('coverage', 0):.1%}",
        "Precision": f"{m['mean_precision']:.1%}",
        "Density": f"{m['mean_density']:.2f}",
        "Time (s)": f"{m['elapsed_seconds']:.0f}",
    })
df = pd.DataFrame(rows).sort_values("SIPP cov.", ascending=False)
df.index = range(1, len(df) + 1)
df
```

Per-source coverage varies dramatically across surveys. SIPP coverage is high across methods ({eval}`r.zi_maf.sipp_pct` for ZI-MAF), CPS moderate ({eval}`r.zi_maf.cps_pct`), and PSID coverage is 0% for all methods. The PSID result reflects a fundamental limitation of the current shared variable set: with only 2 conditioning variables (age, sex) and 15 PSID-specific columns, the model cannot learn the 15-dimensional joint structure from demographics alone.

We report per-source results as the primary metrics rather than aggregating across sources, since averaging with a degenerate 0% source obscures the pattern. The current results are from a single train/test split with seed=42; multi-seed evaluation with confidence intervals is needed to establish statistical significance of the differences between methods.

### The zero-inflation effect

The differential impact of zero-inflation across model families is the most consistent pattern in the results:

| Base model | Without ZI | With ZI | Lift |
|-----------|-----------|---------|------|
| MAF | {eval}`r.maf.coverage_pct` | {eval}`r.zi_maf.coverage_pct` | +{eval}`r.zi_maf_vs_maf_lift` |
| QDNN | {eval}`r.qdnn.coverage_pct` | {eval}`r.zi_qdnn.coverage_pct` | +{eval}`r.zi_qdnn_vs_qdnn_lift` |
| QRF | {eval}`r.qrf.coverage_pct` | {eval}`r.zi_qrf.coverage_pct` | +{eval}`r.zi_qrf_vs_qrf_lift` |

MAF without zero-inflation ({eval}`r.maf.coverage_pct` mean coverage) performs worse than plain QRF ({eval}`r.qrf.coverage_pct`). Adding zero-inflation lifts MAF to the top of the ranking ({eval}`r.zi_maf.coverage_pct`). This is consistent with the hypothesis that normalizing flows cannot jointly model the zero mass and positive-value density as a single distribution, but perform well on the smooth positive-value conditional once freed from this burden.

QRF is naturally robust to zero-inflation because quantile forests can represent mixed distributions — leaf nodes containing both zero and positive training observations produce quantile predictions that implicitly capture the zero mass. The minimal ZI lift for QRF (+{eval}`r.zi_qrf_vs_qrf_lift`) may simply reflect that forests already handle this case, rather than indicating that zero-inflation is unimportant for tree methods.

### Speed-accuracy tradeoff

ZI-QRF completes in {eval}`r.zi_qrf.time_str`, compared to ZI-MAF's {eval}`r.zi_maf.time_str` ({eval}`r.zi_speedup_over_maf` slower). The SIPP coverage difference ({eval}`r.zi_qrf.sipp_pct` vs. {eval}`r.zi_maf.sipp_pct`) is modest, though the difference on CPS ({eval}`r.zi_qrf.cps_pct` vs. {eval}`r.zi_maf.cps_pct`) is larger. For production pipelines requiring frequent regeneration, ZI-QRF may be preferred given its speed advantage, though multi-seed evaluation is needed to confirm whether these differences are statistically reliable.

### Reweighting calibration

We evaluate reweighting methods on {eval}`f"{r.rw_n_records:,}"` records with {eval}`r.rw_n_targets_total` calibration targets ({eval}`r.rw_n_marginal_targets` categorical marginals spanning age group and sex, plus {eval}`r.rw_n_continuous_targets` continuous target for total population weight). Target values are perturbed from the sample distribution by 10-30% to simulate calibration to known population totals.

```{code-cell} python
:tags: [remove-input]

import json
import pandas as pd
from pathlib import Path

with open(Path("..") / "benchmarks" / "results" / "reweighting_full.json") as f:
    rw_data = json.load(f)

rows = []
for name, m in rw_data["methods"].items():
    rows.append({
        "Method": name,
        "Mean rel. error": f"{m['mean_relative_error']:.1%}",
        "Max rel. error": f"{m['max_relative_error']:.1%}",
        "Weight CV": f"{m['weight_cv']:.3f}",
        "Time (s)": f"{m['elapsed_seconds']:.2f}",
    })
df = pd.DataFrame(rows)
# Show calibration methods first (sorted by mean error), then sparse
cal_methods = df[df["Method"].isin(["Entropy", "IPF", "SparseCalibrator"])]
sparse_methods = df[~df["Method"].isin(["Entropy", "IPF", "SparseCalibrator"])]
df = pd.concat([cal_methods.sort_values("Mean rel. error"), sparse_methods])
df.index = range(1, len(df) + 1)
df
```

Among calibration methods, entropy balancing achieves the lowest mean relative error ({eval}`r.rw_entropy.mean_error_pct`), {eval}`r.entropy_vs_ipf_error_reduction` lower than IPF ({eval}`r.rw_ipf.mean_error_pct`). SparseCalibrator matches IPF accuracy while producing {eval}`r.sparse_cal_cv_vs_ipf` lower weight coefficient of variation ({eval}`r.rw_sparse_cal.cv_str` vs. {eval}`r.rw_ipf.cv_str`), meaning smoother weights that are less likely to amplify noise in downstream estimates.

The $L_1$- and $L_0$-sparse methods show high errors ({eval}`r.rw_l1.mean_error_pct`) because they optimize for subset selection (minimizing $\|w\|_p$) rather than population calibration. They satisfy categorical constraints but cannot match continuous targets, making them unsuitable for general-purpose calibration despite their sparsity advantages.

The tradeoff between entropy and SparseCalibrator is instructive: entropy achieves lower mean error but higher max error ({eval}`r.rw_entropy.max_error_pct` vs. {eval}`r.rw_sparse_cal.max_error_pct`), while SparseCalibrator provides more uniform error across targets with smoother weights. For production microsimulation where extreme weights distort variance estimates, SparseCalibrator's lower CV may be preferred despite slightly higher mean error.

## Discussion

### Zero-inflation as architectural choice

The most consistent finding is that zero-inflation handling provides large coverage gains for neural methods while barely affecting tree-based methods. A two-stage decomposition — random forest classifier {cite:p}`breiman2001random` for zero vs. non-zero, followed by a conditional model on positive values only — transforms underperforming MAF and QDNN methods into the top performers. This has practical implications: researchers choosing a synthesis method for economic survey data should implement zero-inflation handling before optimizing the base model.

The mechanism follows from the structure of economic survey variables. Income sources (wages, dividends, transfers) are zero for large population fractions. Without ZI, a normalizing flow or neural network must simultaneously model: (a) the probability of being a recipient, and (b) the distribution of amounts conditional on receipt. These are fundamentally different modeling tasks — one is a classification boundary, the other a continuous density estimation — and conflating them degrades both. Tree-based methods handle this naturally through leaf node composition.

### Limitations

**Shared variable bottleneck.** With only age and sex as shared conditioning variables, the model cannot capture the covariance structure that depends on education, occupation, geography, and other demographics. The 0% PSID coverage across all methods demonstrates this limitation. Expanding shared variables is the highest-priority improvement.

**Conditional independence assumption.** Non-shared variables are generated independently conditional on shared variables. The synthetic joint distribution is $\prod_v P(v \mid V_{\text{shared}})$, which preserves each marginal conditional but destroys cross-source correlations. For microsimulation applications, the correlation between (e.g.) SIPP program participation and CPS income components is precisely what is needed. The current framework does not capture these relationships, and we do not evaluate cross-source correlation fidelity. A full joint model — via a unified latent space, conditional dependency chains, or copula-based approaches {cite:p}`dorazio2006statistical` — would address this at the cost of additional complexity.

**Single evaluation run.** The reported results are from a single train/test split with a single random seed. The differences between the top methods (1-2 percentage points of mean coverage) may be within sampling variability. Multi-seed evaluation with confidence intervals is needed to establish statistical robustness of the rankings.

**Survey weights not used in training.** The benchmark treats all survey records equally, ignoring complex sampling designs and survey weights. This biases the learned distributions toward oversampled strata and may not reflect the population distributions that practitioners need.

**Household structure.** Current synthesis operates at the person level. Realistic microdata requires consistent household structure: spouses should have compatible incomes, dependents should be children, tax unit filing status should match household composition. We plan to add hierarchical synthesis and relationship pointers (spouse_person_id, parent_person_id) in future work.

**Deep generative baselines.** We exclude CTGAN {cite:p}`xu2019modeling` and TVAE {cite:p}`xu2019tvae` from the current benchmark due to dependency constraints. Adding these baselines, along with recent diffusion-based methods like Forest Flow {cite:p}`jolicoeurmartineau2024generating`, would strengthen the comparison.

### Future work

The most impactful improvement would be expanding the shared variable set beyond age and sex to include employment status, education, marital status, filing status, and disability indicators. This would strengthen the conditioning bridge between sources and address the 0% PSID coverage. Second, hierarchical synthesis preserving household, tax unit, and person structure {cite:p}`gale2022simulating` would make the synthetic data usable for tax-benefit microsimulation. Third, adding diffusion-based methods (Forest Flow {cite:p}`jolicoeurmartineau2024generating`) and established baselines (CTGAN, TVAE) would strengthen the methodological comparison. Finally, multi-seed evaluation with confidence intervals is needed to establish the statistical reliability of the method rankings.

## Conclusion

We presented microplex, a framework for generating synthetic microdata from multiple government surveys using per-variable conditional models. The central empirical finding is that zero-inflation handling — a two-stage decomposition separating the zero/non-zero classification from the positive-value distribution — provides large coverage gains for neural methods (MAF: +{eval}`r.zi_maf_vs_maf_lift`; QDNN: +{eval}`r.zi_qdnn_vs_qdnn_lift`) while barely affecting tree-based methods (+{eval}`r.zi_qrf_vs_qrf_lift` for QRF). This suggests that practitioners working with economic survey data should implement zero-inflation handling before selecting a base model.

The framework has clear limitations in its current form: the conditional independence assumption and narrow shared variable set (age, sex) mean cross-source correlations are not captured, as demonstrated by the 0% PSID coverage. These limitations are addressable — expanding shared variables and modeling cross-source dependencies are the highest-priority improvements for making the synthetic data usable in production microsimulation.

## References

```{bibliography}
:style: unsrt
```
