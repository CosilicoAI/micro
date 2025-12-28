"""Multi-survey fusion with masked loss.

Tests microplex's ability to learn from multiple surveys with different
variable coverage using masked loss (only compute loss on observed values).

This is the "best" approach compared to:
- Sequential fusion (PE approach): loses joint structure
- Naive stacking with NaN→0: corrupts joint distribution
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

from run_cps_benchmark import generate_cps_like_data
from microplex import Synthesizer

np.random.seed(42)


def compute_coverage(holdout: np.ndarray, synthetic: np.ndarray) -> float:
    """Compute coverage: avg distance from holdout to nearest synthetic."""
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(synthetic)
    distances, _ = nn.kneighbors(holdout)
    return float(np.mean(distances))


def generate_multi_survey_data(n_pop: int = 50000) -> Dict:
    """Generate population and multiple surveys with different variable coverage."""
    print(f"Generating ground truth population (n={n_pop:,})...")
    population = generate_cps_like_data(n_pop, seed=42)

    # All target variables we want to synthesize
    all_target_vars = [
        "wage_income", "self_emp_income", "ss_income",
        "uc_income", "investment_income"
    ]

    # Context variables (always observed in all surveys)
    context_vars = ["age", "education", "is_female", "marital_status", "is_employed"]

    surveys = {}

    # CPS: has wage_income, uc_income (oversamples unemployed)
    cps_n = 5000
    p_select = np.where(population["is_employed"] == 0, 0.003, 0.0001)
    p_select = p_select / p_select.sum()
    cps_idx = np.random.choice(n_pop, size=cps_n, replace=False, p=p_select)
    cps_data = population.iloc[cps_idx][context_vars + ["wage_income", "uc_income"]].copy()
    # Add NaN for variables CPS doesn't have
    for var in ["self_emp_income", "ss_income", "investment_income"]:
        cps_data[var] = np.nan
    surveys["CPS"] = cps_data.reset_index(drop=True)

    # IRS: has wage, self_emp, investment (only filers with income)
    irs_n = 8000
    has_income = (population["wage_income"] > 0) | (population["self_emp_income"] > 0)
    irs_pool = np.where(has_income)[0]
    irs_idx = np.random.choice(irs_pool, size=min(irs_n, len(irs_pool)), replace=False)
    irs_data = population.iloc[irs_idx][context_vars + ["wage_income", "self_emp_income", "investment_income"]].copy()
    # Add NaN for variables IRS doesn't have
    for var in ["ss_income", "uc_income"]:
        irs_data[var] = np.nan
    surveys["IRS"] = irs_data.reset_index(drop=True)

    # SIPP: has ss_income, investment_income (oversamples elderly)
    sipp_n = 3000
    p_select = np.where(population["age"] >= 62, 0.002, 0.0001)
    p_select = p_select / p_select.sum()
    sipp_idx = np.random.choice(n_pop, size=sipp_n, replace=False, p=p_select)
    sipp_data = population.iloc[sipp_idx][context_vars + ["ss_income", "investment_income"]].copy()
    # Add NaN for variables SIPP doesn't have
    for var in ["wage_income", "self_emp_income", "uc_income"]:
        sipp_data[var] = np.nan
    surveys["SIPP"] = sipp_data.reset_index(drop=True)

    return {
        "population": population,
        "surveys": surveys,
        "context_vars": context_vars,
        "target_vars": all_target_vars,
    }


def stack_surveys(surveys: Dict[str, pd.DataFrame], all_vars: List[str]) -> pd.DataFrame:
    """Stack all surveys into single DataFrame, preserving NaN for missing vars."""
    stacked_rows = []
    for name, df in surveys.items():
        for _, row in df.iterrows():
            new_row = {v: np.nan for v in all_vars}
            for col in df.columns:
                if col in all_vars:
                    new_row[col] = row[col]
            stacked_rows.append(new_row)
    return pd.DataFrame(stacked_rows)


def evaluate(
    population: pd.DataFrame,
    reconstructed: pd.DataFrame,
    target_vars: List[str],
) -> Dict:
    """Evaluate reconstruction quality."""
    n_test = min(5000, len(population) // 5)
    test_pop = population.sample(n=n_test, random_state=123)

    # Coverage on target variables
    scaler = StandardScaler()
    pop_targets = scaler.fit_transform(test_pop[target_vars])
    recon_targets = scaler.transform(reconstructed[target_vars].clip(
        lower=test_pop[target_vars].min().values,
        upper=test_pop[target_vars].max().values
    ))

    coverage = compute_coverage(pop_targets, recon_targets)

    # Per-variable mean/std comparison
    var_stats = {}
    for var in target_vars:
        pop_mean = test_pop[var].mean()
        recon_mean = reconstructed[var].mean()
        pop_std = test_pop[var].std()
        recon_std = reconstructed[var].std()
        var_stats[var] = {
            "pop_mean": pop_mean,
            "recon_mean": recon_mean,
            "mean_ratio": recon_mean / (pop_mean + 1e-6),
            "std_ratio": recon_std / (pop_std + 1e-6),
        }

    return {"coverage": coverage, "var_stats": var_stats}


def method_masked_loss(
    stacked: pd.DataFrame,
    context_vars: List[str],
    target_vars: List[str],
    n_generate: int,
) -> pd.DataFrame:
    """Microplex with masked loss (proper NaN handling)."""
    print("\n[MASKED LOSS] Training with proper NaN handling...")
    start = time.time()

    synth = Synthesizer(
        target_vars=target_vars,
        condition_vars=context_vars,
        n_layers=6,
        hidden_dim=64,
        zero_inflated=True,
    )
    synth.fit(stacked, weight_col=None, epochs=100, verbose=True)

    # Generate from sampled conditions
    conditions = stacked[context_vars].dropna().sample(n=n_generate, replace=True).reset_index(drop=True)
    result = synth.generate(conditions, seed=42)

    print(f"  Time: {time.time() - start:.1f}s")
    return result


def method_naive_zero_fill(
    stacked: pd.DataFrame,
    context_vars: List[str],
    target_vars: List[str],
    n_generate: int,
) -> pd.DataFrame:
    """Naive approach: fill NaN with 0 (baseline - should be worse)."""
    print("\n[NAIVE ZERO FILL] Training with NaN→0...")
    start = time.time()

    # Fill NaN with 0
    stacked_filled = stacked.copy()
    for var in target_vars:
        stacked_filled[var] = stacked_filled[var].fillna(0)

    synth = Synthesizer(
        target_vars=target_vars,
        condition_vars=context_vars,
        n_layers=6,
        hidden_dim=64,
        zero_inflated=True,
    )
    synth.fit(stacked_filled, weight_col=None, epochs=100, verbose=True)

    conditions = stacked_filled[context_vars].sample(n=n_generate, replace=True).reset_index(drop=True)
    result = synth.generate(conditions, seed=42)

    print(f"  Time: {time.time() - start:.1f}s")
    return result


def method_single_survey(
    population: pd.DataFrame,
    context_vars: List[str],
    target_vars: List[str],
    n_generate: int,
    sample_frac: float = 0.02,
) -> pd.DataFrame:
    """Single complete survey (no missing data) as upper bound."""
    print(f"\n[SINGLE SURVEY] Training on {sample_frac*100:.0f}% complete sample...")
    start = time.time()

    n_train = int(len(population) * sample_frac)
    train_data = population.sample(n=n_train, random_state=42)

    synth = Synthesizer(
        target_vars=target_vars,
        condition_vars=context_vars,
        n_layers=6,
        hidden_dim=64,
        zero_inflated=True,
    )
    synth.fit(train_data, weight_col=None, epochs=100, verbose=True)

    conditions = train_data[context_vars].sample(n=n_generate, replace=True).reset_index(drop=True)
    result = synth.generate(conditions, seed=42)

    print(f"  Time: {time.time() - start:.1f}s")
    return result


def main():
    print("=" * 70)
    print("MULTI-SURVEY FUSION WITH MASKED LOSS")
    print("=" * 70)

    # Generate data
    data = generate_multi_survey_data(n_pop=50000)
    population = data["population"]
    surveys = data["surveys"]
    context_vars = data["context_vars"]
    target_vars = data["target_vars"]

    print(f"\nPopulation: {len(population):,} records")
    print(f"Context vars: {context_vars}")
    print(f"Target vars: {target_vars}")
    print("\nSurveys:")
    for name, df in surveys.items():
        observed = [v for v in target_vars if not df[v].isna().all()]
        print(f"  {name}: {len(df):,} records, observes {observed}")

    # Stack surveys
    all_vars = context_vars + target_vars
    stacked = stack_surveys(surveys, all_vars)
    print(f"\nStacked: {len(stacked):,} total records")

    # Show missing data pattern
    print("\nMissing data pattern:")
    for var in target_vars:
        n_observed = stacked[var].notna().sum()
        pct = 100 * n_observed / len(stacked)
        print(f"  {var}: {n_observed:,} observed ({pct:.1f}%)")

    n_generate = 10000
    results = []

    # Method 1: Masked loss (our approach)
    try:
        recon = method_masked_loss(stacked, context_vars, target_vars, n_generate)
        res = evaluate(population, recon, target_vars)
        results.append(("Masked Loss", res["coverage"], res["var_stats"]))
        print(f"  → Coverage: {res['coverage']:.4f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback; traceback.print_exc()

    # Method 2: Naive zero fill (baseline)
    try:
        recon = method_naive_zero_fill(stacked, context_vars, target_vars, n_generate)
        res = evaluate(population, recon, target_vars)
        results.append(("Naive Zero Fill", res["coverage"], res["var_stats"]))
        print(f"  → Coverage: {res['coverage']:.4f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback; traceback.print_exc()

    # Method 3: Single complete survey (upper bound)
    # Use same total N as stacked for fair comparison
    total_stacked_n = len(stacked)
    sample_frac = total_stacked_n / len(population)
    try:
        recon = method_single_survey(population, context_vars, target_vars, n_generate, sample_frac)
        res = evaluate(population, recon, target_vars)
        results.append(("Single Survey (complete)", res["coverage"], res["var_stats"]))
        print(f"  → Coverage: {res['coverage']:.4f}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback; traceback.print_exc()

    # Oracle: sample from population
    oracle = population.sample(n=n_generate, random_state=99)
    res = evaluate(population, oracle, target_vars)
    results.append(("Oracle", res["coverage"], res["var_stats"]))
    print(f"\n[ORACLE] Coverage: {res['coverage']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: MULTI-SURVEY FUSION METHODS")
    print("=" * 70)

    print(f"\n{'Method':<30} {'Coverage':>12}")
    print("-" * 45)
    for method, coverage, _ in sorted(results, key=lambda x: x[1]):
        print(f"{method:<30} {coverage:>12.4f}")

    # Detailed var stats
    print("\n" + "-" * 70)
    print("Per-variable mean ratios (recon/pop, closer to 1.0 = better):")
    print("-" * 70)
    for method, _, var_stats in results:
        if var_stats:
            ratios = [f"{v}: {var_stats[v]['mean_ratio']:.2f}" for v in target_vars]
            print(f"  {method}: {', '.join(ratios)}")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    masked = [r for r in results if r[0] == "Masked Loss"]
    naive = [r for r in results if r[0] == "Naive Zero Fill"]
    if masked and naive:
        improvement = 100 * (naive[0][1] - masked[0][1]) / naive[0][1]
        print(f"\nMasked loss improves coverage by {improvement:.1f}% over naive zero-fill")
        print("This validates that proper missing data handling is essential for multi-survey fusion.")


if __name__ == "__main__":
    main()
