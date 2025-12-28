"""Test reconstruction when surveys cover only 1-2% of population.

This tests the real scenario: can we reconstruct 330M people
from surveys that only observe ~1% of them?

Key question: Does generative modeling help fill in the
unobserved 98% better than just reweighting observed records?
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

from large_scale_reconstruction import generate_large_population, PopulationConfig
from compare_qrf import SequentialQRFWithZeroInflation
from multivariate_metrics import compute_mmd
from microplex import Synthesizer

np.random.seed(42)


def create_sparse_surveys(population: pd.DataFrame, coverage: float = 0.01) -> Dict[str, pd.DataFrame]:
    """Create surveys with realistic sparse coverage.

    Args:
        population: Full population
        coverage: Total fraction of population observed across all surveys
    """
    surveys = {}
    n_pop = len(population)
    rng = np.random.default_rng(42)

    # Scale sample sizes to achieve target coverage
    # Real ratios: ACS ~3.5M, CPS ~100k, IRS ~180k, SIPP ~100k, SCF ~6k
    total_real = 3500000 + 100000 + 180000 + 100000 + 6000  # ~3.9M
    scale = (coverage * n_pop) / total_real

    # ACS: ~90% of coverage (largest survey)
    acs_vars = [
        "age", "is_female", "race", "education", "marital_status",
        "is_citizen", "has_disability", "is_veteran",
        "household_size", "n_children", "housing_tenure",
        "in_labor_force", "is_employed",
        "state", "metro_status", "region"
    ]
    acs_n = int(3500000 * scale)
    acs_idx = rng.choice(n_pop, size=min(acs_n, n_pop), replace=False)
    surveys["ACS"] = population.iloc[acs_idx][
        [v for v in acs_vars if v in population.columns]
    ].copy().reset_index(drop=True)

    # CPS: Labor force details
    cps_vars = [
        "age", "is_female", "education", "marital_status",
        "household_size", "is_employed", "hours_worked", "weeks_worked",
        "industry", "occupation", "class_worker",
        "wage_income", "ui_income",
        "state"
    ]
    cps_n = int(100000 * scale)
    # Oversample unemployed
    p_select = np.where(population["is_employed"] == 0, 0.003, 0.0001)
    p_select = p_select / p_select.sum()
    cps_idx = rng.choice(n_pop, size=min(cps_n, n_pop), replace=False, p=p_select)
    surveys["CPS"] = population.iloc[cps_idx][
        [v for v in cps_vars if v in population.columns]
    ].copy().reset_index(drop=True)

    # IRS: Tax filers only
    irs_vars = [
        "age", "marital_status",
        "wage_income", "self_emp_income", "ss_income", "pension_income",
        "dividend_income", "interest_income", "rental_income",
        "state"
    ]
    irs_n = int(180000 * scale)
    total_income = population["wage_income"] + population["self_emp_income"]
    is_filer = total_income > 12000
    filer_idx = np.where(is_filer)[0]
    irs_idx = rng.choice(filer_idx, size=min(irs_n, len(filer_idx)), replace=False)
    surveys["IRS"] = population.iloc[irs_idx][
        [v for v in irs_vars if v in population.columns]
    ].copy().reset_index(drop=True)

    # SIPP: Program participation
    sipp_vars = [
        "age", "is_female", "education", "marital_status",
        "has_disability", "household_size",
        "ss_income", "ssi_income", "snap_benefits", "ui_income",
        "pension_income", "workers_comp_income", "veterans_benefits",
        "state"
    ]
    sipp_n = int(100000 * scale)
    on_program = (population["ssi_income"] > 0) | (population["snap_benefits"] > 0)
    p_select = np.where(on_program, 0.003, 0.0001)
    p_select = p_select / p_select.sum()
    sipp_idx = rng.choice(n_pop, size=min(sipp_n, n_pop), replace=False, p=p_select)
    surveys["SIPP"] = population.iloc[sipp_idx][
        [v for v in sipp_vars if v in population.columns]
    ].copy().reset_index(drop=True)

    return surveys


def approach_weighted_resampling(
    surveys: Dict[str, pd.DataFrame],
    population: pd.DataFrame,
    target_n: int,
) -> Tuple[pd.DataFrame, float]:
    """Traditional approach: Resample observed records with weights.

    This is essentially what happens when you use survey weights
    to expand a sample to population level.
    """
    print("\n[WEIGHTED RESAMPLING] Expand observed records with weights...")
    start = time.time()

    # Use ACS as base (largest, most representative)
    acs = surveys["ACS"]

    # Simple resampling with replacement
    # In reality, would use calibrated weights
    resampled = acs.sample(n=target_n, replace=True, random_state=42).reset_index(drop=True)

    # Impute missing variables from other surveys
    cps = surveys["CPS"]
    irs = surveys["IRS"]

    # Impute income from CPS
    cps_conds = [v for v in resampled.columns if v in cps.columns]
    cps_targets = ["wage_income", "ui_income"]
    cps_targets = [v for v in cps_targets if v in cps.columns]
    if cps_targets and len(cps) > 100:
        qrf = SequentialQRFWithZeroInflation(
            cps_targets,
            [c for c in cps_conds if c not in cps_targets],
            n_estimators=50, max_depth=8
        )
        qrf.fit(cps, verbose=False)
        imputed = qrf.generate(resampled[[c for c in cps_conds if c not in cps_targets]])
        for v in cps_targets:
            resampled[v] = imputed[v]

    # Impute from IRS
    irs_conds = [v for v in resampled.columns if v in irs.columns]
    irs_targets = ["self_emp_income", "ss_income", "dividend_income", "interest_income"]
    irs_targets = [v for v in irs_targets if v in irs.columns and v not in resampled.columns]
    if irs_targets and len(irs) > 100:
        qrf = SequentialQRFWithZeroInflation(
            irs_targets,
            [c for c in irs_conds if c not in irs_targets],
            n_estimators=50, max_depth=8
        )
        qrf.fit(irs, verbose=False)
        imputed = qrf.generate(resampled[[c for c in irs_conds if c not in irs_targets]])
        for v in irs_targets:
            resampled[v] = imputed[v]

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s, Records: {len(resampled)}, Vars: {len(resampled.columns)}")
    return resampled, elapsed


def approach_generative_synthesis(
    surveys: Dict[str, pd.DataFrame],
    population: pd.DataFrame,
    target_n: int,
) -> Tuple[pd.DataFrame, float]:
    """Generative approach: Train model on observed, generate new records.

    This should be able to create combinations that weren't observed
    in the original surveys.
    """
    print("\n[GENERATIVE SYNTHESIS] Train microplex, generate new records...")
    start = time.time()

    # Combine surveys (with imputation) to get training data
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    # Get all variables
    all_vars = set()
    for df in surveys.values():
        all_vars.update(df.columns)
    all_vars = sorted(all_vars)

    # Stack surveys
    stacked_rows = []
    for name, df in surveys.items():
        for _, row in df.iterrows():
            new_row = {v: np.nan for v in all_vars}
            for col in df.columns:
                new_row[col] = row[col]
            stacked_rows.append(new_row)

    stacked = pd.DataFrame(stacked_rows)

    # Impute to get complete training data
    # Use HistGradientBoosting which handles NaN natively
    from sklearn.ensemble import HistGradientBoostingRegressor
    imputer = IterativeImputer(
        estimator=HistGradientBoostingRegressor(max_iter=50, random_state=42),
        max_iter=5, random_state=42
    )
    imputed_arr = imputer.fit_transform(stacked)
    complete = pd.DataFrame(imputed_arr, columns=all_vars)

    # Post-process
    for var in all_vars:
        complete[var] = complete[var].clip(lower=0)

    # Define conditions and targets
    cond_vars = ["age", "is_female", "education", "marital_status",
                 "household_size", "is_employed", "state"]
    target_vars = ["wage_income", "self_emp_income", "ss_income",
                   "dividend_income", "interest_income"]

    cond_vars = [v for v in cond_vars if v in all_vars]
    target_vars = [v for v in target_vars if v in all_vars]

    print(f"  Training on {len(complete)} records...")

    # Train microplex
    model = Synthesizer(
        target_vars=target_vars,
        condition_vars=cond_vars,
        n_layers=6, hidden_dim=64, zero_inflated=True
    )
    model.fit(complete, epochs=50, batch_size=256, verbose=False)

    # Generate new records
    synthetic = model.sample(target_n, seed=42)

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s, Records: {len(synthetic)}, Vars: {len(synthetic.columns)}")
    return synthetic, elapsed


def measure_coverage_of_rare_combinations(
    population: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> Dict:
    """Measure how well rare demographic × income combinations are covered."""

    # Define "rare" combinations
    # e.g., elderly (65+) with high self-employment income
    rare_elderly_selfemp = (
        (population["age"] >= 65) &
        (population["self_emp_income"] > 50000)
    )
    n_rare_pop = rare_elderly_selfemp.sum()

    if "age" in reconstructed.columns and "self_emp_income" in reconstructed.columns:
        rare_recon = (
            (reconstructed["age"] >= 65) &
            (reconstructed["self_emp_income"] > 50000)
        )
        n_rare_recon = rare_recon.sum()
    else:
        n_rare_recon = 0

    # Young (18-25) with high dividends
    rare_young_div = (
        (population["age"] >= 18) & (population["age"] <= 25) &
        (population["dividend_income"] > 10000)
    )
    n_rare_young_pop = rare_young_div.sum()

    if "age" in reconstructed.columns and "dividend_income" in reconstructed.columns:
        rare_young_recon = (
            (reconstructed["age"] >= 18) & (reconstructed["age"] <= 25) &
            (reconstructed["dividend_income"] > 10000)
        )
        n_rare_young_recon = rare_young_recon.sum()
    else:
        n_rare_young_recon = 0

    return {
        "elderly_selfemp_pop": n_rare_pop,
        "elderly_selfemp_recon": n_rare_recon,
        "elderly_selfemp_ratio": n_rare_recon / max(1, n_rare_pop * len(reconstructed) / len(population)),
        "young_dividend_pop": n_rare_young_pop,
        "young_dividend_recon": n_rare_young_recon,
        "young_dividend_ratio": n_rare_young_recon / max(1, n_rare_young_pop * len(reconstructed) / len(population)),
    }


def compute_coverage_metric(
    holdout: np.ndarray,
    synthetic: np.ndarray,
    k: int = 1,
) -> Tuple[float, np.ndarray]:
    """Compute coverage: avg distance from holdout to nearest synthetic.

    For each holdout record, find distance to k-th nearest synthetic record.
    Lower = better coverage of the population space.

    Returns:
        mean_coverage: Average distance to nearest synthetic
        distances: Array of distances for distribution analysis
    """
    from sklearn.neighbors import NearestNeighbors

    # Fit on synthetic, query with holdout
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn.fit(synthetic)
    distances, _ = nn.kneighbors(holdout)

    # Distance to k-th nearest neighbor
    dist_to_kth = distances[:, k-1]

    return float(np.mean(dist_to_kth)), dist_to_kth


def evaluate(population: pd.DataFrame, reconstructed: pd.DataFrame, name: str) -> Dict:
    """Evaluate reconstruction quality with coverage metric."""
    common_vars = [v for v in reconstructed.columns if v in population.columns]

    # Use a holdout sample from population
    n_test = min(10000, len(population) // 10)
    test_pop = population.sample(n=n_test, random_state=123)

    demo_vars = ["age", "is_female", "education", "marital_status"]
    income_vars = ["wage_income", "self_emp_income", "ss_income", "dividend_income"]

    demo_vars = [v for v in demo_vars if v in common_vars]
    income_vars = [v for v in income_vars if v in common_vars]
    joint_vars = demo_vars + income_vars

    results = {"method": name}

    # MMD metrics
    for grp_name, grp_vars in [("demo", demo_vars), ("income", income_vars)]:
        if grp_vars:
            scaler = StandardScaler()
            pop_norm = scaler.fit_transform(test_pop[grp_vars])

            recon_clipped = reconstructed[grp_vars].copy()
            for v in grp_vars:
                recon_clipped[v] = recon_clipped[v].clip(
                    lower=test_pop[v].quantile(0.001),
                    upper=test_pop[v].quantile(0.999)
                )
            recon_norm = scaler.transform(recon_clipped)

            # Subsample for MMD
            if len(pop_norm) > 5000:
                idx = np.random.choice(len(pop_norm), 5000, replace=False)
                pop_norm_sub = pop_norm[idx]
            else:
                pop_norm_sub = pop_norm
            if len(recon_norm) > 5000:
                idx = np.random.choice(len(recon_norm), 5000, replace=False)
                recon_norm_sub = recon_norm[idx]
            else:
                recon_norm_sub = recon_norm

            results[f"{grp_name}_mmd"] = compute_mmd(pop_norm_sub, recon_norm_sub)

    # COVERAGE METRIC: For each holdout person, distance to nearest synthetic
    if joint_vars:
        scaler = StandardScaler()
        pop_norm = scaler.fit_transform(test_pop[joint_vars])

        recon_clipped = reconstructed[joint_vars].copy()
        for v in joint_vars:
            if v in test_pop.columns:
                recon_clipped[v] = recon_clipped[v].clip(
                    lower=test_pop[v].quantile(0.001),
                    upper=test_pop[v].quantile(0.999)
                )
        recon_norm = scaler.transform(recon_clipped)

        # Subsample for efficiency
        if len(pop_norm) > 5000:
            idx = np.random.choice(len(pop_norm), 5000, replace=False)
            pop_norm = pop_norm[idx]

        mean_coverage, distances = compute_coverage_metric(pop_norm, recon_norm, k=1)
        results["coverage_mean"] = mean_coverage
        results["coverage_median"] = float(np.median(distances))
        results["coverage_p90"] = float(np.percentile(distances, 90))
        results["coverage_p99"] = float(np.percentile(distances, 99))

    # Rare combination coverage
    rare = measure_coverage_of_rare_combinations(population, reconstructed)
    results.update(rare)

    return results


if __name__ == "__main__":
    # Generate large population
    config = PopulationConfig(n=500000, seed=42)
    population = generate_large_population(config)

    # Test different coverage levels
    coverage_levels = [0.10, 0.02, 0.01]  # 10%, 2%, 1%

    all_results = []

    for coverage in coverage_levels:
        print(f"\n{'='*80}")
        print(f"TESTING WITH {coverage*100:.0f}% SURVEY COVERAGE")
        print(f"{'='*80}")

        surveys = create_sparse_surveys(population, coverage=coverage)

        total_observed = sum(len(df) for df in surveys.values())
        print(f"\nSurveys created:")
        for name, df in surveys.items():
            print(f"  {name}: {len(df):,} records, {len(df.columns)} vars")
        print(f"  Total observed: {total_observed:,} ({total_observed/len(population)*100:.1f}%)")

        target_n = 100000

        # Weighted resampling
        try:
            recon_weighted, time_w = approach_weighted_resampling(surveys, population, target_n)
            res = evaluate(population, recon_weighted, f"Weighted ({coverage*100:.0f}%)")
            res["time"] = time_w
            res["coverage"] = coverage
            all_results.append(res)
            print(f"  → Demo MMD: {res.get('demo_mmd', 'N/A'):.4f}, Income MMD: {res.get('income_mmd', 'N/A'):.4f}")
            print(f"  → Rare elderly+selfemp: {res['elderly_selfemp_ratio']:.2f}x expected")
        except Exception as e:
            print(f"  ✗ {e}")

        # Generative synthesis
        try:
            recon_gen, time_g = approach_generative_synthesis(surveys, population, target_n)
            res = evaluate(population, recon_gen, f"Generative ({coverage*100:.0f}%)")
            res["time"] = time_g
            res["coverage"] = coverage
            all_results.append(res)
            print(f"  → Demo MMD: {res.get('demo_mmd', 'N/A'):.4f}, Income MMD: {res.get('income_mmd', 'N/A'):.4f}")
            print(f"  → Rare elderly+selfemp: {res['elderly_selfemp_ratio']:.2f}x expected")
        except Exception as e:
            print(f"  ✗ {e}")
            import traceback; traceback.print_exc()

    # Oracle
    oracle = population.sample(n=100000, random_state=99).reset_index(drop=True)
    res = evaluate(population, oracle, "Oracle")
    res["time"] = 0
    res["coverage"] = 1.0
    all_results.append(res)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY: SPARSE COVERAGE RECONSTRUCTION")
    print("=" * 100)

    df = pd.DataFrame(all_results)
    print(f"\n{'Method':<25} {'Survey %':>10} {'Cov Mean':>10} {'Cov P90':>10} {'Income MMD':>12}")
    print("-" * 75)
    for _, row in df.iterrows():
        cov_mean = row.get("coverage_mean", float("nan"))
        cov_p90 = row.get("coverage_p90", float("nan"))
        income = row.get("income_mmd", float("nan"))
        print(f"{row['method']:<25} {row.get('coverage', 0)*100:>9.0f}% {cov_mean:>10.4f} {cov_p90:>10.4f} {income:>12.4f}")

    print("\n" + "=" * 100)
    print("COVERAGE METRIC INTERPRETATION")
    print("=" * 100)
    print("""
Coverage = avg distance from each holdout person to nearest synthetic record
  - Lower = better (every real person has a close synthetic match)
  - Measures whether synthetic data "covers" the full population space

Key question: With 1% survey coverage, can generative synthesis create
records that cover the 99% unobserved population better than just
reweighting the 1% observed records?

If generative has LOWER coverage distance than weighted resampling,
it's successfully generating novel combinations that weren't observed.
""")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "sparse_coverage.csv", index=False)
