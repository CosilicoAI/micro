"""Unified reconstruction with calibration constraints.

Building on unified_reconstruction.py - adding known marginal constraints
from the "ACS" (population-representative survey) to correct selection bias.

The key insight: We know TRUE marginals from ACS. Can we use these as
constraints to improve reconstruction quality?
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

from run_cps_benchmark import generate_cps_like_data
from compare_qrf import SequentialQRFWithZeroInflation
from multivariate_metrics import compute_mmd, compute_energy_distance
from microplex import Synthesizer

np.random.seed(42)


def generate_population_and_surveys(n_pop: int = 50000):
    """Generate population and surveys with known ground truth marginals."""
    print(f"Generating ground truth population (n={n_pop:,})...")
    population = generate_cps_like_data(n_pop, seed=42)
    population["state"] = np.random.choice(50, size=n_pop)

    # Define surveys with different selection mechanisms
    surveys = {}

    # CPS: oversamples unemployed
    cps_vars = ["age", "education", "is_female", "marital_status", "household_size",
                "is_employed", "wage_income", "uc_income"]
    cps_n = 5000
    p_select = np.where(population["is_employed"] == 0, 0.003, 0.0001)
    p_select = p_select / p_select.sum()
    cps_idx = np.random.choice(n_pop, size=cps_n, replace=False, p=p_select)
    surveys["CPS"] = population.iloc[cps_idx][cps_vars].copy().reset_index(drop=True)

    # IRS: only filers (has income)
    irs_vars = ["age", "marital_status", "wage_income", "self_emp_income",
                "investment_income", "ss_income"]
    irs_n = 10000
    has_income = (population["wage_income"] > 0) | (population["self_emp_income"] > 0)
    irs_pool = np.where(has_income)[0]
    irs_idx = np.random.choice(irs_pool, size=min(irs_n, len(irs_pool)), replace=False)
    surveys["IRS"] = population.iloc[irs_idx][irs_vars].copy().reset_index(drop=True)

    # SIPP: oversamples elderly
    sipp_vars = ["age", "education", "is_female", "household_size",
                 "ss_income", "investment_income", "is_employed"]
    sipp_n = 3000
    p_select = np.where(population["age"] >= 62, 0.002, 0.0001)
    p_select = p_select / p_select.sum()
    sipp_idx = np.random.choice(n_pop, size=sipp_n, replace=False, p=p_select)
    surveys["SIPP"] = population.iloc[sipp_idx][sipp_vars].copy().reset_index(drop=True)

    # ACS: representative (no selection bias) - this gives us TRUE marginals
    acs_vars = ["age", "education", "is_female", "marital_status", "household_size",
                "is_employed", "state"]
    acs_n = 20000
    acs_idx = np.random.choice(n_pop, size=acs_n, replace=False)
    surveys["ACS"] = population.iloc[acs_idx][acs_vars].copy().reset_index(drop=True)

    # Compute true marginals from ACS (these are our calibration targets)
    true_marginals = {
        "mean_age": surveys["ACS"]["age"].mean(),
        "employed_rate": surveys["ACS"]["is_employed"].mean(),
        "female_rate": surveys["ACS"]["is_female"].mean(),
        "age_bins": np.histogram(surveys["ACS"]["age"], bins=[0, 25, 45, 65, 100])[0] / len(surveys["ACS"]),
    }

    return population, surveys, true_marginals


def calibrate_weights(data: pd.DataFrame, true_marginals: Dict) -> np.ndarray:
    """Compute calibration weights using entropy balancing.

    Find weights w such that:
    - sum(w * x) = target for each constraint
    - minimize KL(w || uniform)
    """
    n = len(data)
    initial_weights = np.ones(n) / n

    # Build constraint functions
    constraints = []

    # Age constraint
    def age_constraint(w):
        return np.sum(w * data["age"]) - true_marginals["mean_age"]
    constraints.append({"type": "eq", "fun": age_constraint})

    # Employed rate
    def employed_constraint(w):
        return np.sum(w * data["is_employed"]) - true_marginals["employed_rate"]
    constraints.append({"type": "eq", "fun": employed_constraint})

    # Female rate
    def female_constraint(w):
        return np.sum(w * data["is_female"]) - true_marginals["female_rate"]
    constraints.append({"type": "eq", "fun": female_constraint})

    # Normalization
    def norm_constraint(w):
        return np.sum(w) - 1.0
    constraints.append({"type": "eq", "fun": norm_constraint})

    # Objective: minimize KL divergence from uniform
    def objective(w):
        w = np.clip(w, 1e-10, None)
        return np.sum(w * np.log(w * n))

    # Bounds: weights must be positive
    bounds = [(1e-10, 1) for _ in range(n)]

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 100}
    )

    if result.success:
        return result.x
    else:
        print(f"  Warning: Calibration did not converge: {result.message}")
        return initial_weights


def weighted_sample(data: pd.DataFrame, weights: np.ndarray, n: int) -> pd.DataFrame:
    """Sample from data with calibration weights."""
    idx = np.random.choice(len(data), size=n, replace=True, p=weights)
    return data.iloc[idx].reset_index(drop=True)


def sequential_fusion_with_calibration(
    surveys: Dict[str, pd.DataFrame],
    true_marginals: Dict,
    target_n: int,
    all_vars: List[str],
) -> pd.DataFrame:
    """Sequential fusion + calibration weights."""
    print("\n[SEQ FUSION + CALIBRATION] Chain imputations, then calibrate...")
    start = time.time()

    # Start with ACS as base
    base = surveys["ACS"].copy()

    # Impute income from CPS
    cps = surveys["CPS"]
    cps_cond = [v for v in base.columns if v in cps.columns and v not in ["wage_income", "uc_income"]]
    qrf_cps = SequentialQRFWithZeroInflation(
        ["wage_income", "uc_income"], cps_cond, n_estimators=50, max_depth=8
    )
    qrf_cps.fit(cps, verbose=False)
    imputed = qrf_cps.generate(base[cps_cond])
    base["wage_income"] = imputed["wage_income"]
    base["uc_income"] = imputed["uc_income"]

    # Impute from IRS
    irs = surveys["IRS"]
    irs_cond = ["age", "marital_status", "wage_income"]
    qrf_irs = SequentialQRFWithZeroInflation(
        ["self_emp_income", "investment_income", "ss_income"],
        irs_cond, n_estimators=50, max_depth=8
    )
    qrf_irs.fit(irs, verbose=False)
    imputed = qrf_irs.generate(base[irs_cond])
    base["self_emp_income"] = imputed["self_emp_income"]
    base["investment_income"] = imputed["investment_income"]
    base["ss_income"] = imputed["ss_income"]

    # Now calibrate weights
    print("  Calibrating weights to ACS marginals...")
    weights = calibrate_weights(base, true_marginals)

    # Sample with calibration weights
    result = weighted_sample(base[all_vars], weights, target_n)

    print(f"  Time: {time.time() - start:.1f}s")
    return result


def stacked_with_calibration(
    surveys: Dict[str, pd.DataFrame],
    true_marginals: Dict,
    target_n: int,
    all_vars: List[str],
) -> pd.DataFrame:
    """Stack + impute + calibrate."""
    print("\n[STACKED + CALIBRATION] Joint imputation, then calibrate...")
    start = time.time()

    # Stack all surveys
    stacked_rows = []
    for name, df in surveys.items():
        for _, row in df.iterrows():
            new_row = {v: np.nan for v in all_vars}
            for col in df.columns:
                if col in all_vars:
                    new_row[col] = row[col]
            stacked_rows.append(new_row)

    stacked = pd.DataFrame(stacked_rows)

    # Impute
    imputer = IterativeImputer(max_iter=10, random_state=42)
    imputed_arr = imputer.fit_transform(stacked)
    complete = pd.DataFrame(imputed_arr, columns=all_vars)

    # Post-process
    for var in all_vars:
        complete[var] = complete[var].clip(lower=0)
    for var in ["is_female", "is_employed", "marital_status", "education"]:
        if var in complete.columns:
            complete[var] = complete[var].round().clip(0, 1 if var in ["is_female", "is_employed"] else 4)

    # Calibrate
    print("  Calibrating weights...")
    weights = calibrate_weights(complete, true_marginals)

    # Sample
    result = weighted_sample(complete[all_vars], weights, target_n)

    print(f"  Time: {time.time() - start:.1f}s")
    return result


def evaluate(population: pd.DataFrame, reconstructed: pd.DataFrame, name: str, all_vars: List[str]) -> Dict:
    """Evaluate reconstruction quality."""
    demo_vars = ["age", "education", "is_female", "marital_status", "household_size", "is_employed"]
    income_vars = ["wage_income", "self_emp_income", "ss_income", "uc_income", "investment_income"]

    demo_vars = [v for v in demo_vars if v in all_vars]
    income_vars = [v for v in income_vars if v in all_vars]

    n_test = min(5000, len(population) // 5)
    test_pop = population.sample(n=n_test, random_state=123)[all_vars]

    results = {"method": name}

    for name_grp, vars_grp in [("demo", demo_vars), ("income", income_vars), ("joint", all_vars)]:
        if vars_grp:
            scaler = StandardScaler()
            pop_norm = scaler.fit_transform(test_pop[vars_grp])
            recon_clipped = reconstructed[vars_grp].clip(
                lower=test_pop[vars_grp].min().values,
                upper=test_pop[vars_grp].max().values
            )
            recon_norm = scaler.transform(recon_clipped)
            results[f"{name_grp}_mmd"] = compute_mmd(pop_norm, recon_norm)

    return results


if __name__ == "__main__":
    population, surveys, true_marginals = generate_population_and_surveys(50000)

    all_vars = [
        "age", "education", "is_female", "marital_status", "household_size",
        "is_employed", "state",
        "wage_income", "self_emp_income", "ss_income", "uc_income", "investment_income"
    ]
    all_vars = [v for v in all_vars if v in population.columns]

    target_n = 20000
    results = []

    # Sequential without calibration (baseline)
    print("\n[SEQ FUSION] Without calibration (baseline)...")
    from unified_reconstruction import sequential_fusion
    try:
        recon = sequential_fusion(surveys, target_n, all_vars)
        res = evaluate(population, recon, "Seq Fusion (no cal)", all_vars)
        results.append(res)
        print(f"  → Joint MMD: {res['joint_mmd']:.4f}")
    except Exception as e:
        print(f"  ✗ {e}")

    # Sequential with calibration
    try:
        recon = sequential_fusion_with_calibration(surveys, true_marginals, target_n, all_vars)
        res = evaluate(population, recon, "Seq Fusion + Cal", all_vars)
        results.append(res)
        print(f"  → Joint MMD: {res['joint_mmd']:.4f}")
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()

    # Stacked without calibration
    from unified_reconstruction import stacked_imputation
    try:
        recon = stacked_imputation(surveys, target_n, all_vars)
        res = evaluate(population, recon, "Stacked (no cal)", all_vars)
        results.append(res)
        print(f"  → Joint MMD: {res['joint_mmd']:.4f}")
    except Exception as e:
        print(f"  ✗ {e}")

    # Stacked with calibration
    try:
        recon = stacked_with_calibration(surveys, true_marginals, target_n, all_vars)
        res = evaluate(population, recon, "Stacked + Cal", all_vars)
        results.append(res)
        print(f"  → Joint MMD: {res['joint_mmd']:.4f}")
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()

    # Oracle
    oracle = population.sample(n=target_n, random_state=99)[all_vars].reset_index(drop=True)
    res = evaluate(population, oracle, "Oracle", all_vars)
    results.append(res)
    print(f"\n[ORACLE] Joint MMD: {res['joint_mmd']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: EFFECT OF CALIBRATION")
    print("=" * 70)

    df = pd.DataFrame(results)
    print(f"\n{'Method':<25} {'Demo MMD':>12} {'Income MMD':>12} {'Joint MMD':>12}")
    print("-" * 65)
    for _, row in df.sort_values("joint_mmd").iterrows():
        print(f"{row['method']:<25} {row.get('demo_mmd', 0):>12.4f} {row.get('income_mmd', 0):>12.4f} {row.get('joint_mmd', 0):>12.4f}")

    print("\n" + "=" * 70)
    print("INSIGHT: Does calibration help correct selection bias?")
    print("=" * 70)
