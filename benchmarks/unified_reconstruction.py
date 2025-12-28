"""Unified Population Reconstruction Experiment.

The core question: Is there a more elegant way to reconstruct the full population
from multiple partial surveys than chaining imputation → synthesis → reweighting?

Framing: The country is a generating function for individual/household characteristics.
Different surveys sample different slices (different n, different attributes observed).
Can we jointly model all sources to reconstruct the true joint distribution?

SIMPLIFICATION: This version uses only PERSON-level variables to avoid entity
harmonization complexity (person ↔ household ↔ tax unit ↔ PEU). Real implementation
would need to handle entity crosswalks.

Approaches tested:
1. SEQUENTIAL FUSION: Chain imputations from source to source
2. STACKED IMPUTATION: Stack all sources, treat missing as MAR, impute jointly
3. MOSAIC SYNTHESIS: Train generative model on overlapping pieces
4. UNIFIED GENERATIVE: Single model on all data
5. ORACLE: True population sample (upper bound)
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

from run_cps_benchmark import generate_cps_like_data
from compare_qrf import SequentialQRFWithZeroInflation
from multivariate_metrics import compute_mmd, compute_energy_distance, normalize_data
from microplex import Synthesizer

np.random.seed(42)


def generate_population(n: int = 50000) -> pd.DataFrame:
    """Generate the 'true' population we're trying to reconstruct."""
    print(f"Generating ground truth population (n={n:,})...")
    data = generate_cps_like_data(n, seed=42)

    # Add state (for geographic dimension)
    data["state"] = np.random.choice(50, size=n)  # 0-49 representing states

    return data


def create_surveys(population: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create simulated partial surveys from the population.

    Each survey:
    - Has different sample size
    - Observes different variables
    - May have selection bias
    """
    surveys = {}
    n_pop = len(population)

    # CPS-like: Demographics + some income (wage, unemployment)
    # Oversamples unemployed
    cps_vars = ["age", "education", "is_female", "marital_status", "household_size",
                "is_employed", "wage_income", "uc_income"]
    cps_n = 5000
    # Selection: oversample unemployed
    p_select = np.where(population["is_employed"] == 0, 0.003, 0.0001)
    p_select = p_select / p_select.sum()
    cps_idx = np.random.choice(n_pop, size=cps_n, replace=False, p=p_select)
    surveys["CPS"] = population.iloc[cps_idx][cps_vars].copy().reset_index(drop=True)

    # IRS-like: Tax-relevant income (no demographics like gender, education)
    irs_vars = ["age", "marital_status", "wage_income", "self_emp_income",
                "investment_income", "ss_income"]
    irs_n = 10000
    # Selection: everyone who filed taxes (has income)
    has_income = (population["wage_income"] > 0) | (population["self_emp_income"] > 0)
    irs_pool = np.where(has_income)[0]
    irs_idx = np.random.choice(irs_pool, size=min(irs_n, len(irs_pool)), replace=False)
    surveys["IRS"] = population.iloc[irs_idx][irs_vars].copy().reset_index(drop=True)

    # SIPP-like: Detailed program participation, retirement income
    sipp_vars = ["age", "education", "is_female", "household_size",
                 "ss_income", "investment_income", "is_employed"]
    sipp_n = 3000
    # Selection: oversamples elderly (SS recipients)
    p_select = np.where(population["age"] >= 62, 0.002, 0.0001)
    p_select = p_select / p_select.sum()
    sipp_idx = np.random.choice(n_pop, size=sipp_n, replace=False, p=p_select)
    surveys["SIPP"] = population.iloc[sipp_idx][sipp_vars].copy().reset_index(drop=True)

    # ACS-like: Demographics + geography (no detailed income)
    acs_vars = ["age", "education", "is_female", "marital_status", "household_size",
                "is_employed", "state"]
    acs_n = 20000
    acs_idx = np.random.choice(n_pop, size=acs_n, replace=False)
    surveys["ACS"] = population.iloc[acs_idx][acs_vars].copy().reset_index(drop=True)

    return surveys


def print_survey_summary(surveys: Dict[str, pd.DataFrame]):
    """Print what each survey observes."""
    all_vars = set()
    for df in surveys.values():
        all_vars.update(df.columns)
    all_vars = sorted(all_vars)

    print("\n" + "=" * 80)
    print("SURVEY COVERAGE MATRIX")
    print("=" * 80)
    print(f"\n{'Variable':<20}", end="")
    for name in surveys:
        print(f"{name:>10}", end="")
    print()
    print("-" * (20 + 10 * len(surveys)))

    for var in all_vars:
        print(f"{var:<20}", end="")
        for name, df in surveys.items():
            if var in df.columns:
                print(f"{'✓':>10}", end="")
            else:
                print(f"{'-':>10}", end="")
        print()

    print("-" * (20 + 10 * len(surveys)))
    print(f"{'Sample size':<20}", end="")
    for name, df in surveys.items():
        print(f"{len(df):>10,}", end="")
    print("\n")


# ==============================================================================
# APPROACH 1: SEQUENTIAL FUSION
# ==============================================================================
def sequential_fusion(
    surveys: Dict[str, pd.DataFrame],
    target_n: int,
    all_vars: List[str],
) -> pd.DataFrame:
    """Chain imputations from source to source.

    Strategy: Start with largest survey (ACS), impute income from others.
    """
    print("\n[SEQUENTIAL FUSION] Chaining imputations...")
    start = time.time()

    # Start with ACS as base (largest, has demographics + geography)
    base = surveys["ACS"].copy()

    # Variables we need to impute
    income_vars = ["wage_income", "self_emp_income", "ss_income", "uc_income", "investment_income"]

    # Train QRF on CPS to impute wage_income, uc_income
    cps = surveys["CPS"]
    cps_cond = [v for v in base.columns if v in cps.columns]
    qrf_cps = SequentialQRFWithZeroInflation(
        ["wage_income", "uc_income"],
        [c for c in cps_cond if c not in ["wage_income", "uc_income"]],
        n_estimators=50, max_depth=8
    )
    qrf_cps.fit(cps, verbose=False)
    imputed_cps = qrf_cps.generate(base[[c for c in cps_cond if c not in ["wage_income", "uc_income"]]])
    base["wage_income"] = imputed_cps["wage_income"]
    base["uc_income"] = imputed_cps["uc_income"]

    # Train QRF on IRS to impute self_emp, investment, ss_income
    irs = surveys["IRS"]
    irs_cond = [v for v in ["age", "marital_status", "wage_income"] if v in base.columns]
    qrf_irs = SequentialQRFWithZeroInflation(
        ["self_emp_income", "investment_income", "ss_income"],
        irs_cond,
        n_estimators=50, max_depth=8
    )
    qrf_irs.fit(irs, verbose=False)
    imputed_irs = qrf_irs.generate(base[irs_cond])
    base["self_emp_income"] = imputed_irs["self_emp_income"]
    base["investment_income"] = imputed_irs["investment_income"]
    base["ss_income"] = imputed_irs["ss_income"]

    # Ensure all vars present
    result = base[all_vars].copy()

    print(f"  Time: {time.time() - start:.1f}s")
    return result


# ==============================================================================
# APPROACH 2: STACKED IMPUTATION
# ==============================================================================
def stacked_imputation(
    surveys: Dict[str, pd.DataFrame],
    target_n: int,
    all_vars: List[str],
) -> pd.DataFrame:
    """Stack all surveys, treat missing as MAR, use iterative imputation."""
    print("\n[STACKED IMPUTATION] Joint imputation on stacked data...")
    start = time.time()

    # Stack all surveys with missing values
    stacked_rows = []
    for name, df in surveys.items():
        for _, row in df.iterrows():
            new_row = {v: np.nan for v in all_vars}
            for col in df.columns:
                if col in all_vars:
                    new_row[col] = row[col]
            new_row["_source"] = name
            stacked_rows.append(new_row)

    stacked = pd.DataFrame(stacked_rows)
    source_col = stacked["_source"]
    stacked = stacked.drop("_source", axis=1)

    print(f"  Stacked {len(stacked):,} records from {len(surveys)} surveys")
    print(f"  Missing rate per variable:")
    for var in all_vars:
        miss_rate = stacked[var].isna().mean() * 100
        print(f"    {var}: {miss_rate:.1f}%")

    # Use iterative imputation (MICE-like)
    imputer = IterativeImputer(max_iter=10, random_state=42)
    imputed_arr = imputer.fit_transform(stacked)
    result = pd.DataFrame(imputed_arr, columns=all_vars)

    # Post-process: clip negatives, round categoricals
    for var in all_vars:
        result[var] = result[var].clip(lower=0)

    for var in ["is_female", "is_employed", "marital_status", "education"]:
        if var in result.columns:
            result[var] = result[var].round().clip(0, 1 if var in ["is_female", "is_employed"] else 4)

    # Sample to target size
    if len(result) > target_n:
        result = result.sample(n=target_n, random_state=42).reset_index(drop=True)

    print(f"  Time: {time.time() - start:.1f}s")
    return result


# ==============================================================================
# APPROACH 3: MOSAIC SYNTHESIS (microplex on overlapping pieces)
# ==============================================================================
def mosaic_synthesis(
    surveys: Dict[str, pd.DataFrame],
    target_n: int,
    all_vars: List[str],
) -> pd.DataFrame:
    """Train microplex on each survey's observed variables, combine.

    Key insight: Each survey gives us a partial view of the joint distribution.
    We can train separate models on each, then combine predictions.
    """
    print("\n[MOSAIC SYNTHESIS] Training microplex on each survey's variables...")
    start = time.time()

    # Use ACS as base (largest, best demographic coverage)
    base = surveys["ACS"].sample(n=target_n, replace=True, random_state=42).reset_index(drop=True)

    # For each missing variable, find a survey that has it and train microplex
    income_vars = ["wage_income", "self_emp_income", "ss_income", "uc_income", "investment_income"]

    for target in income_vars:
        # Find surveys that have this target
        for name, survey in surveys.items():
            if target in survey.columns:
                # Find common condition variables
                cond_vars = [v for v in base.columns if v in survey.columns and v != target]
                if len(cond_vars) >= 2:
                    print(f"  Training microplex for {target} using {name} ({len(cond_vars)} conditions)")

                    model = Synthesizer(
                        target_vars=[target],
                        condition_vars=cond_vars,
                        n_layers=4, hidden_dim=32, zero_inflated=True
                    )
                    model.fit(survey, epochs=30, batch_size=256, verbose=False)

                    conditions = base[cond_vars].copy()
                    imputed = model.generate(conditions)
                    base[target] = imputed[target]
                    break

    result = base[all_vars].copy()
    print(f"  Time: {time.time() - start:.1f}s")
    return result


# ==============================================================================
# APPROACH 4: UNIFIED GENERATIVE (train one model on all data)
# ==============================================================================
def unified_generative(
    surveys: Dict[str, pd.DataFrame],
    target_n: int,
    all_vars: List[str],
) -> pd.DataFrame:
    """Train a single microplex model treating surveys as partial observations.

    Approach: First impute to complete the stacked data, then train microplex
    on the complete data for full synthesis.
    """
    print("\n[UNIFIED GENERATIVE] Single model on imputed stacked data...")
    start = time.time()

    # First, stack and impute (like approach 2)
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
    complete_data = pd.DataFrame(imputed_arr, columns=all_vars)

    # Post-process
    for var in all_vars:
        complete_data[var] = complete_data[var].clip(lower=0)
    for var in ["is_female", "is_employed", "marital_status", "education"]:
        if var in complete_data.columns:
            complete_data[var] = complete_data[var].round().clip(0, 1 if var in ["is_female", "is_employed"] else 4)

    # Now train microplex on complete data
    # Split into conditions (demographics) and targets (income)
    cond_vars = ["age", "education", "is_female", "marital_status", "household_size", "is_employed", "state"]
    target_vars = ["wage_income", "self_emp_income", "ss_income", "uc_income", "investment_income"]

    cond_vars = [v for v in cond_vars if v in all_vars]
    target_vars = [v for v in target_vars if v in all_vars]

    print(f"  Training unified microplex model...")
    model = Synthesizer(
        target_vars=target_vars,
        condition_vars=cond_vars,
        n_layers=6, hidden_dim=64, zero_inflated=True
    )
    model.fit(complete_data, epochs=50, batch_size=256, verbose=False)

    # Generate synthetic data
    result = model.sample(target_n, seed=42)

    print(f"  Time: {time.time() - start:.1f}s")
    return result


# ==============================================================================
# EVALUATION
# ==============================================================================
def evaluate_reconstruction(
    population: pd.DataFrame,
    reconstructed: pd.DataFrame,
    name: str,
    eval_vars: List[str],
) -> Dict:
    """Evaluate how well we reconstructed the true population."""
    # Use a holdout from population for comparison
    n_test = min(5000, len(population) // 5)
    test_pop = population.sample(n=n_test, random_state=123)

    # Normalize
    scaler = StandardScaler()
    pop_norm = scaler.fit_transform(test_pop[eval_vars])
    recon_norm = scaler.transform(reconstructed[eval_vars].clip(
        lower=test_pop[eval_vars].min().values,
        upper=test_pop[eval_vars].max().values
    ))

    return {
        "method": name,
        "mmd": compute_mmd(pop_norm, recon_norm),
        "energy": compute_energy_distance(pop_norm, recon_norm),
    }


def evaluate_by_category(
    population: pd.DataFrame,
    reconstructed: pd.DataFrame,
    name: str,
) -> Dict:
    """Evaluate demographics vs income separately."""
    demo_vars = ["age", "education", "is_female", "marital_status", "household_size", "is_employed"]
    income_vars = ["wage_income", "self_emp_income", "ss_income", "uc_income", "investment_income"]

    demo_vars = [v for v in demo_vars if v in reconstructed.columns]
    income_vars = [v for v in income_vars if v in reconstructed.columns]

    n_test = min(5000, len(population) // 5)
    test_pop = population.sample(n=n_test, random_state=123)

    results = {"method": name}

    # Demographics
    if demo_vars:
        scaler = StandardScaler()
        pop_norm = scaler.fit_transform(test_pop[demo_vars])
        recon_clipped = reconstructed[demo_vars].clip(
            lower=test_pop[demo_vars].min().values,
            upper=test_pop[demo_vars].max().values
        )
        recon_norm = scaler.transform(recon_clipped)
        results["demo_mmd"] = compute_mmd(pop_norm, recon_norm)

    # Income
    if income_vars:
        scaler = StandardScaler()
        pop_norm = scaler.fit_transform(test_pop[income_vars])
        recon_clipped = reconstructed[income_vars].clip(
            lower=test_pop[income_vars].min().values,
            upper=test_pop[income_vars].max().values
        )
        recon_norm = scaler.transform(recon_clipped)
        results["income_mmd"] = compute_mmd(pop_norm, recon_norm)

    # Joint
    all_eval = demo_vars + income_vars
    if all_eval:
        scaler = StandardScaler()
        pop_norm = scaler.fit_transform(test_pop[all_eval])
        recon_clipped = reconstructed[all_eval].clip(
            lower=test_pop[all_eval].min().values,
            upper=test_pop[all_eval].max().values
        )
        recon_norm = scaler.transform(recon_clipped)
        results["joint_mmd"] = compute_mmd(pop_norm, recon_norm)

    return results


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # Generate ground truth population
    population = generate_population(50000)

    # Define all variables we care about
    all_vars = [
        "age", "education", "is_female", "marital_status", "household_size",
        "is_employed", "state",
        "wage_income", "self_emp_income", "ss_income", "uc_income", "investment_income"
    ]
    all_vars = [v for v in all_vars if v in population.columns]

    # Create partial surveys
    surveys = create_surveys(population)
    print_survey_summary(surveys)

    # Target: reconstruct to ~20k records
    target_n = 20000

    # Run each approach
    results = []

    # 1. Sequential fusion
    try:
        recon_seq = sequential_fusion(surveys, target_n, all_vars)
        res = evaluate_by_category(population, recon_seq, "Sequential Fusion")
        results.append(res)
        print(f"  → Demo MMD: {res.get('demo_mmd', 'N/A'):.4f}, Income MMD: {res.get('income_mmd', 'N/A'):.4f}, Joint MMD: {res.get('joint_mmd', 'N/A'):.4f}")
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()

    # 2. Stacked imputation
    try:
        recon_stack = stacked_imputation(surveys, target_n, all_vars)
        res = evaluate_by_category(population, recon_stack, "Stacked Imputation")
        results.append(res)
        print(f"  → Demo MMD: {res.get('demo_mmd', 'N/A'):.4f}, Income MMD: {res.get('income_mmd', 'N/A'):.4f}, Joint MMD: {res.get('joint_mmd', 'N/A'):.4f}")
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()

    # 3. Mosaic synthesis
    try:
        recon_mosaic = mosaic_synthesis(surveys, target_n, all_vars)
        res = evaluate_by_category(population, recon_mosaic, "Mosaic Synthesis")
        results.append(res)
        print(f"  → Demo MMD: {res.get('demo_mmd', 'N/A'):.4f}, Income MMD: {res.get('income_mmd', 'N/A'):.4f}, Joint MMD: {res.get('joint_mmd', 'N/A'):.4f}")
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()

    # 4. Unified generative
    try:
        recon_unified = unified_generative(surveys, target_n, all_vars)
        res = evaluate_by_category(population, recon_unified, "Unified Generative")
        results.append(res)
        print(f"  → Demo MMD: {res.get('demo_mmd', 'N/A'):.4f}, Income MMD: {res.get('income_mmd', 'N/A'):.4f}, Joint MMD: {res.get('joint_mmd', 'N/A'):.4f}")
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()

    # 5. Oracle: Sample from true population (upper bound)
    print("\n[ORACLE] Direct sample from true population...")
    oracle_sample = population.sample(n=target_n, random_state=99)[all_vars].reset_index(drop=True)
    res = evaluate_by_category(population, oracle_sample, "Oracle (True Pop)")
    results.append(res)
    print(f"  → Demo MMD: {res.get('demo_mmd', 'N/A'):.4f}, Income MMD: {res.get('income_mmd', 'N/A'):.4f}, Joint MMD: {res.get('joint_mmd', 'N/A'):.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: RECONSTRUCTION QUALITY (MMD, lower is better)")
    print("=" * 80)

    df = pd.DataFrame(results)
    print(f"\n{'Method':<25} {'Demo MMD':>12} {'Income MMD':>12} {'Joint MMD':>12}")
    print("-" * 65)
    for _, row in df.sort_values("joint_mmd").iterrows():
        demo = row.get("demo_mmd", float("nan"))
        income = row.get("income_mmd", float("nan"))
        joint = row.get("joint_mmd", float("nan"))
        print(f"{row['method']:<25} {demo:>12.4f} {income:>12.4f} {joint:>12.4f}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
This experiment tests unified population reconstruction from partial surveys:

SURVEY SIMULATION:
  - "CPS": 5k records, demographics + wage/unemployment, oversamples unemployed
  - "IRS": 10k records, limited demographics + all income, only filers
  - "SIPP": 3k records, demographics + SS/investment, oversamples elderly
  - "ACS": 20k records, full demographics + geography, no income

APPROACHES TESTED:
  1. Sequential Fusion: Chain QRF imputations survey-to-survey
  2. Stacked Imputation: Stack all, treat missing as MAR, iterative impute
  3. Mosaic Synthesis: Train microplex on each survey's variables, combine
  4. Unified Generative: Impute first, then train single microplex model

THEORETICAL FRAMEWORK:
  - Each survey is a partial observation of latent true population
  - Selection bias in each survey (unemployed in CPS, elderly in SIPP, filers in IRS)
  - Goal: Reconstruct P(demographics, income, geography) from partial observations

NEXT STEPS:
  - Add explicit selection bias correction
  - Test with calibration constraints (known marginals from ACS)
  - Compare to Bayesian hierarchical approach
  - Test panel/longitudinal extension
""")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "unified_reconstruction.csv", index=False)
    print(f"\nResults saved to {output_dir}/unified_reconstruction.csv")
