"""Large-scale population reconstruction experiment.

Tests whether unified approaches scale better than sequential fusion
at realistic data sizes (1M people, 50 variables).

Key questions:
1. Does sequential fusion still win at scale?
2. Does unified latent approach benefit from more data?
3. How do compute times scale?
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

from compare_qrf import SequentialQRFWithZeroInflation
from multivariate_metrics import compute_mmd
from microplex import Synthesizer

np.random.seed(42)


# =============================================================================
# POPULATION GENERATOR - 50 realistic variables
# =============================================================================

@dataclass
class PopulationConfig:
    """Configuration for synthetic population."""
    n: int = 1_000_000
    seed: int = 42


def generate_large_population(config: PopulationConfig) -> pd.DataFrame:
    """Generate population with ~50 correlated variables.

    Variables organized by category:
    - Demographics (10): age, gender, race, education, marital, citizenship,
                         disability, veteran, english_proficiency, nativity
    - Household (8): hh_size, n_children, n_elderly, housing_tenure,
                     housing_type, rooms, vehicles, internet
    - Employment (10): employed, labor_force, hours_worked, weeks_worked,
                       industry, occupation, class_worker, employer_size,
                       multiple_jobs, work_from_home
    - Income (15): wage, self_emp, ss, ssi, snap, unemployment, pension,
                   dividends, interest, rental, alimony, child_support,
                   workers_comp, veterans_benefits, other_income
    - Geographic (5): state, metro, region, puma, urban
    """
    print(f"Generating population with {config.n:,} people...")
    rng = np.random.default_rng(config.seed)
    n = config.n

    data = {}

    # === DEMOGRAPHICS ===
    # Age: realistic distribution
    data["age"] = np.clip(rng.normal(42, 20, n), 0, 95).astype(int)

    # Gender: ~51% female
    data["is_female"] = rng.binomial(1, 0.51, n)

    # Race: simplified categories (0-4)
    race_probs = [0.60, 0.13, 0.06, 0.18, 0.03]  # White, Black, Asian, Hispanic, Other
    data["race"] = rng.choice(5, n, p=race_probs)

    # Education: correlated with age
    # 0=less than HS, 1=HS, 2=some college, 3=bachelors, 4=graduate
    base_edu = rng.choice(5, n, p=[0.10, 0.27, 0.29, 0.22, 0.12])
    # Younger people have more education on average
    age_effect = np.clip((50 - data["age"]) / 100, -0.2, 0.2)
    data["education"] = np.clip(base_edu + rng.binomial(1, np.abs(age_effect), n) * np.sign(age_effect), 0, 4).astype(int)

    # Marital status: correlated with age
    # 0=never married, 1=married, 2=divorced, 3=widowed, 4=separated
    married_prob = np.clip(0.1 + (data["age"] - 20) * 0.02, 0, 0.7)
    divorced_prob = np.clip((data["age"] - 30) * 0.005, 0, 0.15)
    widowed_prob = np.clip((data["age"] - 60) * 0.01, 0, 0.2)
    never_married_prob = 1 - married_prob - divorced_prob - widowed_prob
    marital = np.zeros(n, dtype=int)
    for i in range(n):
        probs = [max(0, never_married_prob[i]), married_prob[i], divorced_prob[i], widowed_prob[i], 0.02]
        probs = np.array(probs) / sum(probs)
        marital[i] = rng.choice(5, p=probs)
    data["marital_status"] = marital

    # Citizenship: 92% citizens
    data["is_citizen"] = rng.binomial(1, 0.92, n)

    # Disability: increases with age
    disability_prob = np.clip(0.05 + (data["age"] - 40) * 0.003, 0.02, 0.4)
    data["has_disability"] = rng.binomial(1, disability_prob)

    # Veteran: more likely male, older
    veteran_base = 0.07 * (1 - data["is_female"]) * np.clip((data["age"] - 18) / 50, 0, 1)
    data["is_veteran"] = rng.binomial(1, veteran_base)

    # English proficiency (0=not at all, 1=not well, 2=well, 3=very well, 4=native)
    data["english_proficiency"] = np.where(
        data["is_citizen"] == 1,
        rng.choice([3, 4], n, p=[0.1, 0.9]),
        rng.choice(5, n, p=[0.05, 0.1, 0.2, 0.35, 0.3])
    )

    # Nativity: born in US
    data["born_in_us"] = rng.binomial(1, 0.86, n)

    # === HOUSEHOLD ===
    # Household size: correlated with marital status
    base_hh = np.where(data["marital_status"] == 1, 2.5, 1.5)
    data["household_size"] = np.clip(rng.poisson(base_hh), 1, 10)

    # Number of children: higher for married, working age
    child_rate = np.where(
        (data["marital_status"] == 1) & (data["age"] > 25) & (data["age"] < 55),
        1.5, 0.3
    )
    data["n_children"] = np.clip(rng.poisson(child_rate), 0, 6)

    # Number of elderly in household
    elderly_rate = np.where(data["age"] >= 65, 0.5, 0.1)
    data["n_elderly"] = np.clip(rng.poisson(elderly_rate), 0, 3)

    # Housing tenure: 0=own, 1=rent, 2=other
    own_prob = np.clip(0.3 + (data["age"] - 25) * 0.01 + data["education"] * 0.05, 0.2, 0.8)
    data["housing_tenure"] = np.where(rng.random(n) < own_prob, 0, 1)

    # Housing type: 0=single family, 1=multi-unit, 2=mobile, 3=other
    data["housing_type"] = rng.choice(4, n, p=[0.6, 0.3, 0.07, 0.03])

    # Number of rooms
    data["n_rooms"] = np.clip(rng.poisson(4 + data["household_size"] * 0.5), 1, 15)

    # Number of vehicles
    data["n_vehicles"] = np.clip(rng.poisson(1.5 + 0.3 * (data["housing_tenure"] == 0)), 0, 5)

    # Internet access
    data["has_internet"] = rng.binomial(1, 0.85 + data["education"] * 0.02)

    # === EMPLOYMENT ===
    # Labor force participation: age and disability dependent
    lf_prob = np.where(
        (data["age"] >= 16) & (data["age"] < 65) & (data["has_disability"] == 0),
        0.8, 0.2
    )
    data["in_labor_force"] = rng.binomial(1, lf_prob)

    # Employment: conditional on labor force
    emp_prob = np.where(data["in_labor_force"] == 1, 0.94, 0)
    data["is_employed"] = rng.binomial(1, emp_prob)

    # Hours worked per week (if employed)
    data["hours_worked"] = np.where(
        data["is_employed"] == 1,
        np.clip(rng.normal(40, 10, n), 1, 80),
        0
    ).astype(int)

    # Weeks worked per year (if employed)
    data["weeks_worked"] = np.where(
        data["is_employed"] == 1,
        np.clip(rng.normal(48, 8, n), 1, 52),
        0
    ).astype(int)

    # Industry (0-12)
    data["industry"] = np.where(
        data["is_employed"] == 1,
        rng.choice(13, n),
        -1
    )

    # Occupation (0-9)
    data["occupation"] = np.where(
        data["is_employed"] == 1,
        rng.choice(10, n),
        -1
    )

    # Class of worker: 0=private, 1=government, 2=self-employed, 3=unpaid family
    data["class_worker"] = np.where(
        data["is_employed"] == 1,
        rng.choice(4, n, p=[0.75, 0.15, 0.09, 0.01]),
        -1
    )

    # Employer size
    data["employer_size"] = np.where(
        data["is_employed"] == 1,
        rng.choice(5, n, p=[0.2, 0.2, 0.2, 0.2, 0.2]),  # 0=<10, 1=10-49, etc.
        -1
    )

    # Multiple jobs
    data["has_multiple_jobs"] = np.where(
        data["is_employed"] == 1,
        rng.binomial(1, 0.07, n),
        0
    )

    # Work from home
    wfh_prob = np.where(
        (data["is_employed"] == 1) & (data["education"] >= 3),
        0.3, 0.05
    )
    data["works_from_home"] = rng.binomial(1, wfh_prob)

    # === INCOME ===
    # Base wage: correlated with education, age, employment
    log_wage_mean = 10 + data["education"] * 0.3 + np.clip((data["age"] - 25) / 30, 0, 1) * 0.5
    log_wage_std = 0.8 - data["education"] * 0.05
    wage = np.where(
        data["is_employed"] == 1,
        np.exp(rng.normal(log_wage_mean, log_wage_std)),
        0
    )
    data["wage_income"] = np.clip(wage, 0, 500000).astype(int)

    # Self-employment income
    self_emp_rate = np.where(data["class_worker"] == 2, 0.9, 0.05)
    has_self_emp = rng.binomial(1, self_emp_rate)
    data["self_emp_income"] = np.where(
        has_self_emp == 1,
        np.clip(np.exp(rng.normal(10, 1.2, n)), 0, 300000),
        0
    ).astype(int)

    # Social Security: primarily elderly
    ss_rate = np.where(data["age"] >= 62, 0.85, 0.05)
    has_ss = rng.binomial(1, ss_rate)
    data["ss_income"] = np.where(
        has_ss == 1,
        np.clip(rng.normal(18000, 6000, n), 0, 50000),
        0
    ).astype(int)

    # SSI: disability or elderly poor
    ssi_rate = np.where(
        (data["has_disability"] == 1) | ((data["age"] >= 65) & (data["wage_income"] < 10000)),
        0.3, 0.01
    )
    has_ssi = rng.binomial(1, ssi_rate)
    data["ssi_income"] = np.where(
        has_ssi == 1,
        np.clip(rng.normal(8000, 2000, n), 0, 12000),
        0
    ).astype(int)

    # SNAP: low income
    total_income = data["wage_income"] + data["self_emp_income"] + data["ss_income"]
    snap_rate = np.where(total_income < 25000, 0.25, 0.02)
    has_snap = rng.binomial(1, snap_rate)
    data["snap_benefits"] = np.where(
        has_snap == 1,
        np.clip(rng.normal(3000, 1500, n), 0, 8000),
        0
    ).astype(int)

    # Unemployment insurance
    ui_rate = np.where(
        (data["in_labor_force"] == 1) & (data["is_employed"] == 0),
        0.5, 0.01
    )
    has_ui = rng.binomial(1, ui_rate)
    data["ui_income"] = np.where(
        has_ui == 1,
        np.clip(rng.normal(8000, 4000, n), 0, 30000),
        0
    ).astype(int)

    # Pension income
    pension_rate = np.where(data["age"] >= 60, 0.3, 0.02)
    has_pension = rng.binomial(1, pension_rate)
    data["pension_income"] = np.where(
        has_pension == 1,
        np.clip(rng.normal(15000, 10000, n), 0, 80000),
        0
    ).astype(int)

    # Dividend income: correlated with education, age
    div_rate = np.clip(0.05 + data["education"] * 0.05 + (data["age"] - 40) * 0.002, 0, 0.4)
    has_div = rng.binomial(1, div_rate)
    data["dividend_income"] = np.where(
        has_div == 1,
        np.clip(np.exp(rng.normal(7, 2, n)), 0, 100000),
        0
    ).astype(int)

    # Interest income
    int_rate = np.clip(0.1 + data["education"] * 0.05, 0, 0.5)
    has_int = rng.binomial(1, int_rate)
    data["interest_income"] = np.where(
        has_int == 1,
        np.clip(np.exp(rng.normal(6, 2, n)), 0, 50000),
        0
    ).astype(int)

    # Rental income
    rental_rate = np.where(data["housing_tenure"] == 0, 0.08, 0.02)
    has_rental = rng.binomial(1, rental_rate)
    data["rental_income"] = np.where(
        has_rental == 1,
        np.clip(np.exp(rng.normal(9, 1.5, n)), 0, 100000),
        0
    ).astype(int)

    # Alimony
    alimony_rate = np.where(data["marital_status"] == 2, 0.05, 0.001)
    has_alimony = rng.binomial(1, alimony_rate)
    data["alimony_income"] = np.where(
        has_alimony == 1,
        np.clip(rng.normal(12000, 6000, n), 0, 50000),
        0
    ).astype(int)

    # Child support
    cs_rate = np.where(
        (data["marital_status"].isin([2, 4])) & (data["n_children"] > 0),
        0.3, 0.01
    ) if hasattr(data["marital_status"], 'isin') else np.where(
        ((data["marital_status"] == 2) | (data["marital_status"] == 4)) & (data["n_children"] > 0),
        0.3, 0.01
    )
    has_cs = rng.binomial(1, cs_rate)
    data["child_support_income"] = np.where(
        has_cs == 1,
        np.clip(rng.normal(6000, 3000, n), 0, 30000),
        0
    ).astype(int)

    # Workers compensation
    wc_rate = np.where(data["has_disability"] == 1, 0.1, 0.005)
    has_wc = rng.binomial(1, wc_rate)
    data["workers_comp_income"] = np.where(
        has_wc == 1,
        np.clip(rng.normal(15000, 8000, n), 0, 50000),
        0
    ).astype(int)

    # Veterans benefits
    vet_ben_rate = np.where(data["is_veteran"] == 1, 0.4, 0)
    has_vet = rng.binomial(1, vet_ben_rate)
    data["veterans_benefits"] = np.where(
        has_vet == 1,
        np.clip(rng.normal(15000, 10000, n), 0, 60000),
        0
    ).astype(int)

    # Other income
    other_rate = 0.1
    has_other = rng.binomial(1, other_rate)
    data["other_income"] = np.where(
        has_other == 1,
        np.clip(np.exp(rng.normal(7, 2, n)), 0, 50000),
        0
    ).astype(int)

    # === GEOGRAPHIC ===
    # State (0-49)
    # Population-weighted
    state_pops = np.array([4.9, 0.7, 7.2, 3.0, 39.5, 5.8, 3.6, 1.0, 21.5, 10.7,
                          1.4, 1.9, 12.8, 6.8, 3.2, 2.9, 4.5, 4.6, 1.4, 6.2,
                          7.0, 10.0, 5.7, 3.0, 6.2, 1.1, 1.9, 3.1, 1.4, 9.3,
                          2.1, 19.8, 10.4, 0.8, 11.8, 4.0, 4.2, 13.0, 1.1, 5.1,
                          0.9, 6.9, 29.1, 3.3, 0.6, 8.6, 7.6, 1.8, 5.9, 0.6])
    state_probs = state_pops / state_pops.sum()
    data["state"] = rng.choice(50, n, p=state_probs)

    # Metro status: 0=principal city, 1=metro not principal, 2=non-metro
    data["metro_status"] = rng.choice(3, n, p=[0.3, 0.5, 0.2])

    # Region: 0=Northeast, 1=Midwest, 2=South, 3=West
    data["region"] = rng.choice(4, n, p=[0.17, 0.21, 0.38, 0.24])

    # PUMA (simplified: 0-999)
    data["puma"] = rng.choice(1000, n)

    # Urban/rural
    urban_prob = np.where(data["metro_status"] < 2, 0.9, 0.3)
    data["is_urban"] = rng.binomial(1, urban_prob)

    df = pd.DataFrame(data)
    print(f"  Generated {len(df.columns)} variables")

    return df


def create_surveys_large(population: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create simulated surveys with realistic coverage patterns."""
    surveys = {}
    n_pop = len(population)
    rng = np.random.default_rng(42)

    # CPS: Monthly labor force survey
    # ~60k households/month, oversamples unemployment areas
    cps_vars = [
        "age", "is_female", "race", "education", "marital_status",
        "household_size", "n_children", "housing_tenure",
        "in_labor_force", "is_employed", "hours_worked", "weeks_worked",
        "industry", "occupation", "class_worker",
        "wage_income", "ui_income",
        "state", "metro_status"
    ]
    cps_n = 60000
    # Selection: oversample low employment areas
    emp_rate_by_state = population.groupby("state")["is_employed"].mean()
    state_weights = 1 / (emp_rate_by_state[population["state"]].values + 0.5)
    p_select = state_weights / state_weights.sum()
    cps_idx = rng.choice(n_pop, size=cps_n, replace=False, p=p_select)
    surveys["CPS"] = population.iloc[cps_idx][cps_vars].copy().reset_index(drop=True)

    # ACS: Large demographic survey
    # ~3.5M people, representative sample
    acs_vars = [
        "age", "is_female", "race", "education", "marital_status",
        "is_citizen", "has_disability", "is_veteran", "english_proficiency", "born_in_us",
        "household_size", "n_children", "n_elderly", "housing_tenure", "housing_type",
        "n_rooms", "n_vehicles", "has_internet",
        "in_labor_force", "is_employed", "works_from_home",
        "state", "metro_status", "region", "puma", "is_urban"
    ]
    acs_n = 200000  # Scaled down from 3.5M
    acs_idx = rng.choice(n_pop, size=acs_n, replace=False)
    surveys["ACS"] = population.iloc[acs_idx][acs_vars].copy().reset_index(drop=True)

    # IRS PUF: Tax filers only
    irs_vars = [
        "age", "marital_status",
        "wage_income", "self_emp_income", "ss_income", "pension_income",
        "dividend_income", "interest_income", "rental_income",
        "alimony_income", "other_income",
        "state"
    ]
    irs_n = 100000
    # Selection: only tax filers (has significant income)
    total_income = (population["wage_income"] + population["self_emp_income"] +
                   population["ss_income"] + population["pension_income"])
    is_filer = total_income > 12000  # Approximate filing threshold
    filer_idx = np.where(is_filer)[0]
    irs_idx = rng.choice(filer_idx, size=min(irs_n, len(filer_idx)), replace=False)
    surveys["IRS"] = population.iloc[irs_idx][irs_vars].copy().reset_index(drop=True)

    # SIPP: Detailed program participation
    sipp_vars = [
        "age", "is_female", "race", "education", "marital_status",
        "has_disability", "is_veteran",
        "household_size", "n_children",
        "is_employed",
        "ss_income", "ssi_income", "snap_benefits", "ui_income",
        "pension_income", "workers_comp_income", "veterans_benefits",
        "state"
    ]
    sipp_n = 50000
    # Selection: oversample program participants
    on_program = (population["ssi_income"] > 0) | (population["snap_benefits"] > 0)
    p_select = np.where(on_program, 0.003, 0.0001)
    p_select = p_select / p_select.sum()
    sipp_idx = rng.choice(n_pop, size=sipp_n, replace=False, p=p_select)
    surveys["SIPP"] = population.iloc[sipp_idx][sipp_vars].copy().reset_index(drop=True)

    # SCF: Wealth survey (very small)
    scf_vars = [
        "age", "is_female", "education", "marital_status",
        "household_size",
        "is_employed",
        "wage_income", "self_emp_income",
        "dividend_income", "interest_income", "rental_income",
        "housing_tenure", "n_vehicles"
    ]
    scf_n = 6000
    # Selection: oversample wealthy
    wealth_proxy = population["dividend_income"] + population["interest_income"] + population["rental_income"]
    p_select = np.where(wealth_proxy > 10000, 0.005, 0.0001)
    p_select = p_select / p_select.sum()
    scf_idx = rng.choice(n_pop, size=scf_n, replace=False, p=p_select)
    surveys["SCF"] = population.iloc[scf_idx][scf_vars].copy().reset_index(drop=True)

    return surveys


def print_coverage_matrix(surveys: Dict[str, pd.DataFrame]):
    """Print variable coverage across surveys."""
    all_vars = set()
    for df in surveys.values():
        all_vars.update(df.columns)
    all_vars = sorted(all_vars)

    print("\n" + "=" * 100)
    print("SURVEY COVERAGE MATRIX")
    print("=" * 100)

    print(f"\n{'Variable':<25}", end="")
    for name in surveys:
        print(f"{name:>12}", end="")
    print()
    print("-" * (25 + 12 * len(surveys)))

    # Group by category
    categories = {
        "Demographics": ["age", "is_female", "race", "education", "marital_status",
                        "is_citizen", "has_disability", "is_veteran", "english_proficiency", "born_in_us"],
        "Household": ["household_size", "n_children", "n_elderly", "housing_tenure",
                     "housing_type", "n_rooms", "n_vehicles", "has_internet"],
        "Employment": ["in_labor_force", "is_employed", "hours_worked", "weeks_worked",
                      "industry", "occupation", "class_worker", "employer_size",
                      "has_multiple_jobs", "works_from_home"],
        "Income": ["wage_income", "self_emp_income", "ss_income", "ssi_income", "snap_benefits",
                  "ui_income", "pension_income", "dividend_income", "interest_income",
                  "rental_income", "alimony_income", "child_support_income",
                  "workers_comp_income", "veterans_benefits", "other_income"],
        "Geographic": ["state", "metro_status", "region", "puma", "is_urban"]
    }

    for cat_name, cat_vars in categories.items():
        print(f"\n{cat_name}:")
        for var in cat_vars:
            if var in all_vars:
                print(f"  {var:<23}", end="")
                for name, df in surveys.items():
                    if var in df.columns:
                        print(f"{'✓':>12}", end="")
                    else:
                        print(f"{'-':>12}", end="")
                print()

    print("\n" + "-" * (25 + 12 * len(surveys)))
    print(f"{'Total variables':<25}", end="")
    for name, df in surveys.items():
        print(f"{len(df.columns):>12}", end="")
    print()
    print(f"{'Sample size':<25}", end="")
    for name, df in surveys.items():
        print(f"{len(df):>12,}", end="")
    print("\n")


# =============================================================================
# RECONSTRUCTION APPROACHES
# =============================================================================

def sequential_fusion_large(
    surveys: Dict[str, pd.DataFrame],
    target_n: int,
) -> Tuple[pd.DataFrame, float]:
    """Sequential fusion at scale."""
    print("\n[SEQUENTIAL FUSION] Chaining imputations...")
    start = time.time()

    # Start with ACS (largest, most variables, representative)
    base = surveys["ACS"].sample(n=target_n, replace=True, random_state=42).reset_index(drop=True)

    # Track which variables we have
    have_vars = set(base.columns)

    # Impute from CPS: employment details, wage, UI
    cps = surveys["CPS"]
    cps_targets = ["hours_worked", "weeks_worked", "industry", "occupation",
                   "class_worker", "wage_income", "ui_income"]
    cps_targets = [v for v in cps_targets if v not in have_vars and v in cps.columns]
    if cps_targets:
        cps_conds = [v for v in base.columns if v in cps.columns and v not in cps_targets]
        print(f"  CPS: imputing {cps_targets} from {len(cps_conds)} conditions...")
        qrf = SequentialQRFWithZeroInflation(cps_targets, cps_conds, n_estimators=50, max_depth=8)
        qrf.fit(cps, verbose=False)
        imputed = qrf.generate(base[cps_conds])
        for var in cps_targets:
            base[var] = imputed[var]
            have_vars.add(var)

    # Impute from IRS: all income types
    irs = surveys["IRS"]
    irs_targets = ["self_emp_income", "ss_income", "pension_income", "dividend_income",
                   "interest_income", "rental_income", "alimony_income", "other_income"]
    irs_targets = [v for v in irs_targets if v not in have_vars and v in irs.columns]
    if irs_targets:
        irs_conds = [v for v in base.columns if v in irs.columns and v not in irs_targets]
        print(f"  IRS: imputing {len(irs_targets)} income vars from {len(irs_conds)} conditions...")
        qrf = SequentialQRFWithZeroInflation(irs_targets, irs_conds, n_estimators=50, max_depth=8)
        qrf.fit(irs, verbose=False)
        imputed = qrf.generate(base[irs_conds])
        for var in irs_targets:
            base[var] = imputed[var]
            have_vars.add(var)

    # Impute from SIPP: program participation
    sipp = surveys["SIPP"]
    sipp_targets = ["ssi_income", "snap_benefits", "workers_comp_income", "veterans_benefits"]
    sipp_targets = [v for v in sipp_targets if v not in have_vars and v in sipp.columns]
    if sipp_targets:
        sipp_conds = [v for v in base.columns if v in sipp.columns and v not in sipp_targets]
        print(f"  SIPP: imputing {sipp_targets} from {len(sipp_conds)} conditions...")
        qrf = SequentialQRFWithZeroInflation(sipp_targets, sipp_conds, n_estimators=50, max_depth=8)
        qrf.fit(sipp, verbose=False)
        imputed = qrf.generate(base[sipp_conds])
        for var in sipp_targets:
            base[var] = imputed[var]
            have_vars.add(var)

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s, Variables: {len(base.columns)}")
    return base, elapsed


def stacked_imputation_large(
    surveys: Dict[str, pd.DataFrame],
    target_n: int,
) -> Tuple[pd.DataFrame, float]:
    """Stack all surveys and impute jointly."""
    print("\n[STACKED IMPUTATION] Joint imputation on stacked data...")
    start = time.time()

    # Get all variables
    all_vars = set()
    for df in surveys.values():
        all_vars.update(df.columns)
    all_vars = sorted(all_vars)

    # Stack with missing values
    stacked_rows = []
    for name, df in surveys.items():
        sample_size = min(len(df), target_n // len(surveys))
        sampled = df.sample(n=sample_size, random_state=42)
        for _, row in sampled.iterrows():
            new_row = {v: np.nan for v in all_vars}
            for col in df.columns:
                new_row[col] = row[col]
            stacked_rows.append(new_row)

    stacked = pd.DataFrame(stacked_rows)
    print(f"  Stacked {len(stacked):,} records, {len(all_vars)} variables")

    # Missing rates
    miss_rates = stacked.isna().mean()
    print(f"  Missing rates: min={miss_rates.min():.1%}, max={miss_rates.max():.1%}, mean={miss_rates.mean():.1%}")

    # Impute
    imputer = IterativeImputer(max_iter=5, random_state=42, n_nearest_features=20)
    imputed_arr = imputer.fit_transform(stacked)
    result = pd.DataFrame(imputed_arr, columns=all_vars)

    # Post-process
    for var in all_vars:
        result[var] = result[var].clip(lower=0)

    # Sample to target
    if len(result) > target_n:
        result = result.sample(n=target_n, random_state=42).reset_index(drop=True)

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s")
    return result, elapsed


def unified_microplex_large(
    surveys: Dict[str, pd.DataFrame],
    target_n: int,
) -> Tuple[pd.DataFrame, float]:
    """Train unified microplex on stacked, imputed data."""
    print("\n[UNIFIED MICROPLEX] Single generative model...")
    start = time.time()

    # First, do stacked imputation to get complete training data
    all_vars = set()
    for df in surveys.values():
        all_vars.update(df.columns)
    all_vars = sorted(all_vars)

    # Stack
    stacked_rows = []
    for name, df in surveys.items():
        sample_size = min(len(df), 50000)  # Limit for training efficiency
        sampled = df.sample(n=sample_size, random_state=42)
        for _, row in sampled.iterrows():
            new_row = {v: np.nan for v in all_vars}
            for col in df.columns:
                new_row[col] = row[col]
            stacked_rows.append(new_row)

    stacked = pd.DataFrame(stacked_rows)

    # Impute
    imputer = IterativeImputer(max_iter=5, random_state=42, n_nearest_features=20)
    imputed_arr = imputer.fit_transform(stacked)
    complete = pd.DataFrame(imputed_arr, columns=all_vars)

    # Post-process
    for var in all_vars:
        complete[var] = complete[var].clip(lower=0)

    # Define condition and target variables
    cond_vars = ["age", "is_female", "race", "education", "marital_status",
                 "household_size", "is_employed", "state", "metro_status"]
    target_vars = ["wage_income", "self_emp_income", "ss_income", "ssi_income",
                   "snap_benefits", "ui_income", "pension_income", "dividend_income",
                   "interest_income", "rental_income"]

    cond_vars = [v for v in cond_vars if v in all_vars]
    target_vars = [v for v in target_vars if v in all_vars]

    print(f"  Training microplex: {len(cond_vars)} conditions → {len(target_vars)} targets")

    # Train
    model = Synthesizer(
        target_vars=target_vars,
        condition_vars=cond_vars,
        n_layers=6, hidden_dim=64, zero_inflated=True
    )
    model.fit(complete, epochs=30, batch_size=512, verbose=False)

    # Generate
    result = model.sample(target_n, seed=42)

    elapsed = time.time() - start
    print(f"  Time: {elapsed:.1f}s")
    return result, elapsed


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_large(
    population: pd.DataFrame,
    reconstructed: pd.DataFrame,
    name: str,
) -> Dict:
    """Evaluate reconstruction quality."""
    # Find common variables
    common_vars = [v for v in reconstructed.columns if v in population.columns]

    # Sample from population for comparison
    n_test = min(10000, len(population) // 10)
    test_pop = population.sample(n=n_test, random_state=123)

    # Separate into categories
    demo_vars = ["age", "is_female", "race", "education", "marital_status"]
    income_vars = ["wage_income", "self_emp_income", "ss_income", "pension_income",
                   "dividend_income", "interest_income"]

    demo_vars = [v for v in demo_vars if v in common_vars]
    income_vars = [v for v in income_vars if v in common_vars]

    results = {"method": name, "n_vars": len(common_vars)}

    for grp_name, grp_vars in [("demo", demo_vars), ("income", income_vars)]:
        if grp_vars:
            scaler = StandardScaler()
            pop_norm = scaler.fit_transform(test_pop[grp_vars])

            # Clip reconstructed to reasonable ranges
            recon_clipped = reconstructed[grp_vars].copy()
            for v in grp_vars:
                recon_clipped[v] = recon_clipped[v].clip(
                    lower=test_pop[v].quantile(0.001),
                    upper=test_pop[v].quantile(0.999)
                )
            recon_norm = scaler.transform(recon_clipped)

            # Subsample for MMD computation (expensive)
            if len(pop_norm) > 5000:
                idx = np.random.choice(len(pop_norm), 5000, replace=False)
                pop_norm = pop_norm[idx]
            if len(recon_norm) > 5000:
                idx = np.random.choice(len(recon_norm), 5000, replace=False)
                recon_norm = recon_norm[idx]

            results[f"{grp_name}_mmd"] = compute_mmd(pop_norm, recon_norm)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Configuration
    config = PopulationConfig(n=500000, seed=42)  # 500k for faster testing

    # Generate population
    population = generate_large_population(config)

    # Create surveys
    surveys = create_surveys_large(population)
    print_coverage_matrix(surveys)

    # Target reconstruction size
    target_n = 100000

    results = []

    # 1. Sequential fusion
    try:
        recon, elapsed = sequential_fusion_large(surveys, target_n)
        res = evaluate_large(population, recon, "Sequential Fusion")
        res["time"] = elapsed
        results.append(res)
        print(f"  → Demo MMD: {res.get('demo_mmd', 'N/A'):.4f}, Income MMD: {res.get('income_mmd', 'N/A'):.4f}")
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()

    # 2. Stacked imputation
    try:
        recon, elapsed = stacked_imputation_large(surveys, target_n)
        res = evaluate_large(population, recon, "Stacked Imputation")
        res["time"] = elapsed
        results.append(res)
        print(f"  → Demo MMD: {res.get('demo_mmd', 'N/A'):.4f}, Income MMD: {res.get('income_mmd', 'N/A'):.4f}")
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()

    # 3. Unified microplex
    try:
        recon, elapsed = unified_microplex_large(surveys, target_n)
        res = evaluate_large(population, recon, "Unified Microplex")
        res["time"] = elapsed
        results.append(res)
        print(f"  → Demo MMD: {res.get('demo_mmd', 'N/A'):.4f}, Income MMD: {res.get('income_mmd', 'N/A'):.4f}")
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()

    # 4. Oracle
    print("\n[ORACLE] Direct sample from population...")
    oracle = population.sample(n=target_n, random_state=99).reset_index(drop=True)
    res = evaluate_large(population, oracle, "Oracle")
    res["time"] = 0
    results.append(res)
    print(f"  → Demo MMD: {res.get('demo_mmd', 'N/A'):.4f}, Income MMD: {res.get('income_mmd', 'N/A'):.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: LARGE-SCALE RECONSTRUCTION")
    print("=" * 80)

    df = pd.DataFrame(results)
    print(f"\n{'Method':<25} {'N Vars':>8} {'Demo MMD':>12} {'Income MMD':>12} {'Time':>10}")
    print("-" * 70)
    for _, row in df.iterrows():
        demo = row.get("demo_mmd", float("nan"))
        income = row.get("income_mmd", float("nan"))
        print(f"{row['method']:<25} {row.get('n_vars', 0):>8} {demo:>12.4f} {income:>12.4f} {row['time']:>9.1f}s")

    # Save
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "large_scale_reconstruction.csv", index=False)
    print(f"\nResults saved to {output_dir}/large_scale_reconstruction.csv")
