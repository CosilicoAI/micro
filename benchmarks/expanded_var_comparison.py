"""Expanded comparison with more variables (11 conditions, 6 targets).

Target variables (survey-reported inputs):
- wage_income, self_emp_income, ss_income, ssi_income, uc_income, investment_income

Condition variables (11 total):
- Demographics: age, is_female, education, marital_status, household_size, n_children
- Geography: state_fips, region
- Employment: is_employed, is_fulltime, is_self_employed
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

from run_cps_benchmark import generate_cps_like_data
from compare_qrf import SequentialQRF, SequentialQRFWithZeroInflation
from multivariate_metrics import (
    compute_mmd, compute_energy_distance, normalize_data,
    compute_authenticity_distance, compute_coverage_distance
)
from microplex import Synthesizer

# Generate data
print("Generating CPS-like data (20k train, 5k test)...")
full_data = generate_cps_like_data(25000, seed=42)
train_data = full_data.iloc[:20000].copy()
test_data = full_data.iloc[20000:].copy()

# EXPANDED: 6 target variables (all survey-reported inputs)
target_vars = [
    "wage_income",          # Employment income
    "self_emp_income",      # Self-employment income
    "ss_income",            # Social Security (reported)
    "ssi_income",           # SSI (reported)
    "uc_income",            # Unemployment compensation
    "investment_income",    # Interest + dividends
]

# EXPANDED: 11 condition variables
condition_vars = [
    # Demographics
    "age", "is_female", "education", "marital_status",
    "household_size", "n_children",
    # Geography
    "state_fips", "region",
    # Employment
    "is_employed", "is_fulltime", "is_self_employed",
]

all_vars = target_vars + condition_vars
test_conditions = test_data[condition_vars].copy()

print(f"\nTarget variables ({len(target_vars)}): {target_vars}")
print(f"Condition variables ({len(condition_vars)}): {condition_vars}")

# Check zero fractions
print("\nZero fractions in training data:")
for var in target_vars:
    zero_frac = (train_data[var] == 0).mean()
    print(f"  {var}: {zero_frac:.1%}")


def evaluate(synthetic, name):
    """Compute all metrics for a synthetic dataset."""
    train_norm, test_norm, synth_norm, _ = normalize_data(
        train_data, test_data, synthetic, target_vars
    )
    mmd = compute_mmd(test_norm, synth_norm)
    energy = compute_energy_distance(test_norm, synth_norm)
    auth = compute_authenticity_distance(synth_norm, test_norm)
    cov = compute_coverage_distance(test_norm, synth_norm)
    return {
        "method": name,
        "mmd": mmd,
        "energy_dist": energy,
        "authenticity": auth["mean"],
        "coverage": cov["mean"],
    }


results = []


# ==============================================================================
# 1. MICROPLEX
# ==============================================================================
print("\n[1/10] microplex (tuned)...")
try:
    start = time.time()
    model = Synthesizer(
        target_vars=target_vars,
        condition_vars=condition_vars,
        n_layers=8, hidden_dim=128, zero_inflated=True,
    )
    model.fit(train_data, epochs=100, batch_size=256, verbose=False)
    synthetic = model.generate(test_conditions)
    train_time = time.time() - start
    res = evaluate(synthetic, "microplex")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")


# ==============================================================================
# 2. QRF+ZI
# ==============================================================================
print("\n[2/10] QRF+ZI (tuned)...")
try:
    start = time.time()
    model = SequentialQRFWithZeroInflation(
        target_vars, condition_vars,
        n_estimators=200, max_depth=15
    )
    model.fit(train_data, verbose=False)
    synthetic = model.generate(test_conditions)
    train_time = time.time() - start
    res = evaluate(synthetic, "QRF+ZI")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")


# ==============================================================================
# 3. QRF (no ZI) - PolicyEngine
# ==============================================================================
print("\n[3/10] QRF (no ZI, PolicyEngine)...")
try:
    start = time.time()
    model = SequentialQRF(
        target_vars, condition_vars,
        n_estimators=200, max_depth=15
    )
    model.fit(train_data, verbose=False)
    synthetic = model.generate(test_conditions)
    train_time = time.time() - start
    res = evaluate(synthetic, "QRF (no ZI)")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")


# ==============================================================================
# 4. BINNING
# ==============================================================================
print("\n[4/10] Binning (discretize + sample)...")
try:
    start = time.time()

    binned_train = train_data.copy()
    binned_test = test_conditions.copy()

    for var in condition_vars:
        if train_data[var].nunique() <= 10:
            binned_train[f"{var}_bin"] = train_data[var]
            binned_test[f"{var}_bin"] = test_conditions[var]
        else:
            bins = pd.qcut(train_data[var], q=5, labels=False, duplicates='drop')
            binned_train[f"{var}_bin"] = bins
            bin_edges = train_data.groupby(bins)[var].mean().values
            test_vals = test_conditions[var].values
            binned_test[f"{var}_bin"] = np.argmin(
                np.abs(test_vals[:, None] - bin_edges[None, :]), axis=1
            )

    bin_cols = [f"{v}_bin" for v in condition_vars]

    synthetic_rows = []
    rng = np.random.default_rng(42)
    for _, row in binned_test.iterrows():
        mask = np.ones(len(binned_train), dtype=bool)
        for col in bin_cols:
            mask &= (binned_train[col] == row[col])

        matches = train_data[mask]
        if len(matches) > 0:
            idx = rng.choice(len(matches))
            synthetic_rows.append(matches.iloc[idx][target_vars].to_dict())
        else:
            idx = rng.choice(len(train_data))
            synthetic_rows.append(train_data.iloc[idx][target_vars].to_dict())

    synthetic = pd.DataFrame(synthetic_rows)
    synthetic[condition_vars] = test_conditions.values

    train_time = time.time() - start
    res = evaluate(synthetic, "Binning")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback; traceback.print_exc()


# ==============================================================================
# 5. LINEAR REGRESSION + ZI
# ==============================================================================
print("\n[5/10] Linear Regression + ZI...")
try:
    start = time.time()

    synthetic_data = test_conditions.copy()
    available_features = list(condition_vars)

    for target in target_vars:
        X_train = train_data[available_features].values
        y_train = train_data[target].values
        X_test = synthetic_data[available_features].values

        y_binary = (y_train > 0).astype(int)
        if 0.01 < y_binary.mean() < 0.99:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_binary)
            p_positive = clf.predict_proba(X_test)[:, 1]
        else:
            p_positive = np.full(len(X_test), y_binary.mean())

        mask = y_train > 0
        predictions = np.zeros(len(X_test))

        if mask.sum() > 10:
            scaler = StandardScaler()
            X_pos = scaler.fit_transform(X_train[mask])
            reg = LinearRegression()
            reg.fit(X_pos, y_train[mask])
            pred_positive = reg.predict(scaler.transform(X_test))
            residual_std = np.std(y_train[mask] - reg.predict(X_pos))
            pred_positive += np.random.normal(0, residual_std * 0.5, len(pred_positive))
            pred_positive = np.maximum(pred_positive, 0)
        else:
            pred_positive = np.full(len(X_test), y_train[mask].mean() if mask.sum() > 0 else 0)

        is_positive = np.random.random(len(X_test)) < p_positive
        predictions[is_positive] = pred_positive[is_positive]

        synthetic_data[target] = predictions
        available_features.append(target)

    train_time = time.time() - start
    res = evaluate(synthetic_data, "Linear+ZI")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")


# ==============================================================================
# 6. QUANTILE REGRESSION + ZI
# ==============================================================================
print("\n[6/10] Quantile Regression + ZI...")
try:
    start = time.time()

    synthetic_data = test_conditions.copy()
    available_features = list(condition_vars)

    for target in target_vars:
        X_train = train_data[available_features].values
        y_train = train_data[target].values
        X_test = synthetic_data[available_features].values

        y_binary = (y_train > 0).astype(int)
        if 0.01 < y_binary.mean() < 0.99:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_binary)
            p_positive = clf.predict_proba(X_test)[:, 1]
        else:
            p_positive = np.full(len(X_test), y_binary.mean())

        mask = y_train > 0
        predictions = np.zeros(len(X_test))

        if mask.sum() > 50:
            scaler = StandardScaler()
            X_pos = scaler.fit_transform(X_train[mask])
            reg = QuantileRegressor(quantile=0.5, alpha=0.01, solver='highs')
            reg.fit(X_pos, y_train[mask])
            pred_median = reg.predict(scaler.transform(X_test))
            y_std = np.std(y_train[mask])
            quantiles = np.random.uniform(0.1, 0.9, len(X_test))
            pred_positive = pred_median + (quantiles - 0.5) * y_std
            pred_positive = np.maximum(pred_positive, 0)
        else:
            pred_positive = np.full(len(X_test), y_train[mask].mean() if mask.sum() > 0 else 0)

        is_positive = np.random.random(len(X_test)) < p_positive
        predictions[is_positive] = pred_positive[is_positive]

        synthetic_data[target] = predictions
        available_features.append(target)

    train_time = time.time() - start
    res = evaluate(synthetic_data, "Quantile+ZI")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")


# ==============================================================================
# 7. NND.HOTDECK (py-statmatch)
# ==============================================================================
print("\n[7/10] NND.hotdeck (py-statmatch)...")
try:
    from statmatch import nnd_hotdeck

    start = time.time()

    scaler = StandardScaler()
    train_normalized = train_data.copy()
    test_normalized = test_conditions.copy()

    for var in condition_vars:
        train_normalized[f"{var}_norm"] = scaler.fit_transform(train_data[[var]])
        test_normalized[f"{var}_norm"] = scaler.transform(test_conditions[[var]])

    norm_vars = [f"{v}_norm" for v in condition_vars]

    result = nnd_hotdeck(
        data_rec=test_normalized.reset_index(drop=True),
        data_don=train_normalized.reset_index(drop=True),
        match_vars=norm_vars,
        dist_fun="euclidean",
    )

    matched_indices = result["noad.index"]
    synthetic = train_data.iloc[matched_indices][target_vars].copy().reset_index(drop=True)
    synthetic[condition_vars] = test_conditions.values

    train_time = time.time() - start
    res = evaluate(synthetic, "NND.hotdeck")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")


# ==============================================================================
# 7b. NND.HOTDECK CONSTRAINED (k=5)
# ==============================================================================
print("\n[7b/10] NND.hotdeck constrained (k=5)...")
try:
    start = time.time()

    result = nnd_hotdeck(
        data_rec=test_normalized.reset_index(drop=True),
        data_don=train_normalized.reset_index(drop=True),
        match_vars=norm_vars,
        dist_fun="euclidean",
        k=5,
        constr_alg="hungarian",
    )

    matched_indices = result["noad.index"]
    synthetic = train_data.iloc[matched_indices][target_vars].copy().reset_index(drop=True)
    synthetic[condition_vars] = test_conditions.values

    train_time = time.time() - start
    res = evaluate(synthetic, "NND.hotdeck (k=5)")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")


# ==============================================================================
# 8. XGBoost + ZI
# ==============================================================================
print("\n[8/10] XGBoost + ZI...")
try:
    import xgboost as xgb

    start = time.time()

    synthetic_data = test_conditions.copy()
    available_features = list(condition_vars)

    for target in target_vars:
        X_train = train_data[available_features].values
        y_train = train_data[target].values
        X_test = synthetic_data[available_features].values

        y_binary = (y_train > 0).astype(int)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_binary)
        p_positive = clf.predict_proba(X_test)[:, 1]

        mask = y_train > 0
        if mask.sum() > 10:
            reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, verbosity=0, random_state=42)
            reg.fit(X_train[mask], y_train[mask])
            pred_positive = reg.predict(X_test)
        else:
            pred_positive = np.full(len(X_test), y_train[mask].mean() if mask.sum() > 0 else 0)

        is_positive = np.random.random(len(X_test)) < p_positive
        predictions = np.where(is_positive, np.maximum(pred_positive, 0), 0)

        synthetic_data[target] = predictions
        available_features.append(target)

    train_time = time.time() - start
    res = evaluate(synthetic_data, "XGBoost+ZI")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")


# ==============================================================================
# 9. CT-GAN
# ==============================================================================
print("\n[9/10] CT-GAN...")
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    start = time.time()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data[all_vars])

    model = CTGANSynthesizer(metadata, epochs=50, verbose=False)
    model.fit(train_data[all_vars])

    synthetic = model.sample(len(test_conditions))
    train_time = time.time() - start

    res = evaluate(synthetic, "CT-GAN")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")


# ==============================================================================
# 10. TVAE
# ==============================================================================
print("\n[10/10] TVAE...")
try:
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import SingleTableMetadata

    start = time.time()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data[all_vars])

    model = TVAESynthesizer(metadata, epochs=50, verbose=False)
    model.fit(train_data[all_vars])

    synthetic = model.sample(len(test_conditions))
    train_time = time.time() - start

    res = evaluate(synthetic, "TVAE")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 90)
print("EXPANDED VARIABLE COMPARISON RESULTS")
print("=" * 90)
print(f"\nTarget variables ({len(target_vars)}): {target_vars}")
print(f"Condition variables ({len(condition_vars)}): {condition_vars}")

df = pd.DataFrame(results)
df = df.sort_values("mmd")

print(f"\n{'Method':<20} {'MMD':>10} {'Energy':>10} {'Auth':>10} {'Coverage':>10} {'Time(s)':>10}")
print("-" * 70)
for _, row in df.iterrows():
    print(f"{row['method']:<20} {row['mmd']:>10.4f} {row['energy_dist']:>10.4f} "
          f"{row['authenticity']:>10.4f} {row['coverage']:>10.4f} {row['time']:>10.1f}")

print("\n" + "=" * 90)
print("WINNER BY METRIC (lower is better)")
print("=" * 90)
print(f"Best MMD (joint dist):      {df.loc[df['mmd'].idxmin(), 'method']}")
print(f"Best Energy Distance:       {df.loc[df['energy_dist'].idxmin(), 'method']}")
print(f"Best Authenticity:          {df.loc[df['authenticity'].idxmin(), 'method']}")
print(f"Best Coverage:              {df.loc[df['coverage'].idxmin(), 'method']}")
print(f"Fastest:                    {df.loc[df['time'].idxmin(), 'method']}")

# Save results
output_path = Path(__file__).parent / "results" / "expanded_var_comparison.csv"
df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
