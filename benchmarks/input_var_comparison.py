"""Comprehensive comparison on INPUT variables only (not derived rules engine outputs).

Target variables are survey-reported inputs that the rules engine uses:
- employment_income (wage_income)
- self_employment_income
- social_security (ss_income)
- unemployment_compensation (uc_income)
- investment_income (interest + dividends)

NOT included (computed by rules engine):
- SNAP benefits
- EITC
- AGI
- Federal tax

Baseline methods added:
- Binning: Discretize conditions, sample from training data in each bin
- Linear Regression + ZI: Two-stage linear model
- Quantile Regression: Predict median with quantile loss
- Statistical Matching: Nearest neighbor from donor pool
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
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

# INPUT variables only (survey-reported, used BY rules engine)
target_vars = [
    "wage_income",          # Employment income
    "self_emp_income",      # Self-employment income
    "ss_income",            # Social Security (reported)
    "uc_income",            # Unemployment compensation (reported)
    "investment_income",    # Interest + dividends
]

# Condition variables (demographics)
condition_vars = ["age", "education", "is_employed", "marital_status"]
all_vars = target_vars + condition_vars

test_conditions = test_data[condition_vars].copy()

print(f"\nTarget variables (inputs only): {target_vars}")
print(f"Condition variables: {condition_vars}")

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
# 1. MICROPLEX (tuned)
# ==============================================================================
print("\n[1/9] microplex (tuned)...")
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
# 2. QRF+ZI (tuned)
# ==============================================================================
print("\n[2/9] QRF+ZI (tuned)...")
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
# 3. QRF (no ZI) - PolicyEngine approach
# ==============================================================================
print("\n[3/9] QRF (no ZI, PolicyEngine)...")
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
# 4. BINNING - Discretize conditions, sample from training within each bin
# ==============================================================================
print("\n[4/9] Binning (discretize + sample)...")
try:
    start = time.time()

    # Create bins for each condition variable
    binned_train = train_data.copy()
    binned_test = test_conditions.copy()

    for var in condition_vars:
        if train_data[var].nunique() <= 5:
            # Already categorical
            binned_train[f"{var}_bin"] = train_data[var]
            binned_test[f"{var}_bin"] = test_conditions[var]
        else:
            # Create quintile bins
            bins = pd.qcut(train_data[var], q=5, labels=False, duplicates='drop')
            binned_train[f"{var}_bin"] = bins
            # Assign test to closest bin edges
            bin_edges = train_data.groupby(bins)[var].mean().values
            test_vals = test_conditions[var].values
            binned_test[f"{var}_bin"] = np.argmin(
                np.abs(test_vals[:, None] - bin_edges[None, :]), axis=1
            )

    bin_cols = [f"{v}_bin" for v in condition_vars]

    # Sample from training data within each bin combination
    synthetic_rows = []
    for _, row in binned_test.iterrows():
        # Find matching bin in training data
        mask = np.ones(len(binned_train), dtype=bool)
        for col in bin_cols:
            mask &= (binned_train[col] == row[col])

        matches = train_data[mask]
        if len(matches) > 0:
            # Random sample from matches
            sampled = matches.sample(n=1, random_state=42).iloc[0]
            synthetic_rows.append(sampled[target_vars].to_dict())
        else:
            # Fallback: random sample from all training
            sampled = train_data.sample(n=1, random_state=42).iloc[0]
            synthetic_rows.append(sampled[target_vars].to_dict())

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
print("\n[5/9] Linear Regression + ZI...")
try:
    start = time.time()

    synthetic_data = test_conditions.copy()
    available_features = list(condition_vars)

    for target in target_vars:
        X_train = train_data[available_features].values
        y_train = train_data[target].values
        X_test = synthetic_data[available_features].values

        # Stage 1: Zero classifier
        y_binary = (y_train > 0).astype(int)
        if y_binary.mean() > 0.01 and y_binary.mean() < 0.99:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_binary)
            p_positive = clf.predict_proba(X_test)[:, 1]
        else:
            p_positive = np.full(len(X_test), y_binary.mean())

        # Stage 2: Linear regression for positive values
        mask = y_train > 0
        predictions = np.zeros(len(X_test))

        if mask.sum() > 10:
            scaler = StandardScaler()
            X_pos = scaler.fit_transform(X_train[mask])
            reg = LinearRegression()
            reg.fit(X_pos, y_train[mask])
            pred_positive = reg.predict(scaler.transform(X_test))

            # Add noise proportional to residual std
            residual_std = np.std(y_train[mask] - reg.predict(X_pos))
            pred_positive += np.random.normal(0, residual_std * 0.5, len(pred_positive))
            pred_positive = np.maximum(pred_positive, 0)
        else:
            pred_positive = np.full(len(X_test), y_train[mask].mean() if mask.sum() > 0 else 0)

        # Sample zeros
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
    import traceback; traceback.print_exc()


# ==============================================================================
# 6. QUANTILE REGRESSION + ZI
# ==============================================================================
print("\n[6/9] Quantile Regression + ZI...")
try:
    start = time.time()

    synthetic_data = test_conditions.copy()
    available_features = list(condition_vars)

    for target in target_vars:
        X_train = train_data[available_features].values
        y_train = train_data[target].values
        X_test = synthetic_data[available_features].values

        # Stage 1: Zero classifier
        y_binary = (y_train > 0).astype(int)
        if y_binary.mean() > 0.01 and y_binary.mean() < 0.99:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_binary)
            p_positive = clf.predict_proba(X_test)[:, 1]
        else:
            p_positive = np.full(len(X_test), y_binary.mean())

        # Stage 2: Quantile regression for positive values
        mask = y_train > 0
        predictions = np.zeros(len(X_test))

        if mask.sum() > 50:
            scaler = StandardScaler()
            X_pos = scaler.fit_transform(X_train[mask])

            # Sample random quantile for each prediction
            quantiles = np.random.uniform(0.1, 0.9, len(X_test))

            # Fit median model (faster than per-sample quantile)
            reg = QuantileRegressor(quantile=0.5, alpha=0.01, solver='highs')
            reg.fit(X_pos, y_train[mask])
            pred_median = reg.predict(scaler.transform(X_test))

            # Add scaled noise to simulate quantile spread
            y_std = np.std(y_train[mask])
            pred_positive = pred_median + (quantiles - 0.5) * y_std
            pred_positive = np.maximum(pred_positive, 0)
        else:
            pred_positive = np.full(len(X_test), y_train[mask].mean() if mask.sum() > 0 else 0)

        # Sample zeros
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
    import traceback; traceback.print_exc()


# ==============================================================================
# 7. STATISTICAL MATCHING (Nearest Neighbor)
# ==============================================================================
print("\n[7/9] Statistical Matching (NN)...")
try:
    start = time.time()

    # Normalize condition variables
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[condition_vars].values)
    X_test = scaler.transform(test_conditions.values)

    # Find nearest neighbor for each test record
    # Use batch processing for memory efficiency
    batch_size = 1000
    matched_indices = []

    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        distances = cdist(batch, X_train, metric='euclidean')
        nearest = np.argmin(distances, axis=1)
        matched_indices.extend(nearest)

    # Get matched records from training data
    synthetic = train_data.iloc[matched_indices][target_vars].copy().reset_index(drop=True)
    synthetic[condition_vars] = test_conditions.values

    train_time = time.time() - start
    res = evaluate(synthetic, "StatMatch (NN)")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Time={train_time:.1f}s")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback; traceback.print_exc()


# ==============================================================================
# 8. XGBoost + ZI
# ==============================================================================
print("\n[8/9] XGBoost + ZI...")
try:
    import xgboost as xgb

    start = time.time()

    synthetic_data = test_conditions.copy()
    available_features = list(condition_vars)

    for target in target_vars:
        X_train = train_data[available_features].values
        y_train = train_data[target].values
        X_test = synthetic_data[available_features].values

        # Zero classifier
        y_binary = (y_train > 0).astype(int)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_binary)
        p_positive = clf.predict_proba(X_test)[:, 1]

        # Regressor for positive values
        mask = y_train > 0
        if mask.sum() > 10:
            reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, verbosity=0, random_state=42)
            reg.fit(X_train[mask], y_train[mask])
            pred_positive = reg.predict(X_test)
        else:
            pred_positive = np.full(len(X_test), y_train[mask].mean() if mask.sum() > 0 else 0)

        # Sample zeros
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
print("\n[9/9] CT-GAN...")
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
# SUMMARY
# ==============================================================================
print("\n" + "=" * 90)
print("INPUT VARIABLES COMPARISON RESULTS")
print("=" * 90)
print(f"\nTarget variables: {target_vars}")
print(f"(These are survey INPUTS - not derived from rules engine)")

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
output_path = Path(__file__).parent / "results" / "input_var_comparison.csv"
df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
