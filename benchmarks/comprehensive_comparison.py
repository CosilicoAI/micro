"""Comprehensive comparison of synthesis methods on CPS-like data."""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

from run_cps_benchmark import generate_cps_like_data
from compare_qrf import SequentialQRFWithZeroInflation
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

target_vars = ["wage_income", "self_emp_income", "ssi_income", "uc_income",
               "snap_benefit", "eitc", "agi", "federal_tax"]
condition_vars = ["age", "education", "is_employed", "marital_status"]
all_vars = target_vars + condition_vars

test_conditions = test_data[condition_vars].copy()

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

# 1. microplex (tuned)
print("\n[1/7] microplex (tuned: L=8, H=128, epochs=100)...")
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
    res = evaluate(synthetic, "microplex (tuned)")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Coverage={res['coverage']:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# 2. QRF+ZI (tuned)
print("\n[2/7] QRF+ZI (tuned: n=200, depth=15)...")
try:
    start = time.time()
    model = SequentialQRFWithZeroInflation(
        target_vars, condition_vars,
        n_estimators=200, max_depth=15
    )
    model.fit(train_data, verbose=False)
    synthetic = model.generate(test_conditions)
    train_time = time.time() - start
    res = evaluate(synthetic, "QRF+ZI (tuned)")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Coverage={res['coverage']:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# 3. TabPFN-based
print("\n[3/7] TabPFN-based synthesis...")
try:
    from tabpfn import TabPFNRegressor

    start = time.time()
    synthetic_rows = []

    # TabPFN works on small context, so we sample training data
    train_sample = train_data.sample(n=min(1000, len(train_data)), random_state=42)

    for target in target_vars:
        X_train = train_sample[condition_vars].values
        y_train = train_sample[target].values
        X_test = test_conditions.values

        model = TabPFNRegressor(device="cpu", n_estimators=4)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        synthetic_rows.append(pd.Series(pred, name=target))

    synthetic = pd.concat(synthetic_rows, axis=1)
    synthetic[condition_vars] = test_conditions.values
    train_time = time.time() - start

    res = evaluate(synthetic, "TabPFN")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Coverage={res['coverage']:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# 4. CT-GAN
print("\n[4/7] CT-GAN...")
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    start = time.time()

    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data[all_vars])

    model = CTGANSynthesizer(metadata, epochs=50, verbose=False)
    model.fit(train_data[all_vars])

    # Generate and filter to match test conditions
    synthetic = model.sample(len(test_conditions))
    train_time = time.time() - start

    res = evaluate(synthetic, "CT-GAN")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Coverage={res['coverage']:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# 5. TVAE
print("\n[5/7] TVAE...")
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
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Coverage={res['coverage']:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# 6. Gaussian Copula
print("\n[6/7] Gaussian Copula...")
try:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata

    start = time.time()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data[all_vars])

    model = GaussianCopulaSynthesizer(metadata)
    model.fit(train_data[all_vars])

    synthetic = model.sample(len(test_conditions))
    train_time = time.time() - start

    res = evaluate(synthetic, "Gaussian Copula")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Coverage={res['coverage']:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# 7. XGBoost Sequential (similar to QRF but with XGBoost)
print("\n[7/7] XGBoost Sequential + Zero-Inflation...")
try:
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression

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
            reg = xgb.XGBRegressor(
                n_estimators=100, max_depth=6,
                learning_rate=0.1, random_state=42, verbosity=0
            )
            reg.fit(X_train[mask], y_train[mask])
            pred_positive = reg.predict(X_test)
        else:
            pred_positive = np.full(len(X_test), y_train[mask].mean() if mask.sum() > 0 else 0)

        # Sample zeros
        is_positive = np.random.random(len(X_test)) < p_positive
        predictions = np.where(is_positive, pred_positive, 0)
        predictions = np.maximum(predictions, 0)  # Clip negatives

        synthetic_data[target] = predictions
        available_features.append(target)

    train_time = time.time() - start

    res = evaluate(synthetic_data, "XGBoost+ZI")
    res["time"] = train_time
    results.append(res)
    print(f"  ✓ MMD={res['mmd']:.4f}, Energy={res['energy_dist']:.4f}, Coverage={res['coverage']:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Summary
print("\n" + "=" * 90)
print("COMPREHENSIVE COMPARISON RESULTS")
print("=" * 90)
df = pd.DataFrame(results)
df = df.sort_values("mmd")

# Format nicely
print(f"\n{'Method':<25} {'MMD':>10} {'Energy':>10} {'Auth':>10} {'Coverage':>10} {'Time(s)':>10}")
print("-" * 75)
for _, row in df.iterrows():
    print(f"{row['method']:<25} {row['mmd']:>10.4f} {row['energy_dist']:>10.4f} "
          f"{row['authenticity']:>10.4f} {row['coverage']:>10.4f} {row['time']:>10.1f}")

print("\n" + "=" * 90)
print("WINNER BY METRIC (lower is better except for context)")
print("=" * 90)
print(f"Best MMD (joint dist):      {df.loc[df['mmd'].idxmin(), 'method']}")
print(f"Best Energy Distance:       {df.loc[df['energy_dist'].idxmin(), 'method']}")
print(f"Best Authenticity:          {df.loc[df['authenticity'].idxmin(), 'method']}")
print(f"Best Coverage:              {df.loc[df['coverage'].idxmin(), 'method']}")
print(f"Fastest:                    {df.loc[df['time'].idxmin(), 'method']}")

# Save results
output_path = Path(__file__).parent / "results" / "comprehensive_comparison.csv"
df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
