"""Compare IMPUTATION vs FULL SYNTHESIS modes.

IMPUTATION: Real conditions → Synthesized targets
- Use case: Enhance existing CPS with missing variables
- Conditions come from holdout, evaluate target quality

FULL SYNTHESIS: Synthesized conditions + targets
- Use case: Generate entirely new microdata
- Evaluate both condition AND target distributions

Also categorizes methods by how they use conditions:
- FILTERING: Find similar records, copy targets (NND.hotdeck, Binning)
- PREDICTION: Model f(conditions) → targets (QRF, XGBoost, Linear)
- JOINT: Model P(conditions, targets) jointly (microplex, CT-GAN)
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

from run_cps_benchmark import generate_cps_like_data
from compare_qrf import SequentialQRFWithZeroInflation
from multivariate_metrics import (
    compute_mmd, compute_energy_distance, normalize_data,
)
from microplex import Synthesizer

# Generate data
print("Generating CPS-like data (20k train, 5k test)...")
full_data = generate_cps_like_data(25000, seed=42)
train_data = full_data.iloc[:20000].copy()
test_data = full_data.iloc[20000:].copy()

# Variables
target_vars = ["wage_income", "self_emp_income", "ss_income", "uc_income", "investment_income"]
condition_vars = ["age", "education", "is_employed", "marital_status", "is_female", "household_size"]
all_vars = condition_vars + target_vars

print(f"\nCondition vars ({len(condition_vars)}): {condition_vars}")
print(f"Target vars ({len(target_vars)}): {target_vars}")


def evaluate_targets(synthetic, test_data, name):
    """Evaluate target variable quality only."""
    train_norm, test_norm, synth_norm, _ = normalize_data(
        train_data, test_data, synthetic, target_vars
    )
    return {
        "method": name,
        "target_mmd": compute_mmd(test_norm, synth_norm),
        "target_energy": compute_energy_distance(test_norm, synth_norm),
    }


def evaluate_full(synthetic, test_data, name):
    """Evaluate both conditions AND targets."""
    # Targets
    train_norm_t, test_norm_t, synth_norm_t, _ = normalize_data(
        train_data, test_data, synthetic, target_vars
    )
    target_mmd = compute_mmd(test_norm_t, synth_norm_t)

    # Conditions
    train_norm_c, test_norm_c, synth_norm_c, _ = normalize_data(
        train_data, test_data, synthetic, condition_vars
    )
    cond_mmd = compute_mmd(test_norm_c, synth_norm_c)

    # Joint (all vars)
    train_norm_j, test_norm_j, synth_norm_j, _ = normalize_data(
        train_data, test_data, synthetic, all_vars
    )
    joint_mmd = compute_mmd(test_norm_j, synth_norm_j)

    return {
        "method": name,
        "cond_mmd": cond_mmd,
        "target_mmd": target_mmd,
        "joint_mmd": joint_mmd,
    }


# ==============================================================================
# MODE 1: IMPUTATION (real conditions → synthetic targets)
# ==============================================================================
print("\n" + "=" * 80)
print("MODE 1: IMPUTATION (real conditions → synthetic targets)")
print("=" * 80)

test_conditions = test_data[condition_vars].copy()
imputation_results = []

# --- FILTERING METHODS ---
print("\n--- FILTERING METHODS (copy from similar donors) ---")

# NND.hotdeck
print("\n[1] NND.hotdeck (FILTERING)...")
try:
    from statmatch import nnd_hotdeck

    start = time.time()
    scaler = StandardScaler()
    train_norm = train_data.copy()
    test_norm = test_conditions.copy()

    for var in condition_vars:
        train_norm[f"{var}_n"] = scaler.fit_transform(train_data[[var]])
        test_norm[f"{var}_n"] = scaler.transform(test_conditions[[var]])

    norm_vars = [f"{v}_n" for v in condition_vars]
    result = nnd_hotdeck(
        data_rec=test_norm.reset_index(drop=True),
        data_don=train_norm.reset_index(drop=True),
        match_vars=norm_vars,
        dist_fun="euclidean",
    )

    synthetic = train_data.iloc[result["noad.index"]][target_vars].copy().reset_index(drop=True)
    synthetic[condition_vars] = test_conditions.values

    res = evaluate_targets(synthetic, test_data, "NND.hotdeck (FILTER)")
    res["time"] = time.time() - start
    res["type"] = "filtering"
    imputation_results.append(res)
    print(f"  ✓ Target MMD={res['target_mmd']:.4f}")
except Exception as e:
    print(f"  ✗ {e}")

# Binning
print("\n[2] Binning (FILTERING)...")
try:
    start = time.time()

    binned_train = train_data.copy()
    binned_test = test_conditions.copy()

    for var in condition_vars:
        if train_data[var].nunique() <= 5:
            binned_train[f"{var}_bin"] = train_data[var]
            binned_test[f"{var}_bin"] = test_conditions[var]
        else:
            bins = pd.qcut(train_data[var], q=5, labels=False, duplicates='drop')
            binned_train[f"{var}_bin"] = bins
            bin_edges = train_data.groupby(bins)[var].mean().values
            binned_test[f"{var}_bin"] = np.argmin(
                np.abs(test_conditions[var].values[:, None] - bin_edges[None, :]), axis=1
            )

    bin_cols = [f"{v}_bin" for v in condition_vars]
    rng = np.random.default_rng(42)

    synthetic_rows = []
    for _, row in binned_test.iterrows():
        mask = np.ones(len(binned_train), dtype=bool)
        for col in bin_cols:
            mask &= (binned_train[col] == row[col])
        matches = train_data[mask]
        if len(matches) > 0:
            synthetic_rows.append(matches.iloc[rng.choice(len(matches))][target_vars].to_dict())
        else:
            synthetic_rows.append(train_data.iloc[rng.choice(len(train_data))][target_vars].to_dict())

    synthetic = pd.DataFrame(synthetic_rows)
    synthetic[condition_vars] = test_conditions.values

    res = evaluate_targets(synthetic, test_data, "Binning (FILTER)")
    res["time"] = time.time() - start
    res["type"] = "filtering"
    imputation_results.append(res)
    print(f"  ✓ Target MMD={res['target_mmd']:.4f}")
except Exception as e:
    print(f"  ✗ {e}")

# --- PREDICTION METHODS ---
print("\n--- PREDICTION METHODS (f(conditions) → targets) ---")

# QRF+ZI
print("\n[3] QRF+ZI (PREDICTION)...")
try:
    start = time.time()
    model = SequentialQRFWithZeroInflation(target_vars, condition_vars, n_estimators=100, max_depth=10)
    model.fit(train_data, verbose=False)
    synthetic = model.generate(test_conditions)

    res = evaluate_targets(synthetic, test_data, "QRF+ZI (PREDICT)")
    res["time"] = time.time() - start
    res["type"] = "prediction"
    imputation_results.append(res)
    print(f"  ✓ Target MMD={res['target_mmd']:.4f}")
except Exception as e:
    print(f"  ✗ {e}")

# --- JOINT METHODS ---
print("\n--- JOINT METHODS (model P(cond, target) jointly) ---")

# microplex
print("\n[4] microplex (JOINT)...")
try:
    start = time.time()
    model = Synthesizer(
        target_vars=target_vars,
        condition_vars=condition_vars,
        n_layers=6, hidden_dim=64, zero_inflated=True,
    )
    model.fit(train_data, epochs=50, batch_size=256, verbose=False)
    synthetic = model.generate(test_conditions)

    res = evaluate_targets(synthetic, test_data, "microplex (JOINT)")
    res["time"] = time.time() - start
    res["type"] = "joint"
    imputation_results.append(res)
    print(f"  ✓ Target MMD={res['target_mmd']:.4f}")
except Exception as e:
    print(f"  ✗ {e}")

# CT-GAN (joint but we only use targets)
print("\n[5] CT-GAN (JOINT)...")
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    start = time.time()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data[all_vars])
    model = CTGANSynthesizer(metadata, epochs=30, verbose=False)
    model.fit(train_data[all_vars])

    # For imputation mode, we sample and take targets
    # (CT-GAN doesn't do conditional generation well)
    synthetic_full = model.sample(len(test_conditions))
    synthetic = synthetic_full[target_vars].copy()
    synthetic[condition_vars] = test_conditions.values

    res = evaluate_targets(synthetic, test_data, "CT-GAN (JOINT*)")
    res["time"] = time.time() - start
    res["type"] = "joint"
    imputation_results.append(res)
    print(f"  ✓ Target MMD={res['target_mmd']:.4f}")
    print("    * Note: CT-GAN doesn't condition on inputs, just samples")
except Exception as e:
    print(f"  ✗ {e}")


# ==============================================================================
# MODE 2: FULL SYNTHESIS (generate both conditions AND targets)
# ==============================================================================
print("\n" + "=" * 80)
print("MODE 2: FULL SYNTHESIS (generate conditions + targets)")
print("=" * 80)

full_synthesis_results = []

# microplex (can only do imputation, not full synthesis)
print("\n[1] microplex - N/A for full synthesis (requires conditions)")

# CT-GAN
print("\n[2] CT-GAN (JOINT - true full synthesis)...")
try:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    start = time.time()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data[all_vars])
    model = CTGANSynthesizer(metadata, epochs=50, verbose=False)
    model.fit(train_data[all_vars])
    synthetic = model.sample(len(test_data))

    res = evaluate_full(synthetic, test_data, "CT-GAN")
    res["time"] = time.time() - start
    full_synthesis_results.append(res)
    print(f"  ✓ Cond MMD={res['cond_mmd']:.4f}, Target MMD={res['target_mmd']:.4f}, Joint MMD={res['joint_mmd']:.4f}")
except Exception as e:
    print(f"  ✗ {e}")

# TVAE
print("\n[3] TVAE (JOINT - true full synthesis)...")
try:
    from sdv.single_table import TVAESynthesizer
    from sdv.metadata import SingleTableMetadata

    start = time.time()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data[all_vars])
    model = TVAESynthesizer(metadata, epochs=50, verbose=False)
    model.fit(train_data[all_vars])
    synthetic = model.sample(len(test_data))

    res = evaluate_full(synthetic, test_data, "TVAE")
    res["time"] = time.time() - start
    full_synthesis_results.append(res)
    print(f"  ✓ Cond MMD={res['cond_mmd']:.4f}, Target MMD={res['target_mmd']:.4f}, Joint MMD={res['joint_mmd']:.4f}")
except Exception as e:
    print(f"  ✗ {e}")

# Gaussian Copula
print("\n[4] Gaussian Copula (JOINT - true full synthesis)...")
try:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata

    start = time.time()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data[all_vars])
    model = GaussianCopulaSynthesizer(metadata)
    model.fit(train_data[all_vars])
    synthetic = model.sample(len(test_data))

    res = evaluate_full(synthetic, test_data, "Gaussian Copula")
    res["time"] = time.time() - start
    full_synthesis_results.append(res)
    print(f"  ✓ Cond MMD={res['cond_mmd']:.4f}, Target MMD={res['target_mmd']:.4f}, Joint MMD={res['joint_mmd']:.4f}")
except Exception as e:
    print(f"  ✗ {e}")


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY: IMPUTATION MODE (real conditions → synthetic targets)")
print("=" * 80)

df_imp = pd.DataFrame(imputation_results).sort_values("target_mmd")
print(f"\n{'Method':<25} {'Type':<12} {'Target MMD':>12} {'Time':>8}")
print("-" * 60)
for _, row in df_imp.iterrows():
    print(f"{row['method']:<25} {row['type']:<12} {row['target_mmd']:>12.4f} {row['time']:>7.1f}s")

print("\n" + "=" * 80)
print("SUMMARY: FULL SYNTHESIS MODE (generate everything)")
print("=" * 80)

if full_synthesis_results:
    df_full = pd.DataFrame(full_synthesis_results).sort_values("joint_mmd")
    print(f"\n{'Method':<20} {'Cond MMD':>10} {'Target MMD':>12} {'Joint MMD':>12} {'Time':>8}")
    print("-" * 65)
    for _, row in df_full.iterrows():
        print(f"{row['method']:<20} {row['cond_mmd']:>10.4f} {row['target_mmd']:>12.4f} {row['joint_mmd']:>12.4f} {row['time']:>7.1f}s")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("""
METHOD TYPES:
  FILTERING: Copy real records from similar donors (NND.hotdeck, Binning)
    + Perfect joint distributions within records
    + Fast, simple
    - Can't generalize beyond training data
    - Privacy risk (copies real records)

  PREDICTION: Build model f(conditions) → targets (QRF, XGBoost, Linear)
    + Can extrapolate to new conditions
    + Flexible, well-understood
    - May break joint structure between targets
    - Sequential chaining can propagate errors

  JOINT: Model full P(conditions, targets) (microplex, CT-GAN, TVAE)
    + Preserves joint structure
    + Can do full synthesis
    - Harder to train
    - May struggle with high-dimensional conditioning

SYNTHESIS MODES:
  IMPUTATION: Use when you have real demographics, need synthetic targets
    → Filtering methods (NND.hotdeck) excel here

  FULL SYNTHESIS: Use when you need entirely synthetic microdata
    → Joint methods (CT-GAN, TVAE) required
    → microplex currently imputation-only
""")

# Save results
output_dir = Path(__file__).parent / "results"
pd.DataFrame(imputation_results).to_csv(output_dir / "imputation_comparison.csv", index=False)
if full_synthesis_results:
    pd.DataFrame(full_synthesis_results).to_csv(output_dir / "full_synthesis_comparison.csv", index=False)
print(f"\nResults saved to {output_dir}/")
