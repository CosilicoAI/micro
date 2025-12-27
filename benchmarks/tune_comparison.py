"""Quick hyperparameter comparison for microplex vs QRF+ZI."""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_cps_benchmark import generate_cps_like_data
from compare_qrf import SequentialQRFWithZeroInflation
from multivariate_metrics import compute_mmd, compute_energy_distance, normalize_data
from microplex import Synthesizer

# Generate data
print("Generating data...")
full_data = generate_cps_like_data(25000, seed=42)
train_data = full_data.iloc[:20000].copy()
test_data = full_data.iloc[20000:].copy()

target_vars = ["wage_income", "self_emp_income", "ssi_income", "uc_income",
               "snap_benefit", "eitc", "agi", "federal_tax"]
condition_vars = ["age", "education", "is_employed", "marital_status"]
test_conditions = test_data[condition_vars].copy()

results = []

# QRF+ZI configurations
qrf_configs = [
    {"n_estimators": 100, "max_depth": 10},   # baseline
    {"n_estimators": 200, "max_depth": 10},   # more trees
    {"n_estimators": 100, "max_depth": 15},   # deeper
    {"n_estimators": 200, "max_depth": 15},   # both
]

print("\n=== QRF+ZI TUNING ===")
for cfg in qrf_configs:
    name = f"QRF+ZI(n={cfg['n_estimators']},d={cfg['max_depth']})"
    print(f"\n{name}...")

    start = time.time()
    model = SequentialQRFWithZeroInflation(
        target_vars, condition_vars,
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"]
    )
    model.fit(train_data, verbose=False)
    train_time = time.time() - start

    synthetic = model.generate(test_conditions)

    # Compute metrics
    train_norm, test_norm, synth_norm, _ = normalize_data(
        train_data, test_data, synthetic, target_vars
    )
    mmd = compute_mmd(test_norm, synth_norm)
    energy = compute_energy_distance(test_norm, synth_norm)

    print(f"  MMD={mmd:.4f}, Energy={energy:.4f}, time={train_time:.1f}s")
    results.append({"method": name, "mmd": mmd, "energy": energy, "time": train_time})

# microplex configurations
microplex_configs = [
    {"n_layers": 6, "hidden_dim": 64, "epochs": 50},    # baseline
    {"n_layers": 8, "hidden_dim": 64, "epochs": 50},    # more layers
    {"n_layers": 6, "hidden_dim": 128, "epochs": 50},   # wider
    {"n_layers": 8, "hidden_dim": 128, "epochs": 100},  # bigger + longer
]

print("\n=== MICROPLEX TUNING ===")
for cfg in microplex_configs:
    name = f"microplex(L={cfg['n_layers']},H={cfg['hidden_dim']},E={cfg['epochs']})"
    print(f"\n{name}...")

    start = time.time()
    model = Synthesizer(
        target_vars=target_vars,
        condition_vars=condition_vars,
        n_layers=cfg["n_layers"],
        hidden_dim=cfg["hidden_dim"],
        zero_inflated=True,
    )
    model.fit(train_data, epochs=cfg["epochs"], batch_size=256, verbose=False)
    train_time = time.time() - start

    synthetic = model.generate(test_conditions)

    # Compute metrics
    train_norm, test_norm, synth_norm, _ = normalize_data(
        train_data, test_data, synthetic, target_vars
    )
    mmd = compute_mmd(test_norm, synth_norm)
    energy = compute_energy_distance(test_norm, synth_norm)

    print(f"  MMD={mmd:.4f}, Energy={energy:.4f}, time={train_time:.1f}s")
    results.append({"method": name, "mmd": mmd, "energy": energy, "time": train_time})

# Summary
print("\n" + "=" * 80)
print("TUNING RESULTS (sorted by MMD)")
print("=" * 80)
df = pd.DataFrame(results).sort_values("mmd")
print(df.to_string(index=False))

print("\n" + "=" * 80)
print("BEST CONFIGS")
print("=" * 80)
best_mmd = df.loc[df["mmd"].idxmin()]
best_energy = df.loc[df["energy"].idxmin()]
print(f"Best MMD:    {best_mmd['method']} (MMD={best_mmd['mmd']:.4f})")
print(f"Best Energy: {best_energy['method']} (Energy={best_energy['energy']:.4f})")
