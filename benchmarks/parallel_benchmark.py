"""Parallel benchmark using multiprocessing."""

import sys
import time
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")

from run_cps_benchmark import generate_cps_like_data
from multivariate_metrics import (
    compute_mmd, compute_energy_distance, normalize_data,
    compute_authenticity_distance, compute_coverage_distance
)

# Generate data once (will be shared)
print(f"Generating data on {cpu_count()} CPUs...")
full_data = generate_cps_like_data(25000, seed=42)
train_data = full_data.iloc[:20000].copy()
test_data = full_data.iloc[20000:].copy()

target_vars = ["wage_income", "self_emp_income", "ssi_income", "uc_income",
               "snap_benefit", "eitc", "agi", "federal_tax"]
condition_vars = ["age", "education", "is_employed", "marital_status"]
all_vars = target_vars + condition_vars
test_conditions = test_data[condition_vars].copy()

# Save to disk for worker processes
data_path = Path("/tmp/benchmark_data.pkl")
with open(data_path, "wb") as f:
    pickle.dump({
        "train_data": train_data,
        "test_data": test_data,
        "test_conditions": test_conditions,
        "target_vars": target_vars,
        "condition_vars": condition_vars,
        "all_vars": all_vars,
    }, f)

def evaluate_synthetic(synthetic, train_data, test_data, target_vars):
    """Compute metrics."""
    train_norm, test_norm, synth_norm, _ = normalize_data(
        train_data, test_data, synthetic, target_vars
    )
    return {
        "mmd": compute_mmd(test_norm, synth_norm),
        "energy_dist": compute_energy_distance(test_norm, synth_norm),
        "authenticity": compute_authenticity_distance(synth_norm, test_norm)["mean"],
        "coverage": compute_coverage_distance(test_norm, synth_norm)["mean"],
    }

def run_microplex(args):
    """Run microplex in worker process."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from microplex import Synthesizer

    with open("/tmp/benchmark_data.pkl", "rb") as f:
        data = pickle.load(f)

    start = time.time()
    model = Synthesizer(
        target_vars=data["target_vars"],
        condition_vars=data["condition_vars"],
        n_layers=8, hidden_dim=128, zero_inflated=True,
    )
    model.fit(data["train_data"], epochs=100, batch_size=256, verbose=False)
    synthetic = model.generate(data["test_conditions"])
    train_time = time.time() - start

    metrics = evaluate_synthetic(synthetic, data["train_data"], data["test_data"], data["target_vars"])
    return {"method": "microplex (tuned)", "time": train_time, **metrics}

def run_qrf_zi(args):
    """Run QRF+ZI in worker process."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from compare_qrf import SequentialQRFWithZeroInflation

    with open("/tmp/benchmark_data.pkl", "rb") as f:
        data = pickle.load(f)

    start = time.time()
    model = SequentialQRFWithZeroInflation(
        data["target_vars"], data["condition_vars"],
        n_estimators=200, max_depth=15
    )
    model.fit(data["train_data"], verbose=False)
    synthetic = model.generate(data["test_conditions"])
    train_time = time.time() - start

    metrics = evaluate_synthetic(synthetic, data["train_data"], data["test_data"], data["target_vars"])
    return {"method": "QRF+ZI (tuned)", "time": train_time, **metrics}

def run_ctgan(args):
    """Run CT-GAN in worker process."""
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    with open("/tmp/benchmark_data.pkl", "rb") as f:
        data = pickle.load(f)

    start = time.time()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data["train_data"][data["all_vars"]])
    model = CTGANSynthesizer(metadata, epochs=50, verbose=False)
    model.fit(data["train_data"][data["all_vars"]])
    synthetic = model.sample(len(data["test_conditions"]))
    train_time = time.time() - start

    metrics = evaluate_synthetic(synthetic, data["train_data"], data["test_data"], data["target_vars"])
    return {"method": "CT-GAN", "time": train_time, **metrics}

def run_xgboost_zi(args):
    """Run XGBoost+ZI in worker process."""
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression

    with open("/tmp/benchmark_data.pkl", "rb") as f:
        data = pickle.load(f)

    start = time.time()
    synthetic_data = data["test_conditions"].copy()
    available_features = list(data["condition_vars"])

    for target in data["target_vars"]:
        X_train = data["train_data"][available_features].values
        y_train = data["train_data"][target].values
        X_test = synthetic_data[available_features].values

        y_binary = (y_train > 0).astype(int)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_binary)
        p_positive = clf.predict_proba(X_test)[:, 1]

        mask = y_train > 0
        if mask.sum() > 10:
            reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, verbosity=0)
            reg.fit(X_train[mask], y_train[mask])
            pred_positive = reg.predict(X_test)
        else:
            pred_positive = np.full(len(X_test), y_train[mask].mean() if mask.sum() > 0 else 0)

        is_positive = np.random.RandomState(42).random(len(X_test)) < p_positive
        predictions = np.where(is_positive, np.maximum(pred_positive, 0), 0)
        synthetic_data[target] = predictions
        available_features.append(target)

    train_time = time.time() - start
    metrics = evaluate_synthetic(synthetic_data, data["train_data"], data["test_data"], data["target_vars"])
    return {"method": "XGBoost+ZI", "time": train_time, **metrics}

if __name__ == "__main__":
    print("\nRunning 4 methods in parallel...")
    start_total = time.time()

    # Run all methods in parallel
    with Pool(4) as pool:
        futures = [
            pool.apply_async(run_microplex, (None,)),
            pool.apply_async(run_qrf_zi, (None,)),
            pool.apply_async(run_ctgan, (None,)),
            pool.apply_async(run_xgboost_zi, (None,)),
        ]
        results = [f.get() for f in futures]

    total_time = time.time() - start_total

    # Print results
    print("\n" + "=" * 80)
    print("PARALLEL BENCHMARK RESULTS")
    print("=" * 80)

    df = pd.DataFrame(results).sort_values("mmd")
    print(f"\n{'Method':<20} {'MMD':>10} {'Energy':>10} {'Auth':>10} {'Coverage':>10} {'Time':>10}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['method']:<20} {row['mmd']:>10.4f} {row['energy_dist']:>10.4f} "
              f"{row['authenticity']:>10.4f} {row['coverage']:>10.4f} {row['time']:>10.1f}s")

    print(f"\nTotal wall time (parallel): {total_time:.1f}s")
    print(f"Sum of individual times:    {df['time'].sum():.1f}s")
    print(f"Speedup:                    {df['time'].sum() / total_time:.1f}x")
