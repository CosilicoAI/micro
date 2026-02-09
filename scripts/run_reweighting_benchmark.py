#!/usr/bin/env python3
"""Run reweighting method comparison benchmark on real data.

Compares IPF, Chi2, Entropy, L1/L2/L0 sparse, SparseCalibrator,
and HardConcrete (if l0-python installed) on target-matching accuracy.

Usage:
    python scripts/run_reweighting_benchmark.py
    python scripts/run_reweighting_benchmark.py --methods ipf entropy l1
    python scripts/run_reweighting_benchmark.py --output benchmarks/results/reweighting.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(data_dir: Path, max_rows: int = 20000) -> pd.DataFrame:
    """Load stacked multi-source data for reweighting benchmark."""
    stacked_path = data_dir / "stacked_comprehensive.parquet"
    if not stacked_path.exists():
        print(f"ERROR: {stacked_path} not found")
        sys.exit(1)

    print(f"Loading {stacked_path}...")
    df = pd.read_parquet(stacked_path)
    print(f"  Total rows: {len(df):,}")

    # Subsample if needed
    if len(df) > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(df), max_rows, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled to {max_rows:,} rows")

    return df


def build_targets_from_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Build realistic calibration targets from the data itself.

    Creates targets that differ from the sample distribution,
    simulating calibration to known population totals.

    Multi-source data has high NaN rates for survey-specific columns,
    so we focus on shared columns (age, is_male) and create age bins.
    """
    marginal_targets = {}
    continuous_targets = {}
    rng = np.random.RandomState(42)

    # Create age bins if age column exists (shared across all surveys)
    if "age" in df.columns:
        bins = [0, 18, 35, 55, 65, 120]
        labels = ["0-17", "18-34", "35-54", "55-64", "65+"]
        df = df.copy()
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
        counts = df["age_group"].value_counts(dropna=True)
        perturbed = {}
        for cat, count in counts.items():
            if pd.isna(cat):
                continue
            perturbed[str(cat)] = round(count * rng.uniform(0.7, 1.3))
        if perturbed:
            # Convert categorical back to string for clean matching
            df["age_group"] = df["age_group"].astype(str)
            marginal_targets["age_group"] = perturbed
            print(f"  Target: age_group ({len(perturbed)} categories)")

    # is_male (shared across all surveys)
    if "is_male" in df.columns and df["is_male"].isna().mean() < 0.05:
        counts = df["is_male"].value_counts(dropna=True)
        perturbed = {}
        for cat, count in counts.items():
            perturbed[cat] = round(count * rng.uniform(0.8, 1.2))
        marginal_targets["is_male"] = perturbed
        print(f"  Target: is_male ({len(perturbed)} categories)")

    # Only use age_group + is_male as categorical targets for reproducibility.
    # Auto-discovery of additional columns changes results when data changes.

    # Continuous targets — use total weight as population count target
    total_weight = df["weight"].sum()
    continuous_targets["weight"] = round(total_weight * rng.uniform(0.9, 1.1))
    print(f"  Target: weight (total={total_weight:,.0f} -> {continuous_targets['weight']:,.0f})")

    return df, marginal_targets, continuous_targets


METHOD_MAP = {
    "ipf": "IPFMethod",
    "chi2": "Chi2Method",
    "entropy": "EntropyMethod",
    "l1": "L1SparseMethod",
    "l2": "L2SparseMethod",
    "l0": "L0SparseMethod",
    "sparse": "SparseCalibratorMethod",
    "hardconcrete": "HardConcreteMethod",
}


def build_methods(method_names: list[str] = None):
    """Build method instances from names."""
    from microplex.eval.reweighting_benchmark import (
        IPFMethod, Chi2Method, EntropyMethod,
        L1SparseMethod, L2SparseMethod, L0SparseMethod,
        SparseCalibratorMethod, HardConcreteMethod,
    )

    all_methods = {
        "ipf": IPFMethod(),
        "chi2": Chi2Method(),
        "entropy": EntropyMethod(),
        "l1": L1SparseMethod(),
        "l2": L2SparseMethod(),
        "l0": L0SparseMethod(),
        "sparse": SparseCalibratorMethod(sparsity_weight=0.01),
        "hardconcrete": HardConcreteMethod(lambda_l0=1e-4, epochs=2000),
    }

    if method_names is None:
        method_names = ["ipf", "chi2", "entropy", "l1", "l2", "l0", "sparse"]
        # Add HardConcrete if l0-python is available
        try:
            import l0
            method_names.append("hardconcrete")
        except ImportError:
            print("  (l0-python not installed, skipping HardConcrete)")

    methods = []
    for name in method_names:
        name_lower = name.lower()
        if name_lower in all_methods:
            methods.append(all_methods[name_lower])
        else:
            print(f"  WARNING: Unknown method '{name}', skipping")

    return methods


def main():
    parser = argparse.ArgumentParser(description="Run reweighting method benchmark")
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to compare (default: all available). "
             "Options: ipf, chi2, entropy, l1, l2, l0, sparse, hardconcrete",
    )
    parser.add_argument("--output", type=str, help="Save results to JSON")
    parser.add_argument(
        "--max-rows", type=int, default=5000,
        help="Max rows (default: 5000, reweighting is O(n) per iteration)",
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=str(Path(__file__).parent.parent / "data"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load data
    df = load_data(data_dir, max_rows=args.max_rows)

    # Build targets (may add derived columns like age_group)
    print("\nBuilding calibration targets...")
    df, marginal_targets, continuous_targets = build_targets_from_data(df)
    print(f"  {len(marginal_targets)} categorical, {len(continuous_targets)} continuous targets")

    if not marginal_targets:
        print("ERROR: No categorical targets found. Need at least one categorical variable.")
        sys.exit(1)

    # Drop rows with NaN in categorical target columns (these cause constraint errors)
    # Continuous targets are handled via fillna(0) in the calibrators
    cat_target_cols = [c for c in marginal_targets.keys() if c in df.columns]
    if cat_target_cols:
        before = len(df)
        df = df.dropna(subset=cat_target_cols).reset_index(drop=True)
        if len(df) < before:
            print(f"  Dropped {before - len(df)} rows with NaN in categorical targets "
                  f"({len(df)} remaining)")

    # Build methods
    methods = build_methods(args.methods)
    print(f"\nMethods to compare ({len(methods)}): {[m.name for m in methods]}")

    # Run benchmark
    from microplex.eval.reweighting_benchmark import ReweightingBenchmarkRunner

    runner = ReweightingBenchmarkRunner(methods=methods)
    t0 = time.time()
    result = runner.run(
        data=df,
        marginal_targets=marginal_targets,
        continuous_targets=continuous_targets if continuous_targets else None,
        seed=args.seed,
    )
    total_elapsed = time.time() - t0

    # Print summary
    print(f"\n{result.summary()}")
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_dict = result.to_dict()
        result_dict["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        result_dict["total_elapsed_seconds"] = round(total_elapsed, 1)
        result_dict["n_records"] = len(df)
        result_dict["n_marginal_targets"] = sum(
            len(v) for v in marginal_targets.values()
        )
        result_dict["n_continuous_targets"] = len(continuous_targets)
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
