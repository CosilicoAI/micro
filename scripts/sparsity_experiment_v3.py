"""
Sparsity Experiment v3: IPF vs GD+L0 with proper normalization.

Key fix: Normalize targets to unit scale before optimization.
"""

import sys
sys.path.insert(0, '/Users/maxghenis/CosilicoAI/microplex/src')

import pandas as pd
import numpy as np
from scipy import sparse
import torch
import matplotlib.pyplot as plt
from l0.calibration import SparseCalibrationWeights

print("="*70)
print("SPARSITY EXPERIMENT V3: IPF vs GD+L0 (Normalized)")
print("="*70)

# Load data
synth = pd.read_parquet('/Users/maxghenis/CosilicoAI/microplex/data/microplex_synthetic_with_blocks.parquet')
synth['state_fips'] = synth['state_fips'].astype(str).str.zfill(2)
blocks = pd.read_parquet('/Users/maxghenis/CosilicoAI/microplex/data/block_probabilities.parquet')
print(f"Loaded {len(synth):,} households")

# Build targets
state_pops = blocks.groupby('state_fips')['population'].sum()
state_targets = {str(k).zfill(2): v for k, v in state_pops.items()}

cd_col = 'cd_id' if 'cd_id' in blocks.columns else 'cd_geoid'
cd_pops = blocks.groupby(cd_col)['population'].sum()
cd_targets = dict(cd_pops)

income_brackets = [0, 25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 1000000, np.inf]
bracket_labels = [f"income_{i}" for i in range(len(income_brackets)-1)]
synth['income_bracket'] = pd.cut(synth['hh_income'], bins=income_brackets, labels=bracket_labels)

income_targets = {}
for bracket in bracket_labels:
    mask = synth['income_bracket'] == bracket
    total = (synth.loc[mask, 'hh_income'] * synth.loc[mask, 'weight']).sum()
    income_targets[bracket] = total

print(f"\nTargets: {len(state_targets)} states + {len(cd_targets)} CDs + {len(income_targets)} income = "
      f"{len(state_targets) + len(cd_targets) + len(income_targets)} total")

# Build NORMALIZED constraint matrix
def build_normalized_constraints(df, state_targets, cd_targets, income_targets):
    """Build constraint matrix with targets normalized to 1.0."""
    n = len(df)
    rows, cols, vals = [], [], []
    raw_targets = []
    target_types = []
    row_idx = 0

    # State constraints
    for state, target in state_targets.items():
        indices = np.where(df['state_fips'] == state)[0]
        if len(indices) > 0:
            rows.extend([row_idx] * len(indices))
            cols.extend(indices)
            vals.extend([1.0 / target] * len(indices))
            raw_targets.append(target)
            target_types.append('persons')
            row_idx += 1

    # CD constraints
    for cd, target in cd_targets.items():
        indices = np.where(df['cd_id'] == cd)[0]
        if len(indices) > 0:
            rows.extend([row_idx] * len(indices))
            cols.extend(indices)
            vals.extend([1.0 / target] * len(indices))
            raw_targets.append(target)
            target_types.append('persons')
            row_idx += 1

    # Income constraints
    for bracket, target in income_targets.items():
        mask = df['income_bracket'] == bracket
        indices = np.where(mask)[0]
        if len(indices) > 0:
            rows.extend([row_idx] * len(indices))
            cols.extend(indices)
            inc_vals = df.loc[df.index[indices], 'hh_income'].values / target
            vals.extend(inc_vals)
            raw_targets.append(target)
            target_types.append('dollars')
            row_idx += 1

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(row_idx, n))
    b = np.ones(row_idx)  # All targets = 1.0
    return A, b, np.array(raw_targets), target_types

A, b, raw_targets, target_types = build_normalized_constraints(
    synth, state_targets, cd_targets, income_targets
)
print(f"Constraint matrix: {A.shape[0]} targets × {A.shape[1]} records (normalized)")

def compute_errors(weights, A, raw_targets, target_types):
    """Compute MAE by target type."""
    pred_norm = A @ weights
    pred_raw = pred_norm * raw_targets

    rel_err = np.abs(pred_norm - 1.0) * 100  # Since targets are 1.0

    persons_mask = np.array([t == 'persons' for t in target_types])
    dollars_mask = np.array([t == 'dollars' for t in target_types])

    return {
        'overall': rel_err.mean(),
        'persons': rel_err[persons_mask].mean() if persons_mask.sum() > 0 else 0,
        'dollars': rel_err[dollars_mask].mean() if dollars_mask.sum() > 0 else 0,
    }

def ipf_calibrate(A_sub, b, max_iter=100, tol=1e-8):
    """IPF on normalized targets."""
    n = A_sub.shape[1]
    weights = np.ones(n)
    for _ in range(max_iter):
        old = weights.copy()
        for i in range(A_sub.shape[0]):
            row = A_sub.getrow(i)
            if row.nnz == 0:
                continue
            current = (weights[row.indices] * row.data).sum()
            if current > 1e-10:
                weights[row.indices] *= b[i] / current
        if np.max(np.abs(weights - old) / (old + 1e-10)) < tol:
            break
    return weights

results = []

# IPF + Subsampling
print("\n" + "="*70)
print("METHOD 1: IPF + Random Subsampling")
print("="*70)

for frac in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
    n_sample = int(len(synth) * frac)
    np.random.seed(42)
    idx = np.random.choice(len(synth), n_sample, replace=False)
    A_sub = A[:, idx]
    w_sub = ipf_calibrate(A_sub, b)
    w_full = np.zeros(len(synth))
    w_full[idx] = w_sub

    errs = compute_errors(w_full, A, raw_targets, target_types)
    print(f"  {frac*100:5.1f}% ({n_sample:>7,}): persons={errs['persons']:.2f}%, "
          f"dollars={errs['dollars']:.2f}%, overall={errs['overall']:.2f}%")

    results.append({'method': 'IPF+Subsample', 'n_records': n_sample, **errs})

# GD + L0
print("\n" + "="*70)
print("METHOD 2: Gradient Descent + L0")
print("="*70)

configs = [
    (0, 1.0),
    (1e-4, 0.9),
    (1e-3, 0.7),
    (5e-3, 0.5),
    (1e-2, 0.3),
    (5e-2, 0.2),
    (1e-1, 0.1),
    (5e-1, 0.05),
    (1.0, 0.02),
]

for lam, init_keep in configs:
    model = SparseCalibrationWeights(n_features=len(synth), init_keep_prob=init_keep)
    model.fit(M=A, y=b, lambda_l0=lam, lr=0.3, epochs=500, verbose=False)

    with torch.no_grad():
        weights = model.get_weights(deterministic=True).cpu().numpy()

    n_active = (weights > 0).sum()
    errs = compute_errors(weights, A, raw_targets, target_types)

    print(f"  λ={lam:.0e}, init={init_keep:.2f}: {n_active:>7,} active, persons={errs['persons']:.2f}%, "
          f"dollars={errs['dollars']:.2f}%, overall={errs['overall']:.2f}%")

    results.append({'method': 'GD+L0', 'n_records': n_active, 'lambda': lam, **errs})

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ipf = [r for r in results if r['method'] == 'IPF+Subsample']
l0 = [r for r in results if r['method'] == 'GD+L0']

for ax, metric, title in [
    (axes[0], 'overall', 'Overall MAE'),
    (axes[1], 'persons', 'Geographic Targets (Persons)'),
    (axes[2], 'dollars', 'Income Targets (Dollars)'),
]:
    ax.plot([r['n_records'] for r in ipf], [r[metric] for r in ipf],
            'o-', label='IPF + Subsample', linewidth=2.5, markersize=10, color='#2ecc71')
    ax.plot([r['n_records'] for r in l0], [r[metric] for r in l0],
            's-', label='GD + L0', linewidth=2.5, markersize=10, color='#e74c3c')
    ax.set_xlabel('Number of Active Records', fontsize=12)
    ax.set_ylabel('MAE (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('IPF vs GD+L0: Mixed Targets with Normalization\n'
             f'({len(state_targets)} states + {len(cd_targets)} CDs + {len(income_targets)} income = '
             f'{len(state_targets)+len(cd_targets)+len(income_targets)} targets)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('/Users/maxghenis/CosilicoAI/microplex/docs/sparsity_comparison_v3.png',
            dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: docs/sparsity_comparison_v3.png")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Method':<18} {'Records':>10} {'Persons':>10} {'Dollars':>10} {'Overall':>10}")
print("-"*60)
for r in sorted(results, key=lambda x: -x['n_records']):
    print(f"{r['method']:<18} {r['n_records']:>10,} {r['persons']:>9.2f}% {r['dollars']:>9.2f}% {r['overall']:>9.2f}%")
