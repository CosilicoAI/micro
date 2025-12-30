"""
Compare IPF vs GD+L0 calibration across sparsity levels.

X-axis: Number of active records
Y-axis: PE-style loss (without L0 penalty)
Lines: IPF (subsampled), GD+L0 (varying lambda)
"""

import sys
sys.path.insert(0, '/Users/maxghenis/CosilicoAI/microplex/src')

import pandas as pd
import numpy as np
from scipy import sparse
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Import L0
from l0.calibration import SparseCalibrationWeights

# Load data
print("Loading data...")
synth = pd.read_parquet('/Users/maxghenis/CosilicoAI/microplex/data/microplex_synthetic_with_blocks.parquet')
synth['state_fips'] = synth['state_fips'].astype(str).str.zfill(2)
print(f"Loaded {len(synth):,} households")

# Load target populations
blocks = pd.read_parquet('/Users/maxghenis/CosilicoAI/microplex/data/block_probabilities.parquet')

# Build state targets
state_pops = blocks.groupby('state_fips')['population'].sum()
state_targets = {str(k).zfill(2): v for k, v in state_pops.items()}
print(f"State targets: {len(state_targets)}")

# Build CD targets
cd_col = 'cd_id' if 'cd_id' in blocks.columns else 'cd_geoid'
cd_pops = blocks.groupby(cd_col)['population'].sum()
cd_targets = dict(cd_pops)
print(f"CD targets: {len(cd_targets)}")

# Build constraint matrix for calibration
def build_constraint_matrix(df, state_targets, cd_targets):
    """Build sparse constraint matrix A and target vector b."""
    n = len(df)
    rows, cols, vals = [], [], []
    targets = []
    names = []
    row_idx = 0

    # State constraints
    for state, target in state_targets.items():
        mask = df['state_fips'] == state
        indices = np.where(mask)[0]
        if len(indices) > 0:
            rows.extend([row_idx] * len(indices))
            cols.extend(indices)
            vals.extend([1.0] * len(indices))
            targets.append(target)
            names.append(f"state_{state}")
            row_idx += 1

    # CD constraints
    for cd, target in cd_targets.items():
        mask = df['cd_id'] == cd
        indices = np.where(mask)[0]
        if len(indices) > 0:
            rows.extend([row_idx] * len(indices))
            cols.extend(indices)
            vals.extend([1.0] * len(indices))
            targets.append(target)
            names.append(f"cd_{cd}")
            row_idx += 1

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(row_idx, n))
    b = np.array(targets)
    return A, b, names

print("Building constraint matrix...")
A, b, target_names = build_constraint_matrix(synth, state_targets, cd_targets)
print(f"Constraints: {A.shape[0]} targets, {A.shape[1]} records")

def compute_pe_loss(weights, A, b):
    """Compute PE-style loss: mean relative squared error."""
    pred = A @ weights
    # Relative error, avoiding division by zero
    rel_err = np.abs(pred - b) / np.maximum(np.abs(b), 1e-10)
    return np.mean(rel_err ** 2)

def compute_mean_abs_error(weights, A, b):
    """Mean absolute percentage error."""
    pred = A @ weights
    rel_err = np.abs(pred - b) / np.maximum(np.abs(b), 1e-10)
    return np.mean(rel_err) * 100

# ============================================================
# Method 1: IPF with subsampling
# ============================================================
print("\n" + "="*60)
print("METHOD 1: IPF with subsampling")
print("="*60)

def ipf_calibrate(df, A, b, max_iter=50, tol=1e-6):
    """Simple IPF calibration."""
    n = len(df)
    weights = np.ones(n)

    for iteration in range(max_iter):
        old_weights = weights.copy()

        # Update for each constraint
        for i in range(A.shape[0]):
            row = A.getrow(i)
            indices = row.indices
            if len(indices) == 0:
                continue

            current = weights[indices].sum()
            target = b[i]
            if current > 1e-10:
                factor = target / current
                weights[indices] *= factor

        # Check convergence
        if np.max(np.abs(weights - old_weights)) < tol:
            break

    return weights

ipf_results = []
sample_fracs = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

for frac in sample_fracs:
    n_sample = int(len(synth) * frac)

    # Subsample
    np.random.seed(42)
    idx = np.random.choice(len(synth), n_sample, replace=False)

    # Build subsampled constraint matrix
    A_sub = A[:, idx]

    # IPF calibrate
    weights_sub = ipf_calibrate(synth.iloc[idx], A_sub, b)

    # Expand to full weights (zeros for non-sampled)
    weights_full = np.zeros(len(synth))
    weights_full[idx] = weights_sub

    # Compute loss
    loss = compute_pe_loss(weights_full, A, b)
    mae = compute_mean_abs_error(weights_full, A, b)

    print(f"  {frac*100:5.1f}% ({n_sample:,} records): Loss={loss:.6f}, MAE={mae:.2f}%")
    ipf_results.append({
        'n_records': n_sample,
        'loss': loss,
        'mae': mae,
        'method': 'IPF+Subsample'
    })

# ============================================================
# Method 2: GD + L0 (varying lambda)
# ============================================================
print("\n" + "="*60)
print("METHOD 2: Gradient Descent + L0")
print("="*60)

l0_results = []
lambdas = [0, 1e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

for lam in lambdas:
    print(f"  λ={lam:.0e}...", end=" ", flush=True)

    model = SparseCalibrationWeights(
        n_features=len(synth),
        init_keep_prob=0.99 if lam > 0 else 1.0,
    )

    model.fit(
        M=A,
        y=b,
        lambda_l0=lam,
        lambda_l2=1e-8,
        lr=0.3,
        epochs=300,
        verbose=False,
    )

    with torch.no_grad():
        weights = model.get_weights(deterministic=True).cpu().numpy()

    n_active = (weights > 0).sum()
    sparsity = (weights == 0).mean()
    loss = compute_pe_loss(weights, A, b)
    mae = compute_mean_abs_error(weights, A, b)

    print(f"Active={n_active:,} ({100-sparsity*100:.1f}%), Loss={loss:.6f}, MAE={mae:.2f}%")

    l0_results.append({
        'n_records': n_active,
        'loss': loss,
        'mae': mae,
        'lambda': lam,
        'method': 'GD+L0'
    })

# ============================================================
# Plot results
# ============================================================
print("\n" + "="*60)
print("Creating plot...")
print("="*60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss vs N records
ipf_n = [r['n_records'] for r in ipf_results]
ipf_loss = [r['loss'] for r in ipf_results]
l0_n = [r['n_records'] for r in l0_results]
l0_loss = [r['loss'] for r in l0_results]

ax1.semilogy(ipf_n, ipf_loss, 'o-', label='IPF + Subsample', linewidth=2, markersize=8)
ax1.semilogy(l0_n, l0_loss, 's-', label='GD + L0', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Active Records', fontsize=12)
ax1.set_ylabel('PE Loss (Mean Relative Squared Error)', fontsize=12)
ax1.set_title('Calibration Loss vs Sample Size', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 550000)

# Plot 2: MAE vs N records
ipf_mae = [r['mae'] for r in ipf_results]
l0_mae = [r['mae'] for r in l0_results]

ax2.plot(ipf_n, ipf_mae, 'o-', label='IPF + Subsample', linewidth=2, markersize=8)
ax2.plot(l0_n, l0_mae, 's-', label='GD + L0', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Active Records', fontsize=12)
ax2.set_ylabel('Mean Absolute Error (%)', fontsize=12)
ax2.set_title('Calibration Error vs Sample Size', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 550000)

plt.tight_layout()
plt.savefig('/Users/maxghenis/CosilicoAI/microplex/docs/sparsity_comparison.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved to docs/sparsity_comparison.png")

# Summary table
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Method':<20} {'Records':>10} {'Loss':>12} {'MAE':>8}")
print("-"*52)
for r in sorted(ipf_results + l0_results, key=lambda x: -x['n_records']):
    print(f"{r['method']:<20} {r['n_records']:>10,} {r['loss']:>12.6f} {r['mae']:>7.2f}%")
