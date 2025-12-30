"""
Sparsity Experiment v2: IPF vs GD+L0 with PE-style loss and mixed targets.

Target types:
- Geographic: Population by state (51) and CD (436) - persons
- Income: Household income by bracket (10) - dollars

Loss function: PE-style relative squared error
  loss = mean(((pred - target + 1) / (target + 1))^2 * weight)
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
print("SPARSITY EXPERIMENT V2: IPF vs GD+L0")
print("="*70)

# Load data
synth = pd.read_parquet('/Users/maxghenis/CosilicoAI/microplex/data/microplex_synthetic_with_blocks.parquet')
synth['state_fips'] = synth['state_fips'].astype(str).str.zfill(2)
blocks = pd.read_parquet('/Users/maxghenis/CosilicoAI/microplex/data/block_probabilities.parquet')
print(f"Loaded {len(synth):,} households")

# =============================================================================
# BUILD TARGETS
# =============================================================================

# 1. State populations (51 targets, unit=persons)
state_pops = blocks.groupby('state_fips')['population'].sum()
state_targets = {str(k).zfill(2): v for k, v in state_pops.items()}

# 2. CD populations (436 targets, unit=persons)
cd_col = 'cd_id' if 'cd_id' in blocks.columns else 'cd_geoid'
cd_pops = blocks.groupby(cd_col)['population'].sum()
cd_targets = dict(cd_pops)

# 3. Income brackets (10 targets, unit=dollars)
# Create income brackets and target total income per bracket
income_brackets = [0, 25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000, 1000000, np.inf]
bracket_labels = [f"income_{i}" for i in range(len(income_brackets)-1)]
synth['income_bracket'] = pd.cut(synth['hh_income'], bins=income_brackets, labels=bracket_labels)

# Target: total income per bracket (using current weighted totals as "truth")
# In reality these would come from IRS SOI data
income_targets = {}
for bracket in bracket_labels:
    mask = synth['income_bracket'] == bracket
    total = (synth.loc[mask, 'hh_income'] * synth.loc[mask, 'weight']).sum()
    income_targets[bracket] = total

print(f"\nTargets:")
print(f"  States: {len(state_targets)} (persons)")
print(f"  CDs: {len(cd_targets)} (persons)")
print(f"  Income brackets: {len(income_targets)} (dollars)")
print(f"  Total: {len(state_targets) + len(cd_targets) + len(income_targets)}")

# =============================================================================
# BUILD CONSTRAINT MATRIX
# =============================================================================

def build_constraints(df, state_targets, cd_targets, income_targets):
    """Build sparse constraint matrix with target metadata."""
    n = len(df)
    rows, cols, vals = [], [], []
    targets = []
    weights = []  # Normalization weights (PE-style)
    names = []
    units = []
    row_idx = 0

    # State constraints (population counts)
    for state, target in state_targets.items():
        indices = np.where(df['state_fips'] == state)[0]
        if len(indices) > 0:
            rows.extend([row_idx] * len(indices))
            cols.extend(indices)
            vals.extend([1.0] * len(indices))  # Count
            targets.append(target)
            weights.append(1.0)  # National-level weight
            names.append(f"state_{state}")
            units.append('persons')
            row_idx += 1

    # CD constraints (population counts)
    for cd, target in cd_targets.items():
        indices = np.where(df['cd_id'] == cd)[0]
        if len(indices) > 0:
            rows.extend([row_idx] * len(indices))
            cols.extend(indices)
            vals.extend([1.0] * len(indices))  # Count
            targets.append(target)
            weights.append(0.5)  # Sub-national weight (PE uses different weights)
            names.append(f"cd_{cd}")
            units.append('persons')
            row_idx += 1

    # Income bracket constraints (dollar sums)
    for bracket, target in income_targets.items():
        mask = df['income_bracket'] == bracket
        indices = np.where(mask)[0]
        if len(indices) > 0:
            rows.extend([row_idx] * len(indices))
            cols.extend(indices)
            # Value is the income amount (not 1.0 for counts)
            vals.extend(df.loc[df.index[indices], 'hh_income'].values)
            targets.append(target)
            weights.append(1.0)  # National-level weight
            names.append(bracket)
            units.append('dollars')
            row_idx += 1

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(row_idx, n))
    return A, np.array(targets), np.array(weights), names, units

A, b, target_weights, target_names, target_units = build_constraints(
    synth, state_targets, cd_targets, income_targets
)
print(f"\nConstraint matrix: {A.shape[0]} targets × {A.shape[1]} records")

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def pe_loss(weights, A, b, target_weights):
    """
    PolicyEngine-style loss: weighted relative squared error.

    loss = mean(((pred - target + 1) / (target + 1))^2 * target_weight)
    """
    pred = A @ weights
    # Relative squared error with +1 smoothing
    rel_sq_err = ((pred - b + 1) / (b + 1)) ** 2
    # Weight by target importance
    weighted_err = rel_sq_err * target_weights
    # Normalize by mean weight
    return weighted_err.mean() / target_weights.mean()

def pe_loss_by_unit(weights, A, b, target_units):
    """Compute PE loss separately for each unit type."""
    pred = A @ weights
    rel_sq_err = ((pred - b + 1) / (b + 1)) ** 2

    losses = {}
    for unit in set(target_units):
        mask = np.array([u == unit for u in target_units])
        if mask.sum() > 0:
            losses[unit] = rel_sq_err[mask].mean()
    return losses

def mae_by_unit(weights, A, b, target_units):
    """Mean absolute percentage error by unit type."""
    pred = A @ weights
    rel_err = np.abs(pred - b) / np.maximum(np.abs(b), 1e-10) * 100

    maes = {}
    for unit in set(target_units):
        mask = np.array([u == unit for u in target_units])
        if mask.sum() > 0:
            maes[unit] = rel_err[mask].mean()
    return maes

# =============================================================================
# IPF CALIBRATION
# =============================================================================

def ipf_calibrate(A_sub, b, max_iter=100, tol=1e-8):
    """IPF calibration."""
    n = A_sub.shape[1]
    weights = np.ones(n)

    for _ in range(max_iter):
        old_weights = weights.copy()
        for i in range(A_sub.shape[0]):
            row = A_sub.getrow(i)
            if row.nnz == 0:
                continue
            # Current weighted sum
            current = (weights[row.indices] * row.data).sum()
            if current > 1e-10:
                factor = b[i] / current
                weights[row.indices] *= factor

        if np.max(np.abs(weights - old_weights) / (old_weights + 1e-10)) < tol:
            break

    return weights

# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

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

    # Expand to full
    w_full = np.zeros(len(synth))
    w_full[idx] = w_sub

    loss = pe_loss(w_full, A, b, target_weights)
    mae_units = mae_by_unit(w_full, A, b, target_units)

    print(f"  {frac*100:5.1f}% ({n_sample:>7,}): PE Loss={loss:.6f}, "
          f"MAE persons={mae_units.get('persons', 0):.2f}%, dollars={mae_units.get('dollars', 0):.2f}%")

    results.append({
        'method': 'IPF+Subsample',
        'n_records': n_sample,
        'pe_loss': loss,
        'mae_persons': mae_units.get('persons', 0),
        'mae_dollars': mae_units.get('dollars', 0),
    })

# GD + L0
print("\n" + "="*70)
print("METHOD 2: Gradient Descent + L0")
print("="*70)

configs = [
    (0, 1.0),       # No sparsity
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
    model.fit(M=A, y=b, lambda_l0=lam, lr=0.5, epochs=500, verbose=False)

    with torch.no_grad():
        weights = model.get_weights(deterministic=True).cpu().numpy()

    n_active = (weights > 0).sum()
    loss = pe_loss(weights, A, b, target_weights)
    mae_units = mae_by_unit(weights, A, b, target_units)

    print(f"  λ={lam:.0e}, init={init_keep:.2f}: {n_active:>7,} active, PE Loss={loss:.6f}, "
          f"MAE persons={mae_units.get('persons', 0):.2f}%, dollars={mae_units.get('dollars', 0):.2f}%")

    results.append({
        'method': 'GD+L0',
        'n_records': n_active,
        'pe_loss': loss,
        'mae_persons': mae_units.get('persons', 0),
        'mae_dollars': mae_units.get('dollars', 0),
        'lambda': lam,
    })

# =============================================================================
# PLOT RESULTS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Separate by method
ipf_res = [r for r in results if r['method'] == 'IPF+Subsample']
l0_res = [r for r in results if r['method'] == 'GD+L0']

# Plot 1: PE Loss
ax = axes[0]
ax.semilogy([r['n_records'] for r in ipf_res], [r['pe_loss'] for r in ipf_res],
            'o-', label='IPF + Subsample', linewidth=2.5, markersize=10, color='#2ecc71')
ax.semilogy([r['n_records'] for r in l0_res], [r['pe_loss'] for r in l0_res],
            's-', label='GD + L0', linewidth=2.5, markersize=10, color='#e74c3c')
ax.set_xlabel('Number of Active Records', fontsize=12)
ax.set_ylabel('PE Loss (log scale)', fontsize=12)
ax.set_title('PolicyEngine-Style Loss', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: MAE Persons
ax = axes[1]
ax.plot([r['n_records'] for r in ipf_res], [r['mae_persons'] for r in ipf_res],
        'o-', label='IPF + Subsample', linewidth=2.5, markersize=10, color='#2ecc71')
ax.plot([r['n_records'] for r in l0_res], [r['mae_persons'] for r in l0_res],
        's-', label='GD + L0', linewidth=2.5, markersize=10, color='#e74c3c')
ax.set_xlabel('Number of Active Records', fontsize=12)
ax.set_ylabel('MAE (%)', fontsize=12)
ax.set_title('Geographic Targets (Persons)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

# Plot 3: MAE Dollars
ax = axes[2]
ax.plot([r['n_records'] for r in ipf_res], [r['mae_dollars'] for r in ipf_res],
        'o-', label='IPF + Subsample', linewidth=2.5, markersize=10, color='#2ecc71')
ax.plot([r['n_records'] for r in l0_res], [r['mae_dollars'] for r in l0_res],
        's-', label='GD + L0', linewidth=2.5, markersize=10, color='#e74c3c')
ax.set_xlabel('Number of Active Records', fontsize=12)
ax.set_ylabel('MAE (%)', fontsize=12)
ax.set_title('Income Targets (Dollars)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Calibration: IPF vs GD+L0 with Mixed Targets\n'
             '(51 states + 436 CDs + 10 income brackets = 497 targets)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('/Users/maxghenis/CosilicoAI/microplex/docs/sparsity_comparison_v2.png',
            dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: docs/sparsity_comparison_v2.png")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Best Results at Different Sparsity Levels")
print("="*70)
print(f"{'Method':<18} {'Records':>10} {'PE Loss':>12} {'Persons MAE':>12} {'Dollars MAE':>12}")
print("-"*66)
for r in sorted(results, key=lambda x: -x['n_records']):
    print(f"{r['method']:<18} {r['n_records']:>10,} {r['pe_loss']:>12.6f} "
          f"{r['mae_persons']:>11.2f}% {r['mae_dollars']:>11.2f}%")
