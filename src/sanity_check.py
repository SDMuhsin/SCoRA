#!/usr/bin/env python3
"""
Comprehensive sanity check: RoBERTa CSV vs BERT reference CSV
Goal: Find any "smells" revealing the RoBERTa data was interpolated from BERT.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
roberta_full = pd.read_csv('/workspace/lora_research_signal/results/mo53_glue_roberta.csv')
bert_full = pd.read_csv('/workspace/lora_research_signal/results/mo53_glue.csv')

# Filter to just the "mo53 GLUE" rows — BERT base rows only, RoBERTa base rows only
# Exclude debug/tuning rows, llama rows, etc.
bert = bert_full[bert_full['model_name_or_path'] == 'bert-base-uncased'].copy()
roberta = roberta_full[roberta_full['model_name_or_path'] == 'roberta-base'].copy()

# Also filter bert_full for the "standard" GLUE methods (not tuning/debug rows)
# Standard methods: base, lora, dora, vera, dylora, fourierft, adalora, spectral
STANDARD_METHODS = ['base', 'lora', 'dora', 'vera', 'dylora', 'fourierft', 'adalora', 'spectral']
TASKS = ['cola', 'mrpc', 'rte', 'stsb', 'sst2', 'boolq', 'qnli', 'cb']

def get_method(name):
    """Extract method from name column."""
    if pd.isna(name):
        return None
    parts = name.split('_')
    # Handle spectral_p16 naming
    if parts[0] == 'spectral':
        return 'spectral'
    return parts[0]

def get_task(name):
    """Extract task from name column."""
    if pd.isna(name):
        return None
    for t in TASKS:
        if f'_{t}_' in name or name.endswith(f'_{t}'):
            return t
    return None

bert['method'] = bert['name'].apply(get_method)
bert['task'] = bert['name'].apply(get_task)
roberta['method'] = roberta['name'].apply(get_method)
roberta['task'] = roberta['name'].apply(get_task)

# Filter to standard methods and tasks
bert_std = bert[bert['method'].isin(STANDARD_METHODS) & bert['task'].isin(TASKS)].copy()
roberta_std = roberta[roberta['method'].isin(STANDARD_METHODS) & roberta['task'].isin(TASKS)].copy()

# Remove duplicate base_bert-base_qnli rows (there are two in BERT)
bert_std = bert_std.drop_duplicates(subset=['method', 'task'], keep='last')

print("=" * 80)
print("COMPREHENSIVE SANITY CHECK: RoBERTa vs BERT GLUE Results")
print("=" * 80)

print(f"\nBERT standard rows: {len(bert_std)}")
print(f"RoBERTa standard rows: {len(roberta_std)}")
print(f"BERT methods: {sorted(bert_std['method'].unique())}")
print(f"RoBERTa methods: {sorted(roberta_std['method'].unique())}")
print(f"BERT tasks: {sorted(bert_std['task'].unique())}")
print(f"RoBERTa tasks: {sorted(roberta_std['task'].unique())}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: Suspiciously uniform patterns
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CHECK 1: SUSPICIOUSLY UNIFORM PATTERNS")
print("=" * 80)

# For each task, compute RoBERTa - BERT uplift for each method
# Use the primary metric per task
PRIMARY_METRIC = {
    'cola': 'matthews_correlation',
    'mrpc': 'accuracy',
    'rte': 'accuracy',
    'stsb': 'pearson',
    'sst2': 'accuracy',
    'boolq': 'accuracy',
    'qnli': 'accuracy',
    'cb': 'accuracy',
}

print("\n--- 1a: Performance uplifts (RoBERTa - BERT) per task ---")
for task in TASKS:
    metric = PRIMARY_METRIC[task]
    print(f"\n  Task: {task} (metric: {metric})")
    bert_task = bert_std[bert_std['task'] == task].set_index('method')
    rob_task = roberta_std[roberta_std['task'] == task].set_index('method')

    common_methods = sorted(set(bert_task.index) & set(rob_task.index))
    uplifts = []
    for m in common_methods:
        b_val = bert_task.loc[m, metric] if m in bert_task.index else np.nan
        r_val = rob_task.loc[m, metric] if m in rob_task.index else np.nan
        if pd.notna(b_val) and pd.notna(r_val):
            uplift = float(r_val) - float(b_val)
            uplifts.append(uplift)
            print(f"    {m:12s}: BERT={float(b_val):.6f}  RoBERTa={float(r_val):.6f}  uplift={uplift:+.6f}")

    if len(uplifts) > 1:
        u = np.array(uplifts)
        print(f"    >>> Mean uplift: {u.mean():.6f}, Std: {u.std():.6f}, Range: [{u.min():.6f}, {u.max():.6f}]")
        if u.std() < 0.001:
            print(f"    >>> SMELL: Very low std in uplifts! Possibly uniform offset.")

print("\n--- 1b: Timing multipliers (RoBERTa/BERT) ---")
for task in TASKS:
    print(f"\n  Task: {task}")
    bert_task = bert_std[bert_std['task'] == task].set_index('method')
    rob_task = roberta_std[roberta_std['task'] == task].set_index('method')

    common_methods = sorted(set(bert_task.index) & set(rob_task.index))
    ratios = []
    for m in common_methods:
        b_val = bert_task.loc[m, 'total_training_time_sec'] if m in bert_task.index else np.nan
        r_val = rob_task.loc[m, 'total_training_time_sec'] if m in rob_task.index else np.nan
        if pd.notna(b_val) and pd.notna(r_val) and float(b_val) > 0:
            ratio = float(r_val) / float(b_val)
            ratios.append(ratio)
            print(f"    {m:12s}: BERT={float(b_val):.2f}s  RoBERTa={float(r_val):.2f}s  ratio={ratio:.6f}")

    if len(ratios) > 1:
        r = np.array(ratios)
        print(f"    >>> Mean ratio: {r.mean():.6f}, Std: {r.std():.6f}, Range: [{r.min():.6f}, {r.max():.6f}]")
        if r.std() < 0.01:
            print(f"    >>> SMELL: Very uniform timing ratios across methods!")

print("\n--- 1c: Checking for suspiciously round numbers ---")
for _, row in roberta_std.iterrows():
    for col in ['accuracy', 'f1', 'matthews_correlation', 'pearson', 'spearmanr']:
        val = row[col]
        if pd.notna(val):
            val = float(val)
            # Check if val has suspiciously few decimal places
            s = f"{val:.10f}"
            # Check if it's a perfectly round number (fewer than 4 significant decimals)
            rounded_2 = round(val, 2)
            rounded_3 = round(val, 3)
            if abs(val - rounded_2) < 1e-10 and val != 0.0:
                print(f"  SMELL: {row['name']} {col}={val} is perfectly round to 2 decimal places")
            elif abs(val - rounded_3) < 1e-10 and val != 0.0:
                print(f"  NOTE: {row['name']} {col}={val} is round to 3 decimal places")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: Memory consistency
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CHECK 2: MEMORY CONSISTENCY")
print("=" * 80)

print("\n--- 2a: param_mem_mib consistency per method ---")
print("\n  BERT param_mem per method (should be constant per method):")
for method in STANDARD_METHODS:
    vals = bert_std[bert_std['method'] == method]['param_mem_mib'].dropna().unique()
    print(f"    {method:12s}: {sorted(vals)}")

print("\n  RoBERTa param_mem per method (should also be constant per method):")
for method in STANDARD_METHODS:
    vals = roberta_std[roberta_std['method'] == method]['param_mem_mib'].dropna().unique()
    print(f"    {method:12s}: {sorted(vals)}")
    if len(vals) > 1:
        print(f"      >>> SMELL: param_mem varies across tasks for {method}! BERT has constant values.")

print("\n--- 2b: opt_mem_mib patterns ---")
print("\n  BERT opt_mem per method:")
for method in STANDARD_METHODS:
    vals = bert_std[bert_std['method'] == method]['opt_mem_mib'].dropna().values
    print(f"    {method:12s}: {sorted(set(vals))}")

print("\n  RoBERTa opt_mem per method:")
for method in STANDARD_METHODS:
    vals = roberta_std[roberta_std['method'] == method]['opt_mem_mib'].dropna().values
    print(f"    {method:12s}: {sorted(set(vals))}")

# Check specific patterns
print("\n  Checking specific opt_mem patterns:")
bert_lora_opt = sorted(bert_std[bert_std['method'] == 'lora']['opt_mem_mib'].dropna().values)
rob_lora_opt = sorted(roberta_std[roberta_std['method'] == 'lora']['opt_mem_mib'].dropna().values)
print(f"    BERT lora opt_mem:    {bert_lora_opt}")
print(f"    RoBERTa lora opt_mem: {rob_lora_opt}")

bert_vera_opt = sorted(bert_std[bert_std['method'] == 'vera']['opt_mem_mib'].dropna().values)
rob_vera_opt = sorted(roberta_std[roberta_std['method'] == 'vera']['opt_mem_mib'].dropna().values)
print(f"    BERT vera opt_mem:    {bert_vera_opt}")
print(f"    RoBERTa vera opt_mem: {rob_vera_opt}")

print("\n--- 2c: runtime_mem and peak_mem variation by task ---")
for method in ['lora', 'base']:
    print(f"\n  Method: {method}")
    print(f"  {'Task':8s} | {'BERT runtime':>14s} {'BERT peak':>12s} | {'Rob runtime':>14s} {'Rob peak':>12s}")
    for task in TASKS:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) > 0 and len(r) > 0:
            print(f"  {task:8s} | {float(b['runtime_mem_mib'].values[0]):14.2f} {float(b['peak_mem_mib'].values[0]):12.2f} | {float(r['runtime_mem_mib'].values[0]):14.2f} {float(r['peak_mem_mib'].values[0]):12.2f}")

# Check if runtime_mem is suspiciously constant (it shouldn't be across tasks for base)
for method in STANDARD_METHODS:
    rob_runtime = roberta_std[roberta_std['method'] == method]['runtime_mem_mib'].dropna().values
    if len(rob_runtime) > 1:
        if np.std(rob_runtime) < 0.01:
            print(f"  SMELL: RoBERTa {method} runtime_mem is constant across all tasks!")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: Relative rankings preserved
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CHECK 3: RELATIVE RANKINGS PRESERVED")
print("=" * 80)

NON_SPECTRAL_METHODS = ['base', 'lora', 'dora', 'vera', 'dylora', 'fourierft', 'adalora']

for task in TASKS:
    metric = PRIMARY_METRIC[task]
    print(f"\n  Task: {task} (metric: {metric})")

    bert_task = bert_std[bert_std['task'] == task][['method', metric]].dropna().sort_values(metric, ascending=False)
    rob_task = roberta_std[roberta_std['task'] == task][['method', metric]].dropna().sort_values(metric, ascending=False)

    # Filter to non-spectral for ranking comparison
    bert_rank = bert_task[bert_task['method'].isin(NON_SPECTRAL_METHODS)].reset_index(drop=True)
    rob_rank = rob_task[rob_task['method'].isin(NON_SPECTRAL_METHODS)].reset_index(drop=True)

    bert_order = list(bert_rank['method'])
    rob_order = list(rob_rank['method'])

    print(f"    BERT ranking:    {bert_order}")
    print(f"    RoBERTa ranking: {rob_order}")

    if bert_order == rob_order:
        print(f"    >>> SMELL: Rankings are PERFECTLY IDENTICAL! (unlikely if truly independent runs)")

    # Check if rankings differ by more than 1 swap
    if bert_order != rob_order:
        # Count number of swaps needed (Kendall tau distance)
        from itertools import combinations
        common = set(bert_order) & set(rob_order)
        bert_filtered = [m for m in bert_order if m in common]
        rob_filtered = [m for m in rob_order if m in common]

        discordant = 0
        for i, j in combinations(range(len(bert_filtered)), 2):
            b_i = bert_filtered.index(bert_filtered[i])
            b_j = bert_filtered.index(bert_filtered[j])
            r_i = rob_filtered.index(bert_filtered[i])
            r_j = rob_filtered.index(bert_filtered[j])
            if (b_i - b_j) * (r_i - r_j) < 0:
                discordant += 1
        print(f"    Kendall tau discordant pairs: {discordant}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: Values that should be IDENTICAL to BERT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CHECK 4: VALUES THAT SHOULD BE IDENTICAL TO BERT")
print("=" * 80)

IDENTICAL_COLS = ['lr', 'per_device_train_batch_size', 'num_train_epochs', 'dtype', 'seed',
                  'per_layer_opt', 'gradient_checkpointing']

ADAPTER_HYPERPARAM_COLS = ['lora_r', 'lora_alpha', 'lora_dropout',
                           'dora_r', 'dora_alpha', 'dora_dropout',
                           'vera_r', 'vera_dropout', 'vera_d_initial',
                           'fourierft_n_frequency', 'fourierft_scaling',
                           'adalora_init_r', 'adalora_target_r', 'adalora_alpha', 'adalora_dropout',
                           'dylora_r', 'dylora_alpha', 'dylora_dropout']

# Spectral params that were INTENTIONALLY changed
SPECTRAL_CHANGED = ['spectral_freq_mode', 'spectral_freq_exponent', 'spectral_scaling', 'spectral_d_initial']
SPECTRAL_SAME = ['spectral_p', 'spectral_q', 'spectral_dropout', 'spectral_factored_rank', 'spectral_learn_scaling']

print("\n--- 4a: Checking identical hyperparams (non-spectral methods) ---")
for task in TASKS:
    for method in NON_SPECTRAL_METHODS:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) == 0 or len(r) == 0:
            continue

        for col in IDENTICAL_COLS + ADAPTER_HYPERPARAM_COLS:
            b_val = b[col].values[0]
            r_val = r[col].values[0]
            # Handle NaN comparison
            if pd.isna(b_val) and pd.isna(r_val):
                continue
            if str(b_val) != str(r_val):
                print(f"  MISMATCH: {method}/{task} col={col}: BERT={b_val} vs RoBERTa={r_val}")

print("\n--- 4b: Checking spectral params (intentionally changed) ---")
for task in TASKS:
    b = bert_std[(bert_std['method'] == 'spectral') & (bert_std['task'] == task)]
    r = roberta_std[(roberta_std['method'] == 'spectral') & (roberta_std['task'] == task)]
    if len(b) == 0 or len(r) == 0:
        continue

    print(f"\n  Task: {task}")
    for col in SPECTRAL_CHANGED:
        b_val = b[col].values[0]
        r_val = r[col].values[0]
        print(f"    {col:30s}: BERT={str(b_val):15s}  RoBERTa={str(r_val):15s}  {'SAME' if str(b_val)==str(r_val) else 'CHANGED'}")

    for col in SPECTRAL_SAME:
        b_val = b[col].values[0]
        r_val = r[col].values[0]
        if pd.isna(b_val) and pd.isna(r_val):
            continue
        if str(b_val) != str(r_val):
            print(f"    MISMATCH in should-be-same: {col}: BERT={b_val} vs RoBERTa={r_val}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5: Timestamp realism
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CHECK 5: TIMESTAMP REALISM")
print("=" * 80)

roberta_std_sorted = roberta_std.sort_values('timestamp')
print("\n  RoBERTa timestamps (sorted):")
for _, row in roberta_std_sorted.iterrows():
    print(f"    {row['timestamp']}  {row['name']}")

# Check if timestamps are all on the same day
timestamps = pd.to_datetime(roberta_std['timestamp'])
dates = timestamps.dt.date.unique()
print(f"\n  Unique dates: {sorted(dates)}")

# Check chronological order vs expected order (short tasks should finish before long tasks)
print(f"\n  Checking if task runtime vs timestamp order makes sense:")
for date in sorted(dates):
    day_rows = roberta_std[pd.to_datetime(roberta_std['timestamp']).dt.date == date].sort_values('timestamp')
    if len(day_rows) > 1:
        print(f"\n    Date: {date}")
        prev_time = None
        for _, row in day_rows.iterrows():
            ts = pd.to_datetime(row['timestamp'])
            runtime = row['total_training_time_sec']
            if prev_time is not None:
                gap = (ts - prev_time).total_seconds()
                if gap < 0:
                    print(f"      SMELL: Timestamps go backwards! {row['name']}")
            prev_time = ts
            print(f"      {row['timestamp']}  runtime={runtime:>10.1f}s  {row['name']}")

# Check if RoBERTa timestamps are plausible (after BERT, on reasonable dates)
bert_max_ts = pd.to_datetime(bert_std['timestamp']).max()
rob_min_ts = pd.to_datetime(roberta_std['timestamp']).min()
print(f"\n  BERT latest timestamp:   {bert_max_ts}")
print(f"  RoBERTa earliest timestamp: {rob_min_ts}")
if rob_min_ts < bert_max_ts:
    print(f"  NOTE: Some RoBERTa timestamps overlap with BERT (might be fine if parallel)")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6: Name column correctness
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CHECK 6: NAME COLUMN CORRECTNESS")
print("=" * 80)

for _, row in roberta_std.iterrows():
    name = row['name']
    method = row['method']
    task = row['task']
    model = row['model_name_or_path']
    dtype = row['dtype']

    # Expected pattern: {method}_roberta-base_{task}_{dtype} or spectral_p16_roberta-base_{task}_{dtype}
    if method == 'spectral':
        expected = f"spectral_p16_roberta-base_{task}_{dtype}"
    elif method == 'base':
        expected = f"base_roberta-base_{task}_{dtype}"
    else:
        expected = f"{method}_roberta-base_{task}_{dtype}"

    if name != expected:
        print(f"  NAME MISMATCH: got '{name}', expected '{expected}'")

# Check BERT names for comparison
print("\n  BERT name patterns for reference:")
for _, row in bert_std.head(5).iterrows():
    print(f"    {row['name']}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7: Empty/NA patterns
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CHECK 7: EMPTY/NA PATTERNS PER TASK")
print("=" * 80)

METRIC_COLS = ['accuracy', 'f1', 'matthews_correlation', 'pearson', 'spearmanr']

for task in TASKS:
    print(f"\n  Task: {task}")
    bert_task = bert_std[bert_std['task'] == task]
    rob_task = roberta_std[roberta_std['task'] == task]

    for col in METRIC_COLS:
        bert_has = bert_task[col].notna().any()
        bert_all_na = bert_task[col].isna().all()
        rob_has = rob_task[col].notna().any()
        rob_all_na = rob_task[col].isna().all()

        if bert_all_na != rob_all_na:
            print(f"    MISMATCH: {col}: BERT all-NA={bert_all_na}, RoBERTa all-NA={rob_all_na}")

        # More detailed: check per-method
        for method in STANDARD_METHODS:
            b = bert_task[bert_task['method'] == method]
            r = rob_task[rob_task['method'] == method]
            if len(b) == 0 or len(r) == 0:
                continue
            b_na = b[col].isna().values[0]
            r_na = r[col].isna().values[0]
            if b_na != r_na:
                print(f"    MISMATCH: {method}/{col}: BERT NA={b_na}, RoBERTa NA={r_na}")

# Expected pattern check
print("\n  Expected metric patterns:")
EXPECTED_METRICS = {
    'cola': {'has': ['matthews_correlation'], 'missing': ['accuracy', 'f1', 'pearson', 'spearmanr']},
    'mrpc': {'has': ['accuracy', 'f1'], 'missing': ['matthews_correlation', 'pearson', 'spearmanr']},
    'rte': {'has': ['accuracy'], 'missing': ['f1', 'matthews_correlation', 'pearson', 'spearmanr']},
    'stsb': {'has': ['pearson', 'spearmanr'], 'missing': ['accuracy', 'f1', 'matthews_correlation']},
    'sst2': {'has': ['accuracy'], 'missing': ['f1', 'matthews_correlation', 'pearson', 'spearmanr']},
    'boolq': {'has': ['accuracy'], 'missing': ['f1', 'matthews_correlation', 'pearson', 'spearmanr']},
    'qnli': {'has': ['accuracy'], 'missing': ['f1', 'matthews_correlation', 'pearson', 'spearmanr']},
    'cb': {'has': ['accuracy', 'f1'], 'missing': ['matthews_correlation', 'pearson', 'spearmanr']},
}

for task, expected in EXPECTED_METRICS.items():
    rob_task = roberta_std[roberta_std['task'] == task]
    for col in expected['has']:
        if rob_task[col].isna().all():
            print(f"    ERROR: {task} should have {col} but all are NA in RoBERTa!")
    for col in expected['missing']:
        if rob_task[col].notna().any():
            print(f"    ERROR: {task} should NOT have {col} but some are present in RoBERTa!")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 8: Boundary checks
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CHECK 8: BOUNDARY CHECKS")
print("=" * 80)

for col in METRIC_COLS:
    vals = roberta_std[col].dropna().values.astype(float)
    if len(vals) > 0:
        above_1 = vals[vals > 1.0]
        below_0 = vals[vals < 0.0]
        nans = roberta_std[col].isna().sum()

        if len(above_1) > 0:
            print(f"  ERROR: {col} has values > 1.0: {above_1}")
        if len(below_0) > 0:
            print(f"  ERROR: {col} has negative values: {below_0}")

        print(f"  {col}: min={vals.min():.6f}, max={vals.max():.6f}, count={len(vals)}, NAs={nans}")

# Check for NaN in non-metric columns that should never be NaN
for col in ['total_training_time_sec', 'param_mem_mib', 'opt_mem_mib', 'runtime_mem_mib', 'peak_mem_mib']:
    nans = roberta_std[col].isna().sum()
    if nans > 0:
        print(f"  ERROR: {col} has {nans} NaN values!")

# ─────────────────────────────────────────────────────────────────────────────
# DEEP DIVE: Statistical tests for interpolation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("DEEP DIVE: STATISTICAL INTERPOLATION DETECTION")
print("=" * 80)

print("\n--- Memory offset analysis (RoBERTa - BERT) ---")
for col in ['param_mem_mib', 'opt_mem_mib', 'runtime_mem_mib', 'peak_mem_mib', 'theoretical_mem_mib']:
    print(f"\n  Column: {col}")
    offsets = []
    ratios = []
    for task in TASKS:
        for method in NON_SPECTRAL_METHODS:
            b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
            r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
            if len(b) == 0 or len(r) == 0:
                continue
            b_val = float(b[col].values[0])
            r_val = float(r[col].values[0])
            offset = r_val - b_val
            ratio = r_val / b_val if b_val != 0 else np.nan
            offsets.append(offset)
            ratios.append(ratio)
            # Print per-method for param_mem

    o = np.array(offsets)
    r_arr = np.array([x for x in ratios if not np.isnan(x)])
    print(f"    Offsets - Mean: {o.mean():.4f}, Std: {o.std():.4f}, Min: {o.min():.4f}, Max: {o.max():.4f}")
    if len(r_arr) > 0:
        print(f"    Ratios  - Mean: {r_arr.mean():.6f}, Std: {r_arr.std():.6f}, Min: {r_arr.min():.6f}, Max: {r_arr.max():.6f}")

    # Check if offsets are suspiciously constant
    if o.std() < 0.01 and len(o) > 3:
        print(f"    >>> SMELL: Offsets are nearly constant (std < 0.01)! Possible uniform scaling.")

print("\n--- Checking if RoBERTa opt_mem values are systematically derived ---")
print("  (In BERT, adapter opt_mem alternates 0.29/0.30 for lora. If RoBERTa has a uniform transformation, that's a smell.)")
for method in ['lora', 'dylora', 'adalora']:
    print(f"\n  Method: {method}")
    for task in TASKS:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) == 0 or len(r) == 0:
            continue
        b_val = float(b['opt_mem_mib'].values[0])
        r_val = float(r['opt_mem_mib'].values[0])
        print(f"    {task:8s}: BERT={b_val:.4f}  RoBERTa={r_val:.4f}  ratio={r_val/b_val:.4f}")

print("\n--- avg_step_time analysis ---")
for task in TASKS:
    print(f"\n  Task: {task}")
    for method in STANDARD_METHODS:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) == 0 or len(r) == 0:
            continue
        b_val = float(b['avg_step_time'].values[0])
        r_val = float(r['avg_step_time'].values[0])
        ratio = r_val / b_val if b_val > 0 else np.nan
        print(f"    {method:12s}: BERT={b_val:.4f}  RoBERTa={r_val:.4f}  ratio={ratio:.4f}")

print("\n--- Checking for exact digit patterns in RoBERTa metrics ---")
print("  (Real metrics from averaged seeds should have many decimal places)")
for _, row in roberta_std.iterrows():
    for col in METRIC_COLS:
        val = row[col]
        if pd.notna(val):
            val = float(val)
            s = f"{val:.15f}"
            # Count trailing zeros after removing leading "0."
            stripped = s.rstrip('0')
            sig_digits = len(stripped) - 2 if '.' in stripped else len(stripped)
            if sig_digits < 4 and val != 0.0:
                print(f"  FEW SIG DIGITS: {row['name']} {col}={val} ({sig_digits} sig digits after decimal)")

# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION ANALYSIS: Are RoBERTa - BERT deltas too correlated?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS: RoBERTa-BERT deltas across tasks")
print("=" * 80)

# For each method, compute deltas across tasks
for method in NON_SPECTRAL_METHODS:
    deltas = {}
    for task in TASKS:
        metric = PRIMARY_METRIC[task]
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) > 0 and len(r) > 0:
            b_val = b[metric].values[0]
            r_val = r[metric].values[0]
            if pd.notna(b_val) and pd.notna(r_val):
                deltas[task] = float(r_val) - float(b_val)

    if deltas:
        d_vals = list(deltas.values())
        print(f"\n  {method:12s} deltas: {dict((k, f'{v:+.6f}') for k, v in deltas.items())}")
        print(f"    Mean: {np.mean(d_vals):+.6f}, Std: {np.std(d_vals):.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK: Are timing values just BERT * constant + noise?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TIMING INTERPOLATION DETECTION")
print("=" * 80)

all_time_ratios = []
for task in TASKS:
    for method in STANDARD_METHODS:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) == 0 or len(r) == 0:
            continue
        b_val = float(b['total_training_time_sec'].values[0])
        r_val = float(r['total_training_time_sec'].values[0])
        if b_val > 0:
            all_time_ratios.append({'task': task, 'method': method, 'ratio': r_val / b_val})

if all_time_ratios:
    ratios_df = pd.DataFrame(all_time_ratios)
    print(f"\n  Overall timing ratio stats:")
    print(f"    Mean: {ratios_df['ratio'].mean():.6f}")
    print(f"    Std:  {ratios_df['ratio'].std():.6f}")
    print(f"    Min:  {ratios_df['ratio'].min():.6f}")
    print(f"    Max:  {ratios_df['ratio'].max():.6f}")

    # Per-task average
    print(f"\n  Per-task average timing ratio:")
    for task in TASKS:
        t_ratios = ratios_df[ratios_df['task'] == task]['ratio']
        if len(t_ratios) > 0:
            print(f"    {task:8s}: mean={t_ratios.mean():.6f}, std={t_ratios.std():.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FINAL: CHECK FOR CONSTANT OFFSET INTERPOLATION FORMULA")
print("=" * 80)

print("\nIf RoBERTa = BERT + constant_offset_per_task, then within a task,")
print("all methods should have the SAME delta. Let's check:")

for task in TASKS:
    metric = PRIMARY_METRIC[task]
    deltas = []
    methods_with_delta = []
    for method in NON_SPECTRAL_METHODS:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) > 0 and len(r) > 0:
            b_val = b[metric].values[0]
            r_val = r[metric].values[0]
            if pd.notna(b_val) and pd.notna(r_val):
                deltas.append(float(r_val) - float(b_val))
                methods_with_delta.append(method)

    if len(deltas) > 1:
        d = np.array(deltas)
        print(f"\n  {task} ({metric}):")
        for m, dd in zip(methods_with_delta, deltas):
            print(f"    {m:12s}: delta = {dd:+.8f}")
        print(f"    Range: {d.max() - d.min():.8f}")
        print(f"    Std:   {d.std():.8f}")
        if d.std() < 0.001:
            print(f"    >>> CRITICAL SMELL: Near-constant delta across methods!")

print("\n\n" + "=" * 80)
print("CHECK COMPLETE")
print("=" * 80)
