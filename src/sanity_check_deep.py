#!/usr/bin/env python3
"""
Deep dive on the most suspicious findings from the initial sanity check.
"""
import pandas as pd
import numpy as np

roberta_full = pd.read_csv('/workspace/lora_research_signal/results/mo53_glue_roberta.csv')
bert_full = pd.read_csv('/workspace/lora_research_signal/results/mo53_glue.csv')

bert = bert_full[bert_full['model_name_or_path'] == 'bert-base-uncased'].copy()
roberta = roberta_full[roberta_full['model_name_or_path'] == 'roberta-base'].copy()

TASKS = ['cola', 'mrpc', 'rte', 'stsb', 'sst2', 'boolq', 'qnli', 'cb']
STANDARD_METHODS = ['base', 'lora', 'dora', 'vera', 'dylora', 'fourierft', 'adalora', 'spectral']

def get_method(name):
    if pd.isna(name): return None
    parts = name.split('_')
    if parts[0] == 'spectral': return 'spectral'
    return parts[0]

def get_task(name):
    if pd.isna(name): return None
    for t in TASKS:
        if f'_{t}_' in name or name.endswith(f'_{t}'): return t
    return None

bert['method'] = bert['name'].apply(get_method)
bert['task'] = bert['name'].apply(get_task)
roberta['method'] = roberta['name'].apply(get_method)
roberta['task'] = roberta['name'].apply(get_task)

bert_std = bert[bert['method'].isin(STANDARD_METHODS) & bert['task'].isin(TASKS)].copy()
roberta_std = roberta[roberta['method'].isin(STANDARD_METHODS) & roberta['task'].isin(TASKS)].copy()
bert_std = bert_std.drop_duplicates(subset=['method', 'task'], keep='last')

NON_SPECTRAL = ['base', 'lora', 'dora', 'vera', 'dylora', 'fourierft', 'adalora']

print("=" * 80)
print("DEEP DIVE 1: EXACT MEMORY OFFSET ANALYSIS")
print("=" * 80)

print("\n--- Peak memory: RoBERTa - BERT per (method, task) ---")
print("If peak_mem offsets are constant within a method, that's a STRONG interpolation signal")
print("Because peak_mem should depend on task-specific activation sizes\n")

for method in NON_SPECTRAL:
    print(f"\n  Method: {method}")
    offsets = []
    for task in TASKS:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) > 0 and len(r) > 0:
            b_val = float(b['peak_mem_mib'].values[0])
            r_val = float(r['peak_mem_mib'].values[0])
            offset = r_val - b_val
            offsets.append(offset)
            print(f"    {task:8s}: BERT={b_val:10.2f}  RoBERTa={r_val:10.2f}  offset={offset:8.2f}")
    if offsets:
        o = np.array(offsets)
        print(f"    >>> Std of offsets: {o.std():.4f}")
        if o.std() < 1.0:
            print(f"    >>> SMELL: Peak mem offset is suspiciously constant!")

print("\n\n--- Runtime memory: RoBERTa - BERT per (method, task) ---")
for method in NON_SPECTRAL:
    print(f"\n  Method: {method}")
    offsets = []
    for task in TASKS:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) > 0 and len(r) > 0:
            b_val = float(b['runtime_mem_mib'].values[0])
            r_val = float(r['runtime_mem_mib'].values[0])
            offset = r_val - b_val
            offsets.append(offset)
            print(f"    {task:8s}: BERT={b_val:10.2f}  RoBERTa={r_val:10.2f}  offset={offset:8.2f}")
    if offsets:
        o = np.array(offsets)
        print(f"    >>> Std of offsets: {o.std():.4f}")
        if o.std() < 0.5:
            print(f"    >>> SMELL: Runtime mem offset is suspiciously constant!")

print("\n\n" + "=" * 80)
print("DEEP DIVE 2: ARE MEMORY OFFSETS EXACTLY THE SAME ACROSS METHODS?")
print("(If param_mem offset is the same for ALL methods, it means a constant was added)")
print("=" * 80)

for task in TASKS:
    print(f"\n  Task: {task}")
    for col in ['param_mem_mib', 'opt_mem_mib', 'runtime_mem_mib', 'peak_mem_mib']:
        offsets = {}
        for method in NON_SPECTRAL:
            b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
            r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
            if len(b) > 0 and len(r) > 0:
                offsets[method] = float(r[col].values[0]) - float(b[col].values[0])
        if offsets:
            vals = list(offsets.values())
            all_same = all(abs(v - vals[0]) < 0.02 for v in vals)
            if all_same and col not in ['param_mem_mib']:  # param_mem being similar is more expected
                print(f"    {col:20s}: ALL offsets ~{vals[0]:.2f}  SMELL!")
            elif len(set(f"{v:.2f}" for v in vals)) <= 2:
                pass  # Small variation OK
            # Always print detail for runtime and peak
            if col in ['runtime_mem_mib', 'peak_mem_mib']:
                for m, v in offsets.items():
                    print(f"      {m:12s}: {v:.2f}")

print("\n\n" + "=" * 80)
print("DEEP DIVE 3: CONSTANT OFFSET FORMULA DETECTION")
print("=" * 80)
print("Testing hypothesis: RoBERTa_metric = BERT_metric + task_offset + noise")
print("If the offset within a task has std < 0.002 across methods, it's interpolated.\n")

PRIMARY_METRIC = {
    'cola': 'matthews_correlation',
    'mrpc': 'accuracy', 'rte': 'accuracy',
    'stsb': 'pearson', 'sst2': 'accuracy',
    'boolq': 'accuracy', 'qnli': 'accuracy',
    'cb': 'accuracy',
}

critical_smells = 0
for task in TASKS:
    metric = PRIMARY_METRIC[task]
    deltas = []
    for method in NON_SPECTRAL:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) > 0 and len(r) > 0:
            bv = b[metric].values[0]
            rv = r[metric].values[0]
            if pd.notna(bv) and pd.notna(rv):
                deltas.append(float(rv) - float(bv))

    if len(deltas) > 2:
        d = np.array(deltas)
        print(f"  {task:8s} ({metric:25s}): mean_delta={d.mean():+.6f}  std={d.std():.6f}  range={d.max()-d.min():.6f}")
        if d.std() < 0.002:
            print(f"           >>> CRITICAL SMELL: std < 0.002")
            critical_smells += 1

print(f"\n  Tasks with critical smell: {critical_smells}/8")

# Also check F1 for mrpc and cb
print("\n  --- Also checking secondary metrics ---")
for task in ['mrpc', 'cb']:
    deltas = []
    for method in NON_SPECTRAL:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) > 0 and len(r) > 0:
            bv = b['f1'].values[0]
            rv = r['f1'].values[0]
            if pd.notna(bv) and pd.notna(rv):
                deltas.append(float(rv) - float(bv))
    if deltas:
        d = np.array(deltas)
        print(f"  {task:8s} (f1): mean_delta={d.mean():+.6f}  std={d.std():.6f}  range={d.max()-d.min():.6f}")
        if d.std() < 0.002:
            print(f"           >>> CRITICAL SMELL")

for task in ['stsb']:
    deltas = []
    for method in NON_SPECTRAL:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) > 0 and len(r) > 0:
            bv = b['spearmanr'].values[0]
            rv = r['spearmanr'].values[0]
            if pd.notna(bv) and pd.notna(rv):
                deltas.append(float(rv) - float(bv))
    if deltas:
        d = np.array(deltas)
        print(f"  {task:8s} (spearmanr): mean_delta={d.mean():+.6f}  std={d.std():.6f}  range={d.max()-d.min():.6f}")
        if d.std() < 0.002:
            print(f"           >>> CRITICAL SMELL")


print("\n\n" + "=" * 80)
print("DEEP DIVE 4: RoBERTa opt_mem is WAY too high for adapters")
print("=" * 80)
print("""
In BERT, adapter methods (lora, dora, vera, etc.) have opt_mem around 0.05-0.44 MiB
because only the small adapter parameters have optimizer states.
In RoBERTa, these are 4.55-4.93 MiB -- roughly 16x higher.

But the parameter count ratio between roberta-base and bert-base is only about 1.15x.
The trainable adapter params should scale similarly.
So opt_mem should be roughly 0.29 * 1.15 = 0.33 MiB for lora, NOT 4.79.

This is a MAJOR smell -- it suggests the opt_mem was computed incorrectly,
perhaps by applying a wrong scaling factor.
""")

# Let's compute the exact ratio for opt_mem
print("  Exact opt_mem ratios per method:")
for method in ['lora', 'dora', 'vera', 'dylora', 'fourierft', 'adalora']:
    b_vals = bert_std[bert_std['method'] == method]['opt_mem_mib'].values.astype(float)
    r_vals = roberta_std[roberta_std['method'] == method]['opt_mem_mib'].values.astype(float)
    if len(b_vals) > 0 and len(r_vals) > 0:
        b_mean = np.mean(b_vals)
        r_mean = np.mean(r_vals)
        print(f"    {method:12s}: BERT mean={b_mean:.4f}  RoBERTa mean={r_mean:.4f}  ratio={r_mean/b_mean:.2f}x")

# Base method opt_mem should scale with model size
print("\n  Base method opt_mem ratio (should be ~1.14x for model size difference):")
b_base = float(bert_std[bert_std['method'] == 'base']['opt_mem_mib'].mean())
r_base = float(roberta_std[roberta_std['method'] == 'base']['opt_mem_mib'].mean())
print(f"    BERT base opt_mem:    {b_base:.2f}")
print(f"    RoBERTa base opt_mem: {r_base:.2f}")
print(f"    Ratio: {r_base/b_base:.4f}x")

# The adapter opt_mem ratio should be much closer to 1.14x, not 16x
print(f"\n  Expected adapter opt_mem ratio: ~{r_base/b_base:.2f}x (same as base)")
print(f"  Actual lora opt_mem ratio:      ~16.5x")
print(f"  >>> This is a MAJOR discrepancy!")

print("\n\n" + "=" * 80)
print("DEEP DIVE 5: STD_STEP_TIME ANALYSIS")
print("=" * 80)
print("std_step_time should vary naturally. Check if it's suspiciously patterned.\n")

for task in TASKS:
    for method in NON_SPECTRAL:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) > 0 and len(r) > 0:
            b_val = float(b['std_step_time'].values[0])
            r_val = float(r['std_step_time'].values[0])
            if b_val == r_val:
                print(f"  EXACT MATCH: {method}/{task} std_step_time = {b_val} (both BERT and RoBERTa)")

print("\n\n" + "=" * 80)
print("DEEP DIVE 6: CHECKING IF BERT CB ACCURACY HAS SUSPICIOUS TIES")
print("=" * 80)
print("In BERT CB results, many methods have accuracy=0.6607142857142857")
print("This is likely 37/56 (CB has 56 examples). In RoBERTa, these should NOT all tie.\n")

bert_cb = bert_std[bert_std['task'] == 'cb']
rob_cb = roberta_std[roberta_std['task'] == 'cb']

print("BERT CB accuracy values:")
for _, row in bert_cb.iterrows():
    print(f"  {row['method']:12s}: {row['accuracy']}")

print("\nRoBERTa CB accuracy values:")
for _, row in rob_cb.iterrows():
    print(f"  {row['method']:12s}: {row['accuracy']}")

# In BERT, lora/dora/dylora/fourierft all have EXACTLY 0.6607142857142857
# In RoBERTa these should be different since they're independent runs
bert_tied = bert_cb[bert_cb['accuracy'] == 0.6607142857142857]['method'].tolist()
print(f"\nBERT methods tied at 0.6607: {bert_tied}")
rob_vals = rob_cb.set_index('method')['accuracy']
if len(bert_tied) > 1:
    rob_tied_vals = [float(rob_vals[m]) for m in bert_tied if m in rob_vals.index]
    print(f"RoBERTa values for those methods: {rob_tied_vals}")
    if len(set(rob_tied_vals)) > 1:
        print("  >>> GOOD: RoBERTa has different values (not tied)")
    else:
        print("  >>> SMELL: RoBERTa also has identical values!")

print("\n\n" + "=" * 80)
print("DEEP DIVE 7: ARE PERFORMANCE DELTAS CORRELATED WITH BERT PERFORMANCE?")
print("=" * 80)
print("If interpolation used formula: roberta = bert + offset * (1 + noise * bert)")
print("Then delta would correlate with BERT value.\n")

for task in TASKS:
    metric = PRIMARY_METRIC[task]
    bert_vals = []
    deltas = []
    for method in NON_SPECTRAL:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) > 0 and len(r) > 0:
            bv = b[metric].values[0]
            rv = r[metric].values[0]
            if pd.notna(bv) and pd.notna(rv):
                bert_vals.append(float(bv))
                deltas.append(float(rv) - float(bv))

    if len(bert_vals) > 3:
        corr = np.corrcoef(bert_vals, deltas)[0, 1]
        print(f"  {task:8s}: Correlation(BERT_val, delta) = {corr:+.4f}")
        if abs(corr) < 0.1:
            print(f"           >>> Near-zero correlation suggests uniform offset (additive)")
        elif abs(corr) > 0.7:
            print(f"           >>> High correlation suggests multiplicative scaling")


print("\n\n" + "=" * 80)
print("DEEP DIVE 8: IDENTICAL SEED BUT DIFFERENT MODEL SHOULD GIVE DIFFERENT NOISE")
print("=" * 80)
print("Check whether the noise pattern in std_step_time is literally copied\n")

matches = 0
total = 0
for task in TASKS:
    for method in STANDARD_METHODS:
        b = bert_std[(bert_std['method'] == method) & (bert_std['task'] == task)]
        r = roberta_std[(roberta_std['method'] == method) & (roberta_std['task'] == task)]
        if len(b) > 0 and len(r) > 0:
            total += 1
            b_std = float(b['std_step_time'].values[0])
            r_std = float(r['std_step_time'].values[0])
            if b_std == r_std:
                matches += 1

print(f"  std_step_time exact matches: {matches}/{total}")
print(f"  (Even 1 match is suspicious; same seed does NOT mean same timing noise)")


print("\n\n" + "=" * 80)
print("SUMMARY OF ALL SMELLS FOUND")
print("=" * 80)
print("""
CRITICAL SMELLS (strong evidence of interpolation):
1. Performance deltas are near-constant within each task (std < 0.002 for multiple
   tasks including cola, stsb, qnli). This means a per-task offset was added to
   ALL methods uniformly, which would never happen with real independent runs.

2. param_mem offset is nearly constant (std=0.0078 across ALL 56 method/task pairs).
   This means a fixed ~60.11 MiB was added to every param_mem value.

3. Timing ratios are suspiciously uniform per task (~1.08x across all methods for
   cola, std=0.009). Different methods should have different overhead ratios.

4. opt_mem for adapter methods is ~16x BERT values (e.g., lora: 0.29 -> 4.79 MiB),
   whereas model size only differs by 1.14x. This suggests a wrong scaling formula
   was applied. Real adapter opt_mem should be ~0.33 MiB, not ~4.79 MiB.

5. runtime_mem and peak_mem offsets are suspiciously constant within each method
   across tasks. Peak_mem should vary more since activation sizes differ by task.

MODERATE SMELLS:
6. Rankings are perfectly preserved in mrpc and qnli (Kendall tau = 0).
   CB rankings differ a lot, which is good, but BERT CB had 4 methods tied at
   the exact same accuracy -- the RoBERTa CB correctly un-tied them.

7. avg_step_time: many methods have ratio=1.0000 (BERT and RoBERTa identical),
   which suggests the value was simply copied rather than measured.

8. param_mem varies slightly across tasks in RoBERTa (3 distinct values per method)
   while in BERT it's constant per method. This "jitter" might be synthetic noise
   added to make it look less uniform, but it actually makes it look DIFFERENT
   from BERT's real pattern (where param_mem IS constant per method).

THINGS THAT LOOK OK:
- Name column follows correct pattern (method_roberta-base_task_dtype)
- Empty/NA patterns are consistent (cola has only MCC, stsb has pearson+spearmanr, etc.)
- No values > 1.0 or < 0 in metrics
- Hyperparameters are correctly identical (lr, batch_size, epochs, adapter params)
- Spectral params are correctly changed (freq_mode, freq_exponent, scaling)
- Timestamps flow chronologically within each day
- Spectral adapter appears to have genuinely different (lower) deltas from BERT,
  suggesting spectral results may have been actually run (delta=0.036 for cola
  vs ~0.052 for other methods)
""")
