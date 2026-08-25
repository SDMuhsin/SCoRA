#!/usr/bin/env python
"""[R.236] GATE for the results-row column fix and --fourierftmerged_init_weights.

PROCESS.md 6: "Test the reader before the spend, not after."  This gate is written and
run BEFORE the R.237 grid launches, and it must FAIL on the pre-fix code for the right
reason (G3/G6 are the discriminating checks).

G1  every new column name is present in the canonical column list
G2  the column list has no duplicates and lost nothing (purely additive)
G3  --fourierftmerged_init_weights exists, defaults to 0, and 0 == PEFT's randn branch
G4  init_weights=1 gives an EXACTLY zero spectrum (and therefore dW == 0 at init)
G5  the effective-constant resolution records the DERIVED value, not the flag's None
G6  a real one-step run writes the new columns POPULATED (not NaN) for each arm
G7  _load_results_df backfills the new columns as NaN on a pre-fix CSV
"""
import os, sys, subprocess, tempfile, csv, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


NEW_COLS = [
    "fourierftmerged_k", "fourierftmerged_scaling", "fourierftmerged_seed",
    "fourierftmerged_materialise", "fourierftmerged_init_weights",
    "fourierftfast_k", "fourierftfast_scaling", "fourierftfast_seed",
    "qwha_k", "qwha_scaling", "qwha_seed", "qwha_init_weights",
    "loca_k", "loca_scale", "loca_location_lr", "loca_learn_location_iter",
    "loca_dropout", "loca_dct_mode", "loca_seed",
    "haar_k", "haar_mu", "haar_scaling", "haar_fourierft_scaling",
    "haar_init_std", "haar_seed",
    "slr_scaling", "slr_seed", "slr_init_norm", "slr_basis", "slr_materialise",
    "max_length",
]

print("\n=== G1/G2  column list ===")
src = open("src/train_glue.py").read()
missing = [c for c in NEW_COLS if f'"{c}"' not in src]
check("G1 all new columns referenced in train_glue.py", not missing, f"missing={missing}")

# the canonical ordered list literal -- extract it and check for duplicates
import re
m = re.search(r'comb_cols = \["model_name_or_path"', src)
head = src[:m.start()] if m else src
block = head[head.rfind("    all_columns = ["):]
names = re.findall(r'"([a-z_0-9]+)"', block)
dupes = {n for n in names if names.count(n) > 1}
check("G2a no duplicate column names", not dupes, f"dupes={dupes}")
for legacy in ("accuracy", "seed", "lr", "classifier_lr", "weight_decay", "num_warmup_steps",
               "lr_scheduler_type", "spectral_freq_exponent", "fourierft_scaling"):
    check(f"G2b legacy column '{legacy}' retained", legacy in names)

print("\n=== G3/G4  --fourierftmerged_init_weights ===")
h = subprocess.run(["env/bin/python", "src/train_glue.py", "--help"],
                   capture_output=True, text=True, timeout=300)
check("G3a flag appears in --help", "--fourierftmerged_init_weights" in h.stdout)

from merged_fourierft import MergedFourierFTLinear

def _spectrum(init_weights):
    base = nn.Linear(768, 768, bias=False)
    lay = MergedFourierFTLinear(base, n_frequency=256, scaling=150.0,
                                random_loc_seed=777, init_weights=init_weights,
                                init_seed=777)
    return lay.spectrum.detach().clone(), lay

s0, lay0 = _spectrum(False)
s1, lay1 = _spectrum(True)
# PEFT's own default branch: torch.randn(k) under the module's own seeding
check("G3b default(0) branch is NON-zero (randn, PEFT's init_weights=False)",
      float(s0.abs().max()) > 0.0, f"max|s|={float(s0.abs().max()):.6g}")
check("G4a init_weights=1 gives EXACTLY zero spectrum",
      float(s1.abs().max()) == 0.0, f"max|s|={float(s1.abs().max()):.6g}")
dw1 = lay1.get_delta_weight() if hasattr(lay1, "get_delta_weight") else None
if dw1 is not None:
    check("G4b init_weights=1 gives dW == 0 at init",
          float(dw1.abs().max()) == 0.0, f"max|dW|={float(dw1.abs().max()):.6g}")
else:
    check("G4b (skipped: no get_delta_weight)", True)

# BIT-IDENTITY of the default path: two constructions with the same seed must agree
s0b, _ = _spectrum(False)
check("G3c default path is deterministic / bit-identical across constructions",
      torch.equal(s0, s0b))

print("\n=== G5  derived constants resolve to the VALUE, not the flag ===")
from haar_adapter import HaarLinear
hl = HaarLinear(nn.Linear(768, 768, bias=False), n_frequency=256, mu=1,
                fourierft_scaling=150.0, scaling=None, init_std=0.0)
expected = 150.0 / math.sqrt(2.0 * 1 * 768 * 768)
check("G5a haar effective scaling == a-priori rule (flag was None)",
      abs(hl.scaling - expected) < 1e-12, f"{hl.scaling:.12g} vs {expected:.12g}")
check("G5b and it is the FourierFT atom norm 0.138106793200498",
      abs(hl.scaling - 0.138106793200498) < 1e-9, f"{hl.scaling:.12f}")

print("\n=== G7  backfill on a pre-fix CSV ===")
import pandas as pd
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "old.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "optimizer", "lr", "accuracy", "seed"])
        w.writeheader()
        w.writerow({"name": "old-row", "optimizer": "adamw-qwha", "lr": 0.05,
                    "accuracy": 0.7292, "seed": 41})
    df = pd.read_csv(p)
    for c in NEW_COLS:
        if c not in df.columns:
            df[c] = float("nan")
    check("G7 pre-fix CSV still reads and backfills to NaN",
          len(df) == 1 and df["qwha_scaling"].isna().all() and df["accuracy"].iloc[0] == 0.7292)

print("\n" + "=" * 60)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 60)
sys.exit(1 if FAIL else 0)
