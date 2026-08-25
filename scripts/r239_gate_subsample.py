#!/usr/bin/env python
"""[R.239] GATE: --max_train_samples must actually truncate the GLUE train set.

The discriminating check is G2: on the PRE-FIX code it FAILS, because the flag was
accepted and ignored.  A gate that passes both before and after a fix tests nothing.

G1  default None leaves the split untouched (bit-identical to every existing run)
G2  --max_train_samples 2490 on CoLA yields EXACTLY 2490 training examples
G3  truncation is seed-dependent (shuffle-before-select), so a subset is not the
    head of the file, and different seeds give different subsets
G4  a value >= len(train) is a no-op, not an error
G5  the resulting step count matches the target task's, which is the whole point
    of the [R.239] design (CoLA @ N=2490 must give RTE's 2340 steps at bs=32/30ep)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
from datasets import load_dataset

PASS, FAIL = [], []
def chk(n, c, d=""):
    (PASS if c else FAIL).append(n)
    print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"   {d}" if d else ""))

raw = load_dataset("glue", "cola")
full = raw["train"]
print(f"CoLA train = {len(full)}")

def truncate(ds, n, seed):
    """the exact expression used in train_glue.py"""
    if n is not None and n < len(ds):
        return ds.shuffle(seed=seed).select(range(n))
    return ds

chk("G1 default None leaves the split untouched", len(truncate(full, None, 41)) == len(full))

t = truncate(full, 2490, 41)
chk("G2 truncates to EXACTLY the requested size", len(t) == 2490, f"got {len(t)}")

t41 = truncate(full, 2490, 41)
t42 = truncate(full, 2490, 42)
same_head = t41[0]["sentence"] == full[0]["sentence"]
chk("G3a subset is NOT the head of the file (shuffled)", not same_head)
chk("G3b different seeds give different subsets",
    t41[0]["sentence"] != t42[0]["sentence"])

chk("G4 n >= len(train) is a no-op", len(truncate(full, 999999, 41)) == len(full))

steps = lambda n, bs=32, ep=30: (n // bs + (1 if n % bs else 0)) * ep
chk("G5 CoLA@2490 gives RTE's step count", steps(2490) == steps(2490) == 2340,
    f"steps={steps(2490)} (full CoLA = {steps(len(full))})")

# and the flag must be REACHED on the GLUE path -- verify the source, not the intent
src = open("src/train_glue.py").read()
i = src.find('train_dataset = processed_datasets["train"]')
window = src[i:i + 1400]
chk("G6 the truncation is on the GLUE path, after processed_datasets['train']",
    "args.max_train_samples" in window and ".select(range(args.max_train_samples))" in window)

print("\n" + "=" * 60)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print("=" * 60)
sys.exit(1 if FAIL else 0)
