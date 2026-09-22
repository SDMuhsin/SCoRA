#!/usr/bin/env python
"""THE READER for `results/clm_wikitext.csv` -- the causal-LM (adapter-only) cells.

⛔ It reads; it launches nothing and it writes nothing but stdout.

WHAT IT ENFORCES, and why each rule is here:

* **Coverage before ranking** (`PROCESS.md` 1.4).  A cell with fewer than the demanded
  seeds is listed as INCOMPLETE and is NEVER given a median.  This defect family has
  bitten five times and always biases toward whatever is further ahead in the queue.
* **One-sided sign test at n=5, 5/5 or nothing** (`PROCESS.md` 1.3).  Paired per seed.
  4/5 is NOT a win and is printed as such.  No t-test exists in this file.
* **Lower perplexity is better**, so every sign test is oriented explicitly and the
  orientation is printed with the verdict -- `[R.52 7a]`'s lesson is that a threshold
  read off a prior rather than off the document is how a verdict gets inverted.
* **The budget-binds flag.**  The reported number is the MIN over epochs; if the argmin
  is the LAST epoch the run was still improving and the number is a LOWER BOUND on what
  the arm can do.  Printed per cell, never silently dropped.
* ⛔⛔ **The UNBRACKETED-recipe label.**  `[R.315]`'s Stage 0 ended with its screen still
  improving at the top rung, so the selected learning rate is **not bracketed** and one
  arm (`fftm`) is further from its own optimum than the other (`scora`).  The reader
  prints that on every study row, because a Half-B comparison read without it would look
  like a clean attribution when `PROCESS 5` test 4's fairness standard is not fully met.
* **The degenerate column is NOT the GLUE one.**  `degen_frac_vs_init` counts epochs no
  better than the FROZEN BACKBONE's own perplexity (`train_clm.degenerate_epochs`).  It
  is printed under that name so it can never be quoted as GLUE's floor statistic.

Usage:
    env/bin/python scripts/clm_read.py [--csv PATH] [--name RUN] [--seeds 5]
    env/bin/python scripts/clm_read.py --pair scora fftm
    env/bin/python scripts/clm_read.py --selftest
"""
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "results", "clm_wikitext.csv")

# n=5, one-sided sign test.  p = 2^-5 = 0.03125 at 5/5 -- the floor, and the ONLY
# outcome this repo certifies.  (CONTEXT.md 2, PROCESS.md 1.3, [USER DECISION 2026-08-17].)
GATE_N = 5


def load(csv_path=DEFAULT_CSV, name=None):
    import pandas as pd
    if not os.path.isfile(csv_path):
        raise SystemExit(f"FAIL CLOSED: {csv_path} does not exist -- nothing has been run.")
    df = pd.read_csv(csv_path)
    if name:
        df = df[df["name"] == name]
    return df


def cell_table(df, want_seeds=GATE_N):
    """One row per (name, arm): median ppl, spread, coverage, budget-binds flag."""
    import numpy as np
    rows = []
    for (nm, arm), g in df.groupby(["name", "arm"], sort=True):
        g = g.drop_duplicates(subset=["seed"])
        ppl = g["ppl"].astype(float)
        finite = ppl[np.isfinite(ppl)]
        last_epoch_argmin = int(((g["best_epoch"].astype(int) + 1)
                                 == g["n_epochs"].astype(int)).sum())
        rows.append({
            "name": nm, "arm": arm,
            "n_seeds": int(len(g)), "complete": bool(len(g) >= want_seeds),
            "ppl_median": float(finite.median()) if len(finite) else float("nan"),
            "ppl_min": float(finite.min()) if len(finite) else float("nan"),
            "ppl_max": float(finite.max()) if len(finite) else float("nan"),
            "ppl_spread": (float(finite.max() - finite.min()) if len(finite) else float("nan")),
            "ppl_sd": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
            "ppl_init": float(g["ppl_init"].astype(float).median()),
            "rel_drop_median": float(g["ppl_rel_drop"].astype(float).median()),
            "degen_frac": float(g["degen_frac_vs_init"].astype(float).median()),
            "n_trainable": int(g["n_trainable"].astype(int).median()),
            "n_non_adapter": int(g["n_non_adapter"].astype(int).max()),
            "budget_binds": last_epoch_argmin,
        })
    return rows


def sign_test(df, arm_a, arm_b, name=None):
    """Paired, per seed, ONE-SIDED: how often does `arm_a` reach a LOWER perplexity?

    Returns (wins, n, verdict).  ⛔ Only wins == n == 5 certifies (p = 0.03125).
    """
    d = df if name is None else df[df["name"] == name]
    a = d[d["arm"] == arm_a].set_index("seed")["ppl"].astype(float)
    b = d[d["arm"] == arm_b].set_index("seed")["ppl"].astype(float)
    seeds = sorted(set(a.index) & set(b.index))
    wins = sum(1 for s in seeds if a[s] < b[s])
    n = len(seeds)
    if n < GATE_N:
        verdict = f"INCOMPLETE ({n}/{GATE_N} paired seeds) -- NOT a verdict"
    elif wins == n:
        verdict = f"CERTIFIED {arm_a} < {arm_b} (lower ppl) {wins}/{n}, p = {2.0 ** -n:.5f}"
    elif wins == 0:
        verdict = f"CERTIFIED {arm_b} < {arm_a} (lower ppl) {n}/{n}, p = {2.0 ** -n:.5f}"
    else:
        verdict = (f"NOT CERTIFIED: {arm_a} wins {wins}/{n}. "
                   f"⛔ 4/5 IS NOT A WIN (PROCESS.md 1.3).")
    return wins, n, verdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--name", default=None, help="restrict to one run label")
    ap.add_argument("--seeds", type=int, default=GATE_N)
    ap.add_argument("--pair", nargs=2, metavar=("ARM_A", "ARM_B"), default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    df = load(a.csv, a.name)
    rows = cell_table(df, a.seeds)
    print(f"{'name':16s} {'arm':10s} {'n':>2s} {'ppl (med)':>10s} {'spread':>8s} "
          f"{'sd':>7s} {'init':>8s} {'drop%':>6s} {'degen':>6s} {'params':>8s} flags")
    for r in rows:
        flags = []
        if not r["complete"]:
            flags.append(f"⛔INCOMPLETE {r['n_seeds']}/{a.seeds} -- no ranking")
        if r["budget_binds"]:
            flags.append(f"⚠budget binds in {r['budget_binds']}/{r['n_seeds']} seeds "
                         f"(argmin at last epoch => LOWER BOUND)")
        if r["n_non_adapter"]:
            flags.append(f"⛔ {r['n_non_adapter']} NON-ADAPTER trainable params")
        print(f"{r['name'][:16]:16s} {r['arm'][:10]:10s} {r['n_seeds']:2d} "
              f"{r['ppl_median']:10.4f} {r['ppl_spread']:8.4f} {r['ppl_sd']:7.4f} "
              f"{r['ppl_init']:8.4f} {100 * r['rel_drop_median']:6.2f} "
              f"{r['degen_frac']:6.2f} {r['n_trainable']:8d} {'; '.join(flags)}")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import r315_plan as _P
        if getattr(_P, "STAGE0_UNBRACKETED", False) and any(
                str(r["name"]).startswith("r315") and "-s0-" not in str(r["name"])
                for r in rows):
            print(f"\n⛔ UNBRACKETED RECIPE: every row above ran at lr = {_P.STUDY_LR:g}, "
                  f"selected by a screen that was STILL IMPROVING at its top rung.\n"
                  f"   [measured] per-doubling gain at the top of the ladder was 68% of "
                  f"initial for fftm vs 29% for scora, and the scora-fftm gap narrowed "
                  f"monotonically 2.760 -> 1.898 across it.\n"
                  f"   ⇒ The step favours scora. A scora-over-fftm win here is a real "
                  f"measurement at a matched step, NOT a clean attribution.")
    except Exception:
        pass

    if a.pair:
        w, n, v = sign_test(df, a.pair[0], a.pair[1], a.name)
        print(f"\nsign test (lower perplexity wins, one-sided, paired by seed): {v}")
    return 0


###############################################################################
#                                  selftest                                   #
###############################################################################
def _frame(rows):
    import pandas as pd
    cols = dict(name="t", model="m", dataset="d", ppl_init=50.0, n_epochs=3,
                best_epoch=0, degen_frac_vs_init=0.0, n_trainable=9216,
                n_non_adapter=0, ppl_rel_drop=0.1)
    out = []
    for r in rows:
        d = dict(cols); d.update(r); out.append(d)
    return pd.DataFrame(out)


def _selftest():
    passed = failed = 0

    def check(nm, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1; print(f"  ✅ {nm}")
        else:
            failed += 1; print(f"  ⛔ {nm} {detail}")

    # ⭐ THE FIXTURE PAIR THAT PROVES THE TEST IS PAIRED: identical medians, opposite
    # verdicts.  A reader that compared medians instead of pairs passes the first and
    # fails the second, and nothing else in this file would notice.
    same_median = [dict(arm="A", seed=s, ppl=p) for s, p in
                   zip(range(42, 47), [10.0, 11.0, 12.0, 13.0, 14.0])]
    b_worse = [dict(arm="B", seed=s, ppl=p) for s, p in
               zip(range(42, 47), [10.1, 11.1, 12.1, 13.1, 14.1])]
    b_mixed = [dict(arm="B", seed=s, ppl=p) for s, p in
               zip(range(42, 47), [9.9, 11.1, 12.1, 13.1, 14.1])]
    f1, f2 = _frame(same_median + b_worse), _frame(same_median + b_mixed)
    med1 = [r for r in cell_table(f1) if r["arm"] == "B"][0]["ppl_median"]
    med2 = [r for r in cell_table(f2) if r["arm"] == "B"][0]["ppl_median"]
    check("G1 the two fixtures have the SAME median for B (so a median-only reader "
          "cannot tell them apart)", abs(med1 - med2) < 1e-9, f"{med1} vs {med2}")
    w1, n1, v1 = sign_test(f1, "A", "B")
    w2, n2, v2 = sign_test(f2, "A", "B")
    check("G1b A beats B 5/5 on the first fixture -> CERTIFIED",
          (w1, n1) == (5, 5) and "CERTIFIED" in v1, v1)
    check("G1c A beats B 4/5 on the second -> NOT CERTIFIED, explicitly",
          (w2, n2) == (4, 5) and "NOT CERTIFIED" in v2, v2)

    # G2 -- orientation.  LOWER perplexity is the win; a reader with the sign flipped
    # would pass every test above if they were symmetric, so assert the direction.
    w, n, v = sign_test(f1, "B", "A")
    check("G2 the test is oriented: B (higher ppl) wins 0/5 and the verdict names A",
          w == 0 and "CERTIFIED" in v and v.split()[1] == "A", v)

    # G3 -- coverage gates ranking (PROCESS.md 1.4).
    short = _frame([dict(arm="A", seed=s, ppl=10.0 + s) for s in range(42, 45)])
    rows = cell_table(short)
    check("G3 a 3-seed cell is INCOMPLETE", rows[0]["complete"] is False)
    w, n, v = sign_test(_frame(same_median[:3] + b_worse[:3]), "A", "B")
    check("G3b a 3-paired-seed sign test refuses to return a verdict",
          "INCOMPLETE" in v and "NOT a verdict" in v, v)

    # G4 -- the budget-binds flag fires exactly when the argmin is the last epoch.
    binds = _frame([dict(arm="A", seed=s, ppl=10.0, best_epoch=2, n_epochs=3)
                    for s in range(42, 47)])
    free = _frame([dict(arm="A", seed=s, ppl=10.0, best_epoch=1, n_epochs=3)
                   for s in range(42, 47)])
    check("G4 argmin at the last epoch flags all 5 seeds",
          cell_table(binds)[0]["budget_binds"] == 5)
    check("G4b an interior argmin flags none",
          cell_table(free)[0]["budget_binds"] == 0)

    # G5 -- a non-adapter trainable parameter is surfaced, never averaged away.
    dirty = _frame([dict(arm="A", seed=s, ppl=10.0, n_non_adapter=(1 if s == 44 else 0))
                    for s in range(42, 47)])
    check("G5 ONE seed with a non-adapter trainable param marks the whole cell",
          cell_table(dirty)[0]["n_non_adapter"] == 1)

    # G6 -- duplicate seed rows cannot inflate coverage.
    dup = _frame([dict(arm="A", seed=42, ppl=10.0)] * 5)
    check("G6 five copies of seed 42 count as ONE seed, not five",
          cell_table(dup)[0]["n_seeds"] == 1)

    # G7 -- spread/sd are computed over finite values only and reported, not hidden.
    r = cell_table(f1)[0]
    check("G7 spread = max - min", abs(r["ppl_spread"] - (r["ppl_max"] - r["ppl_min"])) < 1e-9)

    print(f"\nselftest: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
