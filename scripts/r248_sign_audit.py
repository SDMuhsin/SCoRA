#!/usr/bin/env python
"""[R.248] SIGN-STABILITY AUDIT of every paired two-arm block in the record.

[R.246] found that STANDING #2's k=512 bound is TRUE under median-of-paired and FALSE
under difference-of-medians -- the two definitions disagree in SIGN, not just magnitude.
That defect was found by hand.  This finds every other instance mechanically.

For each block CSV holding exactly two arms over shared seeds, report:
    diff-of-medians, median-of-paired, wins, and whether the two AGREE IN SIGN.

⛔ Reports, never re-reads.  A sign disagreement does not overturn a verdict; it means
the margin must be quoted under BOTH definitions ([R.99 22], [R.226 2]) and cannot be
used as a one-number bound.  Collapsed seeds are KEPT ([R.222], PROCESS 1.4) and
reported separately.
"""
import argparse, glob, os, statistics, sys, math
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# [R.222] collapse = MAJORITY CLASS on accuracy tasks, NOT metric == 0.
MAJORITY = {"rte": 146 / 277, "mrpc": 0.6838}
METRIC = {"rte": "accuracy", "mrpc": "accuracy", "cola": "matthews_correlation",
          "stsb": "pearson", "sst2": "accuracy", "qnli": "accuracy"}


def metric_of(task, cols):
    m = METRIC.get(str(task), "accuracy")
    return m if m in cols else ("accuracy" if "accuracy" in cols else None)


def audit_frame(df, label, min_seeds=4):
    out = []
    if "task_name" not in df.columns or "optimizer" not in df.columns:
        return out
    for task, tg in df.groupby("task_name"):
        met = metric_of(task, tg.columns)
        if met is None:
            continue
        # an "arm" is optimizer + the knob columns that actually vary in this block
        knobs = [c for c in ("slr_rank", "slr_s", "fourierftmerged_k", "qwha_scaling",
                             "haar_mu", "spectral_freq_exponent", "num_warmup_steps",
                             "weight_decay", "lr", "classifier_lr", "freeze_classifier_dense",
                             "slr_init", "loca_scale")
                 if c in tg.columns and tg[c].nunique(dropna=False) > 1]
        tg = tg.copy()
        tg["_arm"] = tg["optimizer"].astype(str)
        for k in knobs:
            tg["_arm"] += "|" + k + "=" + tg[k].astype(str)
        arms = [a for a, g in tg.groupby("_arm") if g[met].notna().sum() >= min_seeds]
        if len(arms) != 2:
            continue
        a, b = sorted(arms)
        pa = tg[tg["_arm"] == a].dropna(subset=[met]).set_index("seed")[met]
        pb = tg[tg["_arm"] == b].dropna(subset=[met]).set_index("seed")[met]
        pa, pb = pa[~pa.index.duplicated()], pb[~pb.index.duplicated()]
        common = sorted(set(pa.index) & set(pb.index))
        if len(common) < min_seeds:
            continue
        va, vb = [float(pa[s]) for s in common], [float(pb[s]) for s in common]
        dl = [y - x for x, y in zip(va, vb)]
        dom = statistics.median(vb) - statistics.median(va)
        mp = statistics.median(dl)
        maj = MAJORITY.get(str(task))
        dead = sum(1 for v in va + vb if maj is not None and abs(v - maj) < 1e-4)
        dead += sum(1 for v in va + vb if met == "matthews_correlation" and abs(v) < 1e-9)
        agree = (dom > 0) == (mp > 0) or (dom == 0 and mp == 0)
        out.append(dict(block=label, task=str(task), n=len(common),
                        arm_a=a[:44], arm_b=b[:44], dom=dom, paired=mp,
                        wins=sum(1 for d in dl if d > 0), dead=dead, agree=agree))
    return out


def main(root="scratchpad"):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) < 8:
            continue
        rows += audit_frame(df, os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f))
    if not rows:
        print("no two-arm paired blocks found"); return rows
    print("=" * 108)
    print("[R.248] SIGN-STABILITY AUDIT -- every two-arm paired block in the record")
    print("=" * 108)
    print(f"{'block':26s} {'task':5s} {'n':>2s} {'diff-of-med':>12s} {'paired':>10s} "
          f"{'wins':>5s} {'dead':>5s}  sign")
    bad = []
    for r in sorted(rows, key=lambda r: (r["agree"], r["block"])):
        flag = "AGREE" if r["agree"] else "*** DISAGREE ***"
        print(f"{r['block'][:26]:26s} {r['task']:5s} {r['n']:>2d} {r['dom']:>+12.4f} "
              f"{r['paired']:>+10.4f} {r['wins']:>3d}/{r['n']:<1d} {r['dead']:>5d}  {flag}")
        if not r["agree"]:
            bad.append(r)
    print("-" * 108)
    print(f"{len(rows)} paired blocks audited; {len(bad)} SIGN-UNSTABLE")
    if bad:
        print("\n⛔ SIGN-UNSTABLE blocks -- each must be quoted under BOTH definitions")
        print("   ([R.99 22], [R.226 2]) and may NOT be used as a one-number bound:")
        for r in bad:
            print(f"   * {r['block']} [{r['task']}]  diff-of-med {r['dom']:+.4f} vs "
                  f"paired {r['paired']:+.4f}, wins {r['wins']}/{r['n']}, dead cells {r['dead']}")
    print("\n⚠️ Collapsed cells are COUNTED and KEPT ([R.222], PROCESS 1.4), never dropped.")
    print("⛔ This audit REPORTS; it re-reads no verdict.")
    return rows


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    # the [R.246] k=512 case, reconstructed -- the audit MUST flag it
    df = pd.DataFrame({
        "task_name": ["rte"] * 10, "seed": [41,42,43,44,45]*2,
        "optimizer": ["adamw-fourierftmerged"]*5 + ["adamw-slr"]*5,
        "accuracy": [0.761733,0.772563,0.750903,0.783394,0.754513,
                     0.703971,0.527076,0.765343,0.768953,0.768953]})
    r = audit_frame(df, "fixture")
    chk("S1 finds the block", len(r) == 1, str(len(r)))
    chk("S2 [R.246]'s k=512 case is flagged SIGN-UNSTABLE", not r[0]["agree"])
    chk("S3 diff-of-medians reproduces +0.0036", abs(r[0]["dom"] - 0.0036) < 5e-4, f"{r[0]['dom']:+.4f}")
    chk("S4 median-of-paired reproduces -0.0144", abs(r[0]["paired"] + 0.0144) < 5e-4, f"{r[0]['paired']:+.4f}")
    chk("S5 the collapsed RTE seed is counted (majority class, not zero)", r[0]["dead"] == 1, str(r[0]["dead"]))
    chk("S6 wins reproduces 2/5", r[0]["wins"] == 2, str(r[0]["wins"]))
    # a clean block must NOT be flagged
    df2 = df.copy(); df2.loc[5:, "accuracy"] = [0.80, 0.81, 0.79, 0.82, 0.80]
    r2 = audit_frame(df2, "fixture")
    chk("S7 a clean block is not flagged", r2[0]["agree"])
    print("\n" + "=" * 56); print(f"  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    print("=" * 56); return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default="scratchpad"); a = ap.parse_args()
    sys.exit(selftest() if a.selftest else (main(a.root) and 0))
