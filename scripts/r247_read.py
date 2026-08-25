#!/usr/bin/env python
"""[R.247] READER -- rank reallocation at the k=512 budget. Verdict per R246 5."""
import argparse, csv, glob, os, statistics, sys, math

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scratchpad", "phaseR", "r247", "csv")
SEEDS = [41, 42, 43, 44, 45]
MAJ = 146 / 277                      # [R.222] RTE collapse = majority class
# frozen comparators, re-derived from r208.csv 2026-08-20 ([R.246 1])
FFT512  = {41: 0.761733, 42: 0.772563, 43: 0.750903, 44: 0.783394, 45: 0.754513}
SLR512  = {41: 0.703971, 42: 0.527076, 43: 0.765343, 44: 0.768953, 45: 0.768953}
GATE = 0.021


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception:
            continue
        if not rows:
            continue
        try:
            v = float(rows[-1].get("accuracy", "nan"))
        except (TypeError, ValueError):
            continue
        if math.isnan(v):
            continue
        n = os.path.basename(p)[:-4]
        try:
            out[int(n.rsplit("s", 1)[-1])] = v
        except ValueError:
            pass
    return out


def both(new, ref):
    """(diff-of-medians, median-of-paired, wins) of new vs ref, paired by seed."""
    ks = sorted(set(new) & set(ref))
    dl = [new[k] - ref[k] for k in ks]
    return (statistics.median([new[k] for k in ks]) - statistics.median([ref[k] for k in ks]),
            statistics.median(dl), sum(1 for d in dl if d > 0), len(ks))


def report(cells, stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    p("=" * 74)
    p("[R.247] SCoRA r=2,s=128 @ 512 params/module (RTE) -- verdict per R246 5")
    p("=" * 74)
    p(f"cells: {len(cells)}/5   seeds {sorted(cells)}")
    if len(cells) < 5:
        p("\n*** INCOMPLETE -- NO VERDICT (PROCESS 1.4). ***")
        return None
    dead = [s for s, v in cells.items() if abs(v - MAJ) < 1e-4]
    p(f"\nper-seed: " + "  ".join(f"s{s}={cells[s]:.4f}" for s in SEEDS))
    p(f"median {statistics.median(cells.values()):.4f}   collapsed seeds: "
      f"{dead if dead else 'NONE'}   <- KEPT in every verdict")

    dm_f, mp_f, w_f, n_f = both(cells, FFT512)
    dm_s, mp_s, w_s, n_s = both(cells, SLR512)
    p(f"\nvs FourierFT k=512: diff-of-med {dm_f:+.4f}  paired {mp_f:+.4f}  wins {w_f}/{n_f}")
    p(f"vs SCoRA r=1,s=256: diff-of-med {dm_s:+.4f}  paired {mp_s:+.4f}  wins {w_s}/{n_s}")
    p(f"   (r=1/s=256 baseline was itself SIGN-UNSTABLE vs FourierFT: "
      f"paired -0.0144, diff-of-med +0.0036, wins 2/5  [R.246 1])")

    p("\n-- VERDICT -------------------------------------------------------------")
    v1 = len(dead) == 0
    p(f"  V1 {'PASS' if v1 else 'FAIL'} (my call): zero collapsed seeds -> {len(dead)}/5 dead")
    v2 = dm_s > 0 and mp_s > 0
    p(f"  V2 {'PASS' if v2 else 'FAIL'}: beats r=1/s=256 on BOTH definitions")
    cleared = (dm_f >= GATE and mp_f >= GATE and w_f == 5)
    p(f"  V3 {'PASS' if not cleared else 'FAIL'} (my call): does NOT clear the gate over FourierFT")
    v4 = (dm_f > 0) == (mp_f > 0)
    p(f"  V4 {'PASS' if v4 else 'FAIL'}: k=512 deficit is SIGN-STABLE (both definitions agree)")

    if v1 and not v2:
        p("\n  ⭐ V1 pass + V2 fail is the INFORMATIVE branch: the collapse was NOT what cost")
        p("     SCoRA k=512 -- the deficit is rank-1 CAPACITY, not instability.")
    p("\n  ⛔ V2 passing is NOT route A and changes no STANDING: it would only make")
    p("     limit 1 quantitative, replacing a sign-unstable number with a measured one.")
    p("  ⚠️ COST REGRESSION IS PART OF THE RESULT: 6,936 flops/token vs r=1's 3,936")
    p("     (1.76x dearer); still 6.0x cheaper than FourierFT k=512's 41,414.")
    return {"V1": v1, "V2": v2, "V3": not cleared, "V4": v4}


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    import io, tempfile
    def mk(td, vals):
        for s, v in vals.items():
            with open(os.path.join(td, f"slr-r2s128-s{s}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["name", "accuracy", "seed"]); w.writeheader()
                w.writerow({"name": f"s{s}", "accuracy": v, "seed": s})

    with tempfile.TemporaryDirectory() as td:
        mk(td, {41: 0.77}); buf = io.StringIO()
        chk("S1 incomplete -> no verdict", report(load(td), buf) is None and "INCOMPLETE" in buf.getvalue())
    with tempfile.TemporaryDirectory() as td:   # healthy, beats r=1 on both
        mk(td, {41: 0.77, 42: 0.78, 43: 0.775, 44: 0.79, 45: 0.78})
        buf = io.StringIO(); v = report(load(td), buf)
        chk("S2 healthy arm: V1 and V2 pass", v["V1"] and v["V2"], str(v))
        chk("S3 V4 sign-stability computed", isinstance(v["V4"], bool))
    with tempfile.TemporaryDirectory() as td:   # one collapse -> V1 fails, seed KEPT
        mk(td, {41: 0.77, 42: MAJ, 43: 0.775, 44: 0.79, 45: 0.78})
        buf = io.StringIO(); v = report(load(td), buf)
        chk("S4 collapse detected (majority class, not zero)", not v["V1"])
        chk("S5 collapsed seed reported as KEPT", "KEPT in every verdict" in buf.getvalue())
    with tempfile.TemporaryDirectory() as td:   # a true gate clearance must fail V3
        mk(td, {41: 0.79, 42: 0.80, 43: 0.79, 44: 0.81, 45: 0.80})
        buf = io.StringIO(); v = report(load(td), buf)
        chk("S6 V3 fails only when the gate is genuinely cleared 5/5", v["V3"] is False, str(v))
    with tempfile.TemporaryDirectory() as td:   # V1 pass, V2 fail -> informative branch printed
        mk(td, {41: 0.60, 42: 0.61, 43: 0.60, 44: 0.62, 45: 0.61})
        buf = io.StringIO(); v = report(load(td), buf)
        chk("S7 V1-pass/V2-fail prints the informative-branch note",
            v["V1"] and not v["V2"] and "INFORMATIVE branch" in buf.getvalue())
    print("\n" + "=" * 56); print(f"  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    print("=" * 56); return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir", default=D); a = ap.parse_args()
    sys.exit(selftest() if a.selftest else (report(load(a.dir)) and 0))
