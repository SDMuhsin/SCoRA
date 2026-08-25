#!/usr/bin/env python
"""[R.240] READER -- is STANDING #2's k=64 margin a small-data effect?
Verdict read mechanically off llmdocs/R240_subsample_prereg.md 3.  Selftest first.
"""
import argparse, csv, glob, os, statistics, sys, math

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scratchpad", "phaseR", "r240")
SEEDS = [41, 42, 43, 44, 45]
ARMS = ["fft-k256", "scora-k256", "fft-k64", "scora-k64"]

# frozen comparators [R.238, re-derived from raw logs]
FULL_COLA_K64 = {"dom": +0.0062, "paired": +0.0150, "wins": 4, "asym": 1.0}
RTE_K64       = {"dom": +0.0469, "asym": 3.8}
X1_BAR   = 0.0250      # half-way CoLA -> RTE
X2_BAND  = 0.0100      # |M - 0.0062| < this
DEFN_TOL = 0.0050      # >this apart => DEFINITION-SENSITIVE, no verdict
SD_BAR   = 0.0340      # [R.224] full-CoLA paired sd


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.csv"))):
        name = os.path.basename(p)[:-4]
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception:
            continue
        if not rows:
            continue
        try:
            v = float(rows[-1].get("matthews_correlation", "nan"))
        except (TypeError, ValueError):
            continue
        if not math.isnan(v):
            out[name] = v
    return out


def report(cells, stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    p("=" * 76)
    p("[R.240] CoLA @ N_train=2490 (RTE's size) -- is the k=64 margin a SMALL-DATA effect?")
    p("        verdict per frozen llmdocs/R240_subsample_prereg.md 3")
    p("=" * 76)

    have = {a: [cells.get(f"{a}-s{s}") for s in SEEDS] for a in ARMS}
    complete = {a: all(v is not None for v in have[a]) for a in ARMS}
    p("cells: " + "  ".join(f"{a}={sum(v is not None for v in have[a])}/5" for a in ARMS))
    if not (complete["fft-k64"] and complete["scora-k64"]):
        p("\n*** INCOMPLETE -- the k=64 rung is not finished. NO VERDICT (PROCESS 1.4). ***")
        return None

    f64, s64 = have["fft-k64"], have["scora-k64"]
    deltas = [b - a for a, b in zip(f64, s64)]
    wins = sum(1 for d in deltas if d > 0)
    dom = statistics.median(s64) - statistics.median(f64)
    paired = statistics.median(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else float("nan")
    degen = sum(1 for v in f64 + s64 if abs(v) < 1e-9)

    p(f"\n-- k=64 margin on CoLA@2490 --------------------------------------------")
    p(f"   FourierFT median {statistics.median(f64):.4f}   SCoRA median {statistics.median(s64):.4f}")
    p(f"   diff-of-medians {dom:+.4f}   median-of-paired {paired:+.4f}   wins {wins}/5")
    p(f"   paired sd {sd:.4f}   (full-CoLA [R.224] {SD_BAR:.4f})")
    p(f"   per-seed: {[f'{d:+.4f}' for d in deltas]}")
    p(f"   degenerate (MCC==0) cells: {degen}   <- KEPT, never folded away")
    p(f"\n   comparators: full CoLA {FULL_COLA_K64['dom']:+.4f}/{FULL_COLA_K64['paired']:+.4f} "
      f"({FULL_COLA_K64['wins']}/5)   RTE {RTE_K64['dom']:+.4f}")

    if abs(dom - paired) > DEFN_TOL:
        p(f"\n*** DEFINITION-SENSITIVE: the two definitions differ by {abs(dom-paired):.4f} "
          f"> {DEFN_TOL} -> NO VERDICT (prereg 3, rule 1). ***")
        return "DEFINITION-SENSITIVE"
    if sd > SD_BAR:
        p(f"\n*** UNDERPOWERED: paired sd {sd:.4f} > full-CoLA {SD_BAR:.4f} -> "
          f"report, do not conclude (prereg 3, rule 3). ***")

    M = dom
    p("\n-- VERDICT -----------------------------------------------------------------")
    if M >= X1_BAR and wins >= 4:
        v = "X1"
        p(f"  X1 PASS: M={M:+.4f} >= {X1_BAR} and {wins}/5 >= 4/5")
        p("  => STANDING #2's k=64 margin is SUBSTANTIALLY A SMALL-DATA EFFECT.")
        p("     Its scope sentence must say so. N_train and warmup-shortfall are still")
        p("     entangled -- prereg 3 rule 4's stage 2 (warmup 41) splits them.")
    elif abs(M - FULL_COLA_K64["dom"]) < X2_BAND:
        v = "X2"
        p(f"  X2 PASS: |M - {FULL_COLA_K64['dom']:+.4f}| = {abs(M-FULL_COLA_K64['dom']):.4f} < {X2_BAND}")
        p("  => TASK IDENTITY drives it; N_train AND warmup-shortfall exonerated together.")
        p("     [R.238 4]'s asymmetry lead survives [R.239].")
    else:
        v = "X3"
        p(f"  X3: M={M:+.4f} lies between the branches -- GRADED.")
        p("     Report the interpolation; claim no mechanism.")

    if complete["fft-k256"] and complete["scora-k256"]:
        cf = statistics.median(have["fft-k64"]) - statistics.median(have["fft-k256"])
        cs = statistics.median(have["scora-k64"]) - statistics.median(have["scora-k256"])
        asym = (cf / cs) if cs else float("inf")
        p(f"\n  budget cost 256->64: FourierFT {cf:+.4f}  SCoRA {cs:+.4f}  asymmetry {asym:.1f}x")
        p(f"  X4 {'PASS' if asym > 2.0 else 'fails'}: asymmetry {'>' if asym > 2.0 else '<='} 2.0x "
          f"(full CoLA {FULL_COLA_K64['asym']}x)")
        if asym > 2.0:
            p("  => the asymmetry is a CONSEQUENCE of N_train, not a cause -- kills [R.238 4].")
    else:
        p("\n  X4: k=256 rung incomplete -- asymmetry not reportable.")
    p("\n  ⛔ NOT a route-A claim. This is a SCOPE measurement on an existing result.")
    return v


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n)
        print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    import io, tempfile

    def mk(td, vals):
        for name, v in vals.items():
            with open(os.path.join(td, name + ".csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["name", "matthews_correlation", "seed"])
                w.writeheader(); w.writerow({"name": name, "matthews_correlation": v, "seed": 41})

    def block(fft, sco, k):
        return {**{f"fft-k{k}-s{s}": fft[i] for i, s in enumerate(SEEDS)},
                **{f"scora-k{k}-s{s}": sco[i] for i, s in enumerate(SEEDS)}}

    # T1 incomplete -> no verdict
    with tempfile.TemporaryDirectory() as td:
        mk(td, {"fft-k64-s41": 0.50, "scora-k64-s41": 0.53})
        buf = io.StringIO(); v = report(load(td), buf)
        chk("T1 incomplete k=64 rung -> NO VERDICT", v is None and "INCOMPLETE" in buf.getvalue())

    # T2 X1: margin jumps to RTE-like
    with tempfile.TemporaryDirectory() as td:
        mk(td, block([0.50]*5, [0.53, 0.535, 0.53, 0.532, 0.531], 64))
        buf = io.StringIO(); v = report(load(td), buf)
        chk("T2 big margin -> X1 (small-data effect)", v == "X1", f"got {v}")

    # T3 X2: margin stays at full-CoLA level
    with tempfile.TemporaryDirectory() as td:
        mk(td, block([0.50]*5, [0.5062, 0.5065, 0.5062, 0.5061, 0.5063], 64))
        buf = io.StringIO(); v = report(load(td), buf)
        chk("T3 unchanged margin -> X2 (task identity)", v == "X2", f"got {v}")

    # T4 X3: in between
    with tempfile.TemporaryDirectory() as td:
        mk(td, block([0.50]*5, [0.518]*5, 64))
        buf = io.StringIO(); v = report(load(td), buf)
        chk("T4 intermediate -> X3 (graded)", v == "X3", f"got {v}")

    # T5 definition-sensitivity short-circuits the verdict
    with tempfile.TemporaryDirectory() as td:
        # dom = 0.55-0.52 = +0.030 ; paired median = +0.020 => 0.010 apart, > DEFN_TOL.
        # (The first fixture here produced NO divergence at all -- the two statistics
        # coincide unless the median-ranked PAIR differs from the median-ranked LEVELS.
        # The selftest caught my fixture, not the reader.)
        mk(td, block([0.50, 0.51, 0.52, 0.53, 0.54], [0.62, 0.53, 0.54, 0.55, 0.56], 64))
        buf = io.StringIO(); v = report(load(td), buf)
        chk("T5 definitions far apart -> DEFINITION-SENSITIVE, no X-verdict",
            v == "DEFINITION-SENSITIVE", f"got {v}")

    # T6 X1 requires 4/5 wins, not just a big median
    with tempfile.TemporaryDirectory() as td:
        mk(td, block([0.50]*5, [0.60, 0.60, 0.60, 0.49, 0.48], 64))
        buf = io.StringIO(); v = report(load(td), buf)
        chk("T6 big median but 3/5 wins -> not X1", v != "X1", f"got {v}")

    # T7 X4 asymmetry computed only when the k=256 rung is complete
    with tempfile.TemporaryDirectory() as td:
        d = block([0.50]*5, [0.53]*5, 64); d.update(block([0.56]*5, [0.55]*5, 256))
        mk(td, d)
        buf = io.StringIO(); report(load(td), buf)
        chk("T7 asymmetry reported when k=256 complete", "asymmetry" in buf.getvalue())
    with tempfile.TemporaryDirectory() as td:
        mk(td, block([0.50]*5, [0.53]*5, 64))
        buf = io.StringIO(); report(load(td), buf)
        chk("T7b asymmetry refused when k=256 missing",
            "k=256 rung incomplete" in buf.getvalue())

    # T8 degenerate MCC cells are counted and kept
    with tempfile.TemporaryDirectory() as td:
        mk(td, block([0.50]*5, [0.0, 0.53, 0.53, 0.53, 0.53], 64))
        buf = io.StringIO(); report(load(td), buf)
        chk("T8 degenerate cells counted, not dropped",
            "degenerate (MCC==0) cells: 1" in buf.getvalue())

    print("\n" + "=" * 60)
    print(f"  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad:
        print("  FAILED: " + ", ".join(bad))
    print("=" * 60)
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir", default=os.path.join(D, "csv"))
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else (report(load(a.dir)) and 0))
