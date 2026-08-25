#!/usr/bin/env python
"""[R.268] Cross-arm meta-read of the grid's SHARED OFAT knobs.

The grid gives every arm the same four budget-neutral knobs as one-knob deltas from that
arm's centre: head LR (1e-3, 1e-2), weight decay 0, cosine schedule.  Each arm's block has
been read on its own; they have never been read ACROSS arms.  The question a tuned-baseline
table needs answered is: does any of these knobs have a TRANSFERABLE sign?

⛔ WHAT THESE DELTAS ARE.  [R.259]: the shared centre is off-optimum for every BASELINE
(and on-optimum for SCoRA).  So a baseline's OFAT delta measures "does this knob rescue a
bad config", NOT "does it help at the optimum".  That is the question this file answers,
and it is stated in the output so the number cannot be quoted as the other one.

⛔ n=1 per cell; RTE is quantised at 1/277 = 0.0036 and the 5/5 gate certifies 0.021.
Deltas below the gate are reported as NOT SEPARABLE, never as small effects.

Usage:  env/bin/python scripts/r268_ofat_meta.py [--selftest]
"""
import argparse, importlib.util, os, sys, collections

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_s = importlib.util.spec_from_file_location("r237_read", os.path.join(ROOT, "scripts", "r237_read.py"))
R = importlib.util.module_from_spec(_s); _s.loader.exec_module(R)

GATE = 0.021          # PROCESS 1.3, the user's decision
EX = 1.0 / 277.0      # one RTE eval example
SHARED = ["clf1e-3", "clf1e-2", "wd0", "cosine"]   # the knobs EVERY arm gets


def deltas(cells):
    """{knob: {arm: delta_vs_that_arms_centre}}, only for arms whose centre is complete."""
    out = collections.defaultdict(dict)
    for k, v in cells.items():
        if "-ofat-" not in k:
            continue
        arm, knob = k.split("-ofat-")[0], k.split("-ofat-")[1]
        cen = cells.get(R.CENTRE.get(arm, ""))
        if cen is None:
            continue
        out[knob][arm] = v["acc"] - cen["acc"]
    return out


def report(cells, stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    d = deltas(cells)
    p("=" * 78)
    p("[R.268] SHARED OFAT KNOBS ACROSS ARMS -- is any sign TRANSFERABLE?")
    p("=" * 78)
    p("⛔ These are deltas from the SHARED CENTRE, which [R.259] shows is off-optimum for")
    p("   every BASELINE.  They measure 'does this knob RESCUE a bad config', not 'does it")
    p("   help at the optimum'.  ⛔ n=1; the 5/5 gate certifies 0.021 = 5.8 eval examples.")
    p("")
    verdicts = {}
    for knob in SHARED:
        row = d.get(knob, {})
        if len(row) < 2:
            p(f"  {knob:9s} only {len(row)} arm(s) -- no cross-arm read")
            continue
        pos = sum(1 for v in row.values() if v > 0)
        neg = sum(1 for v in row.values() if v < 0)
        big = {a: v for a, v in row.items() if abs(v) >= GATE}
        sgn = ("CONSISTENT +" if neg == 0 and pos == len(row) else
               "CONSISTENT -" if pos == 0 and neg == len(row) else "⛔ SIGN REVERSES")
        verdicts[knob] = (sgn, len(big))
        p(f"  {knob:9s} " + "  ".join(f"{a}:{v:+.4f}" for a, v in sorted(row.items())))
        p(f"  {'':9s} {sgn}   ({pos}+ / {neg}-)   "
          + (f"⭐ {len(big)} arm(s) exceed the gate: "
             + ", ".join(f"{a} {v:+.4f}" for a, v in sorted(big.items()))
             if big else "⚠️ NO arm exceeds the gate -- none separable"))
        p("")
    p("-" * 78)
    consistent = [k for k, (s, _) in verdicts.items() if s.startswith("CONSISTENT")]
    p(f"knobs with a transferable sign across all read arms: "
      + (", ".join(consistent) if consistent else "⛔ NONE"))
    return verdicts


def selftest():
    ok, bad = [], []
    def chk(n, c, dd=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {dd}" if dd else ""))
    H = lambda a: {"acc": a, "collapsed": False, "near_floor": False}
    import io
    # consistent-positive knob across two arms, both above the gate
    c = {"fftm-lr5e-2-s150": H(0.70), "fftm-ofat-wd0": H(0.73),
         "loca-lr5e-2-a1.0": H(0.60), "loca-ofat-wd0": H(0.65)}
    d = deltas(c)
    chk("M1 deltas are taken against each arm's OWN centre",
        abs(d["wd0"]["fftm"] - 0.03) < 1e-9 and abs(d["wd0"]["loca"] - 0.05) < 1e-9, str(d))
    buf = io.StringIO(); v = report(c, buf)
    chk("M2 an all-positive knob is CONSISTENT +", v["wd0"][0] == "CONSISTENT +", str(v))
    chk("M2b and gate-exceeding arms are named", "exceed the gate" in buf.getvalue())
    c2 = dict(c); c2["loca-ofat-wd0"] = H(0.55)          # now negative
    buf = io.StringIO(); v = report(c2, buf)
    chk("M3 a sign reversal is flagged, not averaged", v["wd0"][0] == "⛔ SIGN REVERSES", str(v))
    chk("M3b and no knob is then called transferable", "⛔ NONE" in buf.getvalue())
    c3 = {"fftm-lr5e-2-s150": H(0.70), "fftm-ofat-cosine": H(0.7036),
          "loca-lr5e-2-a1.0": H(0.60), "loca-ofat-cosine": H(0.6036)}
    buf = io.StringIO(); v = report(c3, buf)
    chk("M4 sub-gate effects are NOT separable, even when consistent in sign",
        v["cosine"][1] == 0 and "NO arm exceeds the gate" in buf.getvalue(), buf.getvalue())
    c4 = {"fftm-ofat-wd0": H(0.73)}                       # centre missing
    chk("M5 an arm with no complete centre yields no delta", deltas(c4) == {} or
        "fftm" not in deltas(c4).get("wd0", {}), str(deltas(c4)))
    buf = io.StringIO(); report({"fftm-lr5e-2-s150": H(0.7), "fftm-ofat-wd0": H(0.72)}, buf)
    chk("M6 a single arm yields no cross-arm read", "no cross-arm read" in buf.getvalue())
    print(f"\n  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    report(R.load(os.path.join(ROOT, "scratchpad", "phaseR", "r237", "csv")))
