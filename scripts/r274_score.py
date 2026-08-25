#!/usr/bin/env python
"""[R.274] SCORER — applies S1-S5 mechanically to the 10 frozen out-of-sample predictions.

Written BEFORE wave2 completed (14/21 at authoring time) so the rules cannot be
renegotiated once the residuals are visible.  [R.254]'s fragility counts went wrong
precisely because they were hand-read off a printed list ([R.256] 1).

⛔ REFUSES a verdict until wave2 is 21/21 (PROCESS.md 1.4).
⛔ Rules are transcribed from llmdocs/R274_mu_shift_outofsample_prereg.md 3 and are NOT
   parameters of this file: changing them here would be renegotiating a frozen prereg.
"""
import argparse, importlib.util, math, os, statistics, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_s = importlib.util.spec_from_file_location("r237_read", os.path.join(ROOT, "scripts", "r237_read.py"))
R = importlib.util.module_from_spec(_s); _s.loader.exec_module(R)

DELTA = 0.00361          # the fitted mu-shift, frozen
FIT_SD = 0.0068          # sd of the 6 overlap residuals, frozen
S1_TOL, S2_TOL, S4_TOL = 0.0068, 0.6, 0.0136
ARGMAX_CELL = "lr1.5e-1-fs300"
PRED = {                 # frozen point predictions (R274 2)
    "lr1.5e-2-fs150": 0.6390, "lr1.5e-2-fs300": 0.6462, "lr5e-2-fs50": 0.6318,
    "lr5e-2-fs100": 0.6498, "lr5e-2-fs150": 0.6534, "lr5e-2-fs300": 0.6895,
    "lr1.5e-1-fs50": 0.6534, "lr1.5e-1-fs100": 0.6931, "lr1.5e-1-fs150": 0.6968,
    "lr1.5e-1-fs300": 0.7184,
}


def _lr(cell):
    """'lr1.5e-2-fs150' -> 0.015.  ⛔ split('-')[0] gives 'lr1.5e': the exponent's minus
    sign is a separator too.  Use the reader's own tested knob parser instead."""
    import re
    m = re.match(r"lr([0-9][0-9.]*e-?[0-9]+)", cell)
    if not m:
        raise ValueError(f"cannot parse lr from {cell!r}")
    return float(m.group(1))


def score(cells, totals, stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    p("=" * 78); p("[R.274] SCORER -- mu=2 as a pure level shift?  Rules frozen in R274 3.")
    p("=" * 78)
    exp = totals.get("wave2")
    have = [k for k in cells if R.arm_of(k) == "wave2"]
    landed = {c: cells["wave2-" + c]["acc"] for c in PRED if "wave2-" + c in cells}
    for c in sorted(PRED):
        if c in landed:
            p(f"  {c:16s} pred {PRED[c]:.4f}  actual {landed[c]:.4f}  "
              f"resid {landed[c]-PRED[c]:+.4f} ({(landed[c]-PRED[c])*277:+.1f} ex)")
        else:
            p(f"  {c:16s} pred {PRED[c]:.4f}  ⏳ not yet run")
    if exp and len(have) < exp:
        p(f"\n  ⛔ INCOMPLETE: wave2 {len(have)}/{exp}.  NO VERDICT (PROCESS.md 1.4).")
        p(f"     {len(landed)}/{len(PRED)} predicted cells landed -- interim only, NOT scored.")
        return {"verdict": "INCOMPLETE"}
    res = {c: landed[c] - PRED[c] for c in PRED}
    vals = list(res.values())
    mean = statistics.mean(vals)
    mae = statistics.mean(abs(v) for v in vals)
    xs = [math.log(_lr(c)) for c in PRED]; ys = [res[c] for c in PRED]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    r = num / den if den > 0 else 0.0
    plane = {k: v["acc"] for k, v in cells.items()
             if R.arm_of(k) == "wave2" and "-ofat-" not in k}
    amax = max(plane, key=plane.get).split("wave2-")[1]
    s1 = abs(mean) < S1_TOL
    s2 = abs(r) < S2_TOL
    s3 = amax == ARGMAX_CELL
    s4 = mae < S4_TOL
    s5 = s1 and s3 and s4 and not s2
    p("")
    p(f"  mean residual {mean:+.5f} ({mean*277:+.1f} ex)   MAE {mae:.5f}   corr(resid, log lr) {r:+.3f}")
    p(f"  wave2 plane argmax: {amax}")
    p("")
    for n, ok, txt in (("S1", s1, f"|mean resid| {abs(mean):.4f} < {S1_TOL}  (shift transfers)"),
                       ("S2", s2, f"|corr| {abs(r):.3f} < {S2_TOL}  (no interaction with the step)"),
                       ("S3", s3, f"argmax is {ARGMAX_CELL}"),
                       ("S4", s4, f"MAE {mae:.4f} < {S4_TOL}")):
        p(f"  {n}: {'✅ PASS' if ok else '⛔ FAILED'}   {txt}")
    p(f"  S5 (my call: S1,S3,S4 pass AND S2 FAILS): {'✅ CONFIRMED' if s5 else '⛔ FALSIFIED'}")
    p("")
    p("  ⛔ n=1 per cell, seed 41, RTE.  This tests OUR mu=2 rank fix, not published WaveFT.")
    p("  ⛔ Neither WaveFT arm is competitive with tuned FourierFT (0.8195); nothing here changes that.")
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "mean": mean, "mae": mae, "corr": r}


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    import io
    H = lambda a: {"acc": a, "collapsed": False, "near_floor": False}
    # exact predictions => all residuals zero
    perfect = {"wave2-" + c: H(v) for c, v in PRED.items()}
    perfect["wave2-lr5e-3-fs50"] = H(0.58)
    tot = {"wave2": len(perfect)}
    buf = io.StringIO(); r = score(perfect, tot, buf)
    chk("V1 exact predictions pass S1/S2/S4", r["S1"] and r["S2"] and r["S4"], str(r))
    chk("V2 and S3 holds when the argmax is the predicted cell", r["S3"], str(r))
    chk("V3 S5 (my call) is FALSIFIED when S2 passes", r["S5"] is False, str(r))
    # an LR-proportional residual => S2 must fail
    slope = {"wave2-" + c: H(PRED[c] + 0.004 * math.log(_lr(c) / 5e-3)) for c in PRED}
    slope["wave2-lr5e-3-fs50"] = H(0.58)
    buf = io.StringIO(); r = score(slope, tot, buf)
    chk("V4 a residual that GROWS with lr fails S2", r["S2"] is False, f"corr {r['corr']:+.3f}")
    chk("V5 S5 fires only on that pattern", r["S5"] is True or not (r["S1"] and r["S3"] and r["S4"]),
        str(r))
    buf = io.StringIO(); r = score(perfect, {"wave2": 99}, buf)
    chk("V6 an INCOMPLETE arm gets NO VERDICT", r["verdict"] == "INCOMPLETE"
        and "NO VERDICT" in buf.getvalue())
    chk("V7 the frozen constants match the prereg",
        DELTA == 0.00361 and S1_TOL == 0.0068 and S2_TOL == 0.6 and S4_TOL == 0.0136
        and len(PRED) == 10)
    print(f"\n  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    d = os.path.join(ROOT, "scratchpad", "phaseR", "r237")
    score(R.load(os.path.join(d, "csv")), R.arm_totals(d))
