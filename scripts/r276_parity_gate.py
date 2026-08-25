#!/usr/bin/env python
"""[R.276] The FourierFT stock-vs-merged PARITY GATE — scores [R.237]/[R.263] P8 mechanically.

⛔ WHY THIS IS THE MOST CONSEQUENTIAL CELL IN THE GRID.  Every headline in this session --
CONTEXT 2's tuned FourierFT 0.7942/0.8195, [R.257]'s bit-identical drift check, the whole
tuned-baseline table -- is measured on the MERGED implementation (`adamw-fourierftmerged`).
CARRY_FORWARD 1.2 asserts merged == stock == fast at 0% accuracy cost.  These two cells are
the only end-to-end test of that assertion at this protocol.  A failure would put every
`fftm` number in question, which is why it is gated rather than eyeballed.

P8 (frozen, R237_baseline_grid_prereg.md 71): |stock - merged| <= 0.0036 = ONE eval example.

PREMISE CHECK (this file, run BEFORE the cells landed): the two arms must draw the SAME
support, or a parity failure would be trivial rather than informative.
  [measured] peft_indices(768,768,256,777) is BITWISE IDENTICAL to peft 0.13.2's own
  FourierFTLayer.update_layer draw.  Fixture G1 below re-checks it on every run.

Usage:  env/bin/python scripts/r276_parity_gate.py [--selftest]
"""
import argparse, importlib.util, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_s = importlib.util.spec_from_file_location("r237_read", os.path.join(ROOT, "scripts", "r237_read.py"))
R = importlib.util.module_from_spec(_s); _s.loader.exec_module(R)

TOL = 1.0 / 277.0                      # one RTE eval example, P8's frozen threshold
TWINS = [("fftstock-lr5e-2", "fftm-lr5e-2-s150"),
         ("fftstock-lr1.5e-2", "fftm-lr1.5e-2-s150")]


def support_matches(m=768, n=768, k=256, seed=777):
    """Do the merged arm and installed PEFT draw the same support?  The gate's premise."""
    import torch
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from fourierft_fast import peft_indices
    ours = peft_indices(m, n, k, seed).long()
    g = torch.Generator(); g.manual_seed(seed)
    flat = torch.randperm(m * n, generator=g)[:k]
    theirs = torch.stack([flat // n, flat % n], dim=0).long()
    return bool(torch.equal(ours, theirs))


def report(cells, stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    p("=" * 78)
    p("[R.276] FourierFT PARITY GATE -- stock vs merged.  P8: |delta| <= 0.0036 (1 eval example)")
    p("=" * 78)
    okp = support_matches()
    p(f"  premise -- identical support draw: {'✅ YES (bitwise)' if okp else '⛔ NO'}")
    if not okp:
        p("  ⛔ PREMISE FAILED: the arms do not share a support, so any parity result is")
        p("     uninformative about the implementation.  GATE VOID.")
        return {"verdict": "VOID"}
    res, missing = [], []
    for a, b in TWINS:
        ca, cb = cells.get(a), cells.get(b)
        if ca is None or cb is None:
            missing.append((a, b)); continue
        d = ca["acc"] - cb["acc"]
        res.append((a, b, ca["acc"], cb["acc"], d))
        p(f"  {a:18s} {ca['acc']:.4f}   vs {b:20s} {cb['acc']:.4f}   "
          f"delta {d:+.4f} ({d*277:+.1f} ex)  {'✅' if abs(d) <= TOL + 1e-12 else '⛔ EXCEEDS P8'}")
    for a, b in missing:
        p(f"  {a:18s} ⏳ not yet run (twin {b})")
    if missing or not res:
        p(f"\n  ⛔ INCOMPLETE -- {len(missing)}/{len(TWINS)} twin pair(s) missing.  NO VERDICT (PROCESS.md 1.4)")
        return {"verdict": "INCOMPLETE", "missing": len(missing)}
    worst = max(abs(r[4]) for r in res)
    ok = worst <= TOL + 1e-12
    p(f"\n  worst |delta| = {worst:.4f} ({worst*277:.1f} eval examples)")
    p(f"  P8: {'✅ PASS -- stock == merged at this protocol' if ok else '⛔⛔ FAILED'}")
    if not ok:
        p("  ⛔⛔ CARRY_FORWARD 1.2's 'merged == stock, 0% accuracy cost' does NOT hold here.")
        p("     Every fftm number in this session is measured on the merged path and must be")
        p("     re-qualified before use.  Escalate; do not paper over.")
    return {"verdict": "PASS" if ok else "FAIL", "worst": worst}


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    import io
    chk("G1 ⭐ premise: merged and installed PEFT draw a BITWISE IDENTICAL support",
        support_matches() is True)
    H = lambda a: {"acc": a, "collapsed": False, "near_floor": False}
    exact = {"fftstock-lr5e-2": H(0.7545), "fftm-lr5e-2-s150": H(0.7545),
             "fftstock-lr1.5e-2": H(0.7076), "fftm-lr1.5e-2-s150": H(0.7076)}
    buf = io.StringIO(); r = report(exact, buf)
    chk("G2 exact agreement PASSES", r["verdict"] == "PASS", str(r))
    one = dict(exact); one["fftstock-lr5e-2"] = H(0.7545 + 1.0 / 277)
    buf = io.StringIO(); r = report(one, buf)
    chk("G3 exactly ONE eval example still passes (the frozen threshold is inclusive)",
        r["verdict"] == "PASS", str(r))
    two = dict(exact); two["fftstock-lr5e-2"] = H(0.7545 + 2.0 / 277)
    buf = io.StringIO(); r = report(two, buf)
    chk("G4 TWO eval examples FAILS", r["verdict"] == "FAIL", str(r))
    chk("G4b and the failure escalates rather than hedging",
        "must be" in buf.getvalue() and "re-qualified" in buf.getvalue())
    buf = io.StringIO(); r = report({"fftm-lr5e-2-s150": H(0.7545)}, buf)
    chk("G5 a missing twin yields NO VERDICT, not a pass",
        r["verdict"] == "INCOMPLETE" and "NO VERDICT" in buf.getvalue(), str(r))
    print(f"\n  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    report(R.load(os.path.join(ROOT, "scratchpad", "phaseR", "r237", "csv")))
